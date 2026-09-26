"""
End-to-End Inference and Submission Generator.
Streams test records to ensure strict OOM safety.
Outputs exact required TSV format:
    source1_entity_id\tmatched_entity_ids
Every Source 1 ID appears exactly once, with empty strings for singletons.
"""

import argparse
import json
import multiprocessing as mp
import subprocess
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_CONFIG,
    TEST_S1,
    TEST_S2,
    TEST_S3,
    OUTPUT_DIR,
    MODELS_DIR,
    detect_device,
)
from .blocking import CandidateRetriever
from .features import FeatureGenerator, PREPARED_FEATURE_NAMES, prepare_record
from .model import PairwiseRanker
from .selection import SetSelector


_WORKER_FEATURE_POSITIONS = None
_WORKER_PREPARED = None
_WORKER_GROUP_RECORDS = None


def _init_feature_worker(cache_size: int, feature_positions: Tuple[int, ...]) -> None:
    """Initialize a worker without loading or reconstructing the candidate pool."""
    global _WORKER_PREPARED, _WORKER_FEATURE_POSITIONS
    _WORKER_FEATURE_POSITIONS = feature_positions

    @lru_cache(maxsize=cache_size)
    def prepared(cid: str):
        return prepare_record(*_WORKER_GROUP_RECORDS[cid])

    _WORKER_PREPARED = prepared


def _feature_group(job):
    """Build a contiguous feature block from only the group's candidate rows."""
    global _WORKER_GROUP_RECORDS
    items, _WORKER_GROUP_RECORDS = job
    positions = _WORKER_FEATURE_POSITIONS
    native_order = positions == tuple(range(len(PREPARED_FEATURE_NAMES)))
    matrix = np.empty((sum(len(candidates) for _, _, candidates in items), len(positions)), dtype=np.float32)
    row = 0
    for _, raw_s1, candidates in items:
        if not candidates:
            continue
        s1_prepared = prepare_record(*raw_s1)
        for cid, info in candidates.items():
            values = FeatureGenerator.compute_prepared_pair_values(
                s1_prepared, _WORKER_PREPARED(cid), cid, info,
            )
            matrix[row] = values if native_order else tuple(values[i] for i in positions)
            row += 1
    _WORKER_GROUP_RECORDS = None
    return matrix


def generate_predictions(
    model_path: Optional[Path] = None,
    policy_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    s1_path: Optional[Path] = None,
    s2_path: Optional[Path] = None,
    s3_path: Optional[Path] = None,
    batch_size: int = 50000,
    device: str = "auto",
    profile: bool = False,
    limit: Optional[int] = None,
    candidate_cache_size: int = 100000,
    work_batch_size: int = 5000,
    num_workers: int = 1,
) -> Tuple[Path, Path]:
    """
    Generate predictions for test dataset and save official submission TSV.
    """
    t_start = time.time()
    if batch_size <= 0 or work_batch_size <= 0 or num_workers <= 0 or (limit is not None and limit < 0) or candidate_cache_size < 0:
        raise ValueError("batch sizes and num_workers must be positive; limit and cache size cannot be negative")
    m_path = Path(model_path or (MODELS_DIR / "pairwise_ranker.xgboost"))
    if not m_path.exists():
        # Try lightgbm alternative
        m_alt = m_path.with_suffix(".lightgbm")
        if m_alt.exists():
            m_path = m_alt

    pol_path = Path(policy_path or (MODELS_DIR / "selection_policy.json"))
    out_dir = Path(output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    p_s1 = Path(s1_path or TEST_S1)
    p_s2 = Path(s2_path or TEST_S2)
    p_s3 = Path(s3_path or TEST_S3)

    print(f"============================================================")
    print(f"Starting Entity Resolution Inference")
    print(f"Model: {m_path}")
    print(f"Test S1: {p_s1}")
    print(f"Matching Results Output:  {matching_file}")
    print(f"Candidate Pairs Output:   {candidate_file}")
    print(f"============================================================")

    # 1. Load Model and Policy
    print("\n[Step 1/4] Loading trained model and selection policy...")
    ranker = PairwiseRanker.load(m_path)
    ranker.device = detect_device(device)
    inference_device = ranker.device if ranker.model_type == "xgboost" else "cpu"
    if ranker.model_type == "xgboost":
        # load() restores the trees, but assigning ranker.device alone never
        # configures the loaded Booster. DMatrix prediction then uses its default.
        ranker.model.set_param({"device": ranker.device})
    print(f"  Model loaded. Inference device: {inference_device}")
    if ranker.model_type == "lightgbm" and ranker.device == "cuda":
        print("  LightGBM Booster.predict scores on CPU; CUDA is a training device setting.")
    try:
        feature_positions = tuple(PREPARED_FEATURE_NAMES.index(name) for name in ranker.feature_names)
    except ValueError as exc:
        raise ValueError("Trained model requests an unknown inference feature") from exc
    native_feature_order = feature_positions == tuple(range(len(PREPARED_FEATURE_NAMES)))
    checked_xgb_device = False
    if ranker.model_type == "xgboost" and device == "cuda":
        # Fail before indexing ~10M records if this host cannot actually score
        # on CUDA. XGBoost can silently fall back to CPU on its first predict.
        ranker.predict_proba(np.zeros((1, len(ranker.feature_names)), dtype=np.float32))
        effective_device = json.loads(ranker.model.save_config())["learner"]["generic_param"]["device"]
        print(f"  XGBoost effective prediction device: {effective_device}")
        if not effective_device.startswith("cuda"):
            raise RuntimeError("--device cuda requested, but XGBoost fell back to CPU")
        checked_xgb_device = True

    selector_params = DEFAULT_CONFIG.copy()
    if pol_path.exists():
        with open(pol_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            selector_params.update(meta.get("thresholds", {}))
    selector = SetSelector(
        match_threshold=selector_params.get("match_threshold", 0.50),
        empty_threshold=selector_params.get("empty_threshold", 0.35),
        relative_gap=selector_params.get("relative_score_gap", 0.20),
        max_matches=selector_params.get("max_matches_per_entity", 10),
    )
    print(f"  Selector parameters: match_th={selector.match_threshold}, empty_th={selector.empty_threshold}")

    peak_ram_mb = 0.0
    try:
        import psutil
        process = psutil.Process()
        ram_metric = "PSS" if hasattr(process.memory_full_info(), "pss") else "RSS"
    except ImportError:
        process = None
        ram_metric = "RAM"

    def sample_ram():
        nonlocal peak_ram_mb
        if process is not None:
            total_bytes = 0
            for member in [process, *process.children(recursive=True)]:
                try:
                    info = member.memory_full_info() if ram_metric == "PSS" else member.memory_info()
                    total_bytes += info.pss if ram_metric == "PSS" else info.rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak_ram_mb = max(peak_ram_mb, total_bytes / (1024 * 1024))

    # 2. Build Candidate Indexes from Test S2 and S3 (streaming accumulators, no
    # per-chunk index resets, and no DataFrame round-trip for candidate metadata)
    print("\n[Step 2/4] Indexing Test S2 and S3 records...")
    indexing_start = time.perf_counter()
    retriever = CandidateRetriever(config=DEFAULT_CONFIG)
    # One compact, reusable lookup. The blocking indexes hold candidate IDs;
    # feature generation fetches the raw fields only for retrieved IDs.
    cand_records: Dict[str, Tuple[str, str, str]] = {}

    for p, name in [(p_s2, "Test Source 2"), (p_s3, "Test Source 3")]:
        print(f"  Indexing {name} ({p})...")
        for chunk in pd.read_csv(p, sep="\t", chunksize=DEFAULT_CONFIG["chunk_size"], dtype=str, keep_default_na=False):
            retriever.index_chunk(chunk, assume_strings=True)
            ids = chunk["entity_id"].values
            names = chunk["business_name"].values
            addrs = chunk["business_address"].values
            countries = chunk["country"].values if "country" in chunk.columns else [""] * len(chunk)
            for cid, n, a, c in zip(ids, names, addrs, countries):
                cand_records[cid] = (n, a, c)
            sample_ram()
        chunk = None
    retriever.finalize_indexes(release_accumulators=True)
    indexing_time = time.perf_counter() - indexing_start
    print(f"  Candidate pool indexed. Total unique candidates: {len(cand_records):,}")
    print(f"  S2/S3 indexing: {indexing_time:.1f}s")

    feature_pool = None
    if num_workers > 1:
        # Spawned workers receive only records referenced by a work group.
        # The 10M-record lookup and the blocking indexes remain in the parent.
        feature_pool = mp.get_context("spawn").Pool(
            processes=num_workers,
            initializer=_init_feature_worker,
            initargs=(candidate_cache_size // num_workers, feature_positions),
        )

    # 3. Stream through Test S1 in Batches & Write Both Output TSVs
    print("\n[Step 3/4] Streaming Test S1 records and predicting matches...")
    fg = FeatureGenerator(config=DEFAULT_CONFIG)

    @lru_cache(maxsize=candidate_cache_size)
    def prepared_candidate(cid: str):
        return prepare_record(*cand_records[cid])

    total_s1_processed = 0
    non_empty_predictions = 0
    stage_totals = {name: 0.0 for name in ("retrieval", "features", "inference", "selection", "writing")}
    peak_gpu_mb = 0
    gpu_utilization = None

    def sample_resources():
        nonlocal peak_gpu_mb, gpu_utilization
        sample_ram()
        if profile and inference_device == "cuda":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True,
                )
                gpu_utilization, used = (int(x.strip()) for x in result.stdout.splitlines()[0].split(",")[:2])
                peak_gpu_mb = max(peak_gpu_mb, used)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass

    sample_resources()

    def work_batches():
        # Keep --batch-size as the input read size, but bound the expensive
        # retrieval map and float32 feature matrix to a smaller working set.
        for read_chunk in pd.read_csv(p_s1, sep="\t", chunksize=batch_size, dtype=str, keep_default_na=False):
            for start in range(0, len(read_chunk), work_batch_size):
                yield read_chunk.iloc[start:start + work_batch_size]

    with (feature_pool or nullcontext()) as active_pool, open(matching_file, "w", encoding="utf-8") as f_match, open(candidate_file, "w", encoding="utf-8") as f_cand:
        if ranker.model_type == "xgboost" and device == "cuda" and not checked_xgb_device:
            ranker.predict_proba(np.zeros((1, len(ranker.feature_names)), dtype=np.float32))
            effective_device = json.loads(ranker.model.save_config())["learner"]["generic_param"]["device"]
            print(f"  XGBoost effective prediction device: {effective_device}")
            if not effective_device.startswith("cuda"):
                raise RuntimeError("--device cuda requested, but XGBoost fell back to CPU")
            checked_xgb_device = True
        # Write exact required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for batch_number, s1_chunk in enumerate(work_batches(), 1):
            if limit is not None:
                remaining = limit - total_s1_processed
                if remaining <= 0:
                    break
                if len(s1_chunk) > remaining:
                    s1_chunk = s1_chunk.iloc[:remaining]
            batch_start = time.perf_counter()
            chunk_s1_ids = s1_chunk["entity_id"].tolist()
            stage_start = time.perf_counter()
            cands_map = retriever.retrieve_candidates(
                s1_chunk, max_candidates=DEFAULT_CONFIG["max_candidates_per_entity"], assume_strings=True,
            )
            retrieval_time = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            # Retrieval IDs are stripped by CandidateRetriever. Keep the original
            # exact-key lookup behavior of compute_batch_features for S1 rows.
            s1_rows = dict(zip(
                chunk_s1_ids,
                zip(s1_chunk["business_name"], s1_chunk["business_address"],
                    s1_chunk["country"] if "country" in s1_chunk else ("" for _ in chunk_s1_ids)),
            ))
            pair_count = sum(len(candidates) for candidates in cands_map.values())
            feature_matrix = np.empty((pair_count, len(ranker.feature_names)), dtype=np.float32)
            if active_pool is None:
                feature_index = 0
                for s1_id, candidates in cands_map.items():
                    if not candidates:
                        continue
                    s1_prepared = prepare_record(*s1_rows.get(s1_id, ("", "", "")))
                    for cid, info in candidates.items():
                        features = fg.compute_prepared_pair_values(
                            s1_prepared, prepared_candidate(cid), cid, info,
                        )
                        feature_matrix[feature_index] = (features if native_feature_order
                                                         else tuple(features[i] for i in feature_positions))
                        feature_index += 1
                assert feature_index == pair_count
            elif pair_count:
                items = [(sid, s1_rows.get(sid, ("", "", "")), candidates)
                         for sid, candidates in cands_map.items()]
                group_size = max(1, (len(items) + num_workers * 2 - 1) // (num_workers * 2))
                def groups():
                    for start in range(0, len(items), group_size):
                        group_items = items[start:start + group_size]
                        group_records = {cid: cand_records[cid]
                                         for _, _, candidates in group_items for cid in candidates}
                        yield group_items, group_records
                feature_index = 0
                for block in active_pool.imap(_feature_group, groups(), chunksize=1):
                    next_index = feature_index + len(block)
                    feature_matrix[feature_index:next_index] = block
                    feature_index = next_index
                assert feature_index == pair_count
            feature_time = time.perf_counter() - stage_start
            del s1_rows

            stage_start = time.perf_counter()
            if pair_count:
                scores = ranker.predict_proba(feature_matrix)
                if ranker.model_type == "xgboost" and not checked_xgb_device:
                    effective_device = json.loads(ranker.model.save_config())["learner"]["generic_param"]["device"]
                    print(f"  XGBoost effective prediction device: {effective_device}")
                    if device == "cuda" and not effective_device.startswith("cuda"):
                        raise RuntimeError("--device cuda requested, but XGBoost fell back to CPU")
                    checked_xgb_device = True
            else:
                scores = ()
            inference_time = time.perf_counter() - stage_start
            del feature_matrix

            stage_start = time.perf_counter()
            chunk_preds = {}
            offset = 0
            for s1_id, candidates in cands_map.items():
                count = len(candidates)
                if count:
                    candidate_scores = zip(candidates.keys(), scores[offset:offset + count])
                    chunk_preds[s1_id] = set(selector.select_for_entity(candidate_scores))
                    offset += count
            assert offset == len(scores)
            selection_time = time.perf_counter() - stage_start
            del scores

            stage_start = time.perf_counter()
            matching_lines = []
            candidate_lines = []
            for s1_id in chunk_s1_ids:
                c_dict = cands_map.get(s1_id, {})
                cands_list = list(c_dict.keys())
                cand_str = ",".join(cands_list) if cands_list else ""
                candidate_lines.append(f"{s1_id}\t{cand_str}\n")

                matched_set = chunk_preds.get(s1_id, set())
                # Enforce official validator constraint: matches must be subset of candidates
                valid_matched = [m for m in sorted(matched_set) if m in c_dict]
                if valid_matched:
                    matched_str = ",".join(valid_matched)
                    non_empty_predictions += 1
                else:
                    matched_str = ""
                matching_lines.append(f"{s1_id}\t{matched_str}\n")
            f_cand.writelines(candidate_lines)
            f_match.writelines(matching_lines)
            writing_time = time.perf_counter() - stage_start
            del matching_lines, candidate_lines, cands_map

            total_s1_processed += len(chunk_s1_ids)
            batch_times = dict(zip(stage_totals, (retrieval_time, feature_time, inference_time, selection_time, writing_time)))
            for name, elapsed in batch_times.items():
                stage_totals[name] += elapsed
            batch_elapsed = time.perf_counter() - batch_start
            sample_resources()
            print(f"  Batch {batch_number}: {len(chunk_s1_ids):,} S1 in {batch_elapsed:.1f}s "
                  f"({len(chunk_s1_ids) / max(batch_elapsed, 1e-9):,.0f}/s); "
                  + " ".join(f"{name}={elapsed:.1f}s" for name, elapsed in batch_times.items()))
            if profile:
                print(f"    Cumulative {total_s1_processed:,} S1: "
                      + " ".join(f"{name}={elapsed:.1f}s" for name, elapsed in stage_totals.items())
                      + f"; cache={prepared_candidate.cache_info() if active_pool is None else 'per-worker'}; {ram_metric} peak="
                      + (f"{peak_ram_mb:.0f} MiB" if process is not None else "unavailable")
                      + (f"; GPU={gpu_utilization}% / {peak_gpu_mb} MiB peak" if gpu_utilization is not None else ""))

    # 4. Summary & Verification
    print("\n[Step 4/4] Validating submission integrity...")
    print(f"  Total S1 rows written: {total_s1_processed:,}")
    print(f"  Non-empty predictions: {non_empty_predictions:,} ({(non_empty_predictions/max(1, total_s1_processed))*100:.2f}%)")
    print(f"  Singletons (empty):    {total_s1_processed - non_empty_predictions:,} ({((total_s1_processed - non_empty_predictions)/max(1, total_s1_processed))*100:.2f}%)")
    print(f"  Matching Results:      {matching_file.resolve()}")
    print(f"  Candidate Pairs:       {candidate_file.resolve()}")
    print(f"  S2/S3 indexing:        {indexing_time:.1f}s")
    print("  S1 stage totals:       " + " ".join(f"{name}={elapsed:.1f}s" for name, elapsed in stage_totals.items()))
    print(f"  End-to-end throughput: {total_s1_processed / max(time.time() - t_start, 1e-9):,.0f} S1/s")
    print(f"  Observed peak process-tree {ram_metric}: {peak_ram_mb:.0f} MiB" if process is not None
          else "  Observed peak RAM: unavailable (install psutil to enable)")
    if gpu_utilization is not None:
        print(f"  GPU last utilization: {gpu_utilization}%; observed peak memory: {peak_gpu_mb} MiB")
    print(f"Inference completed in {time.time() - t_start:.1f}s.")

    return matching_file, candidate_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Official ER Predictions")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--policy-path", type=str, default=None, help="Path to selection policy JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for S1 streaming")
    parser.add_argument("--work-batch-size", type=int, default=5000, help="Maximum S1 rows held in retrieval and feature working buffers")
    parser.add_argument("--num-workers", type=int, default=1, help="CPU feature workers; >1 spawns workers and sends only needed candidate records")
    parser.add_argument("--s1-path", type=Path, default=None)
    parser.add_argument("--s2-path", type=Path, default=None)
    parser.add_argument("--s3-path", type=Path, default=None)
    parser.add_argument("--profile", action="store_true", help="Print cumulative stage timings and resource samples")
    parser.add_argument("--limit", "--max-s1", type=int, default=None, help="Process only the first N S1 rows using full candidate indexes")
    parser.add_argument("--candidate-cache-size", type=int, default=100000, help="Maximum number of prepared candidate records to retain")
    args = parser.parse_args()

    generate_predictions(
        model_path=Path(args.model_path) if args.model_path else None,
        policy_path=Path(args.policy_path) if args.policy_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device=args.device,
        batch_size=args.batch_size,
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        profile=args.profile,
        limit=args.limit,
        candidate_cache_size=args.candidate_cache_size,
        work_batch_size=args.work_batch_size,
        num_workers=args.num_workers,
    )
