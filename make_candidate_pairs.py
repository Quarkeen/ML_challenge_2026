"""
Blocking-only reconstruction of output/candidate_pairs.tsv.

Reuses the exact same CandidateRetriever indexing/retrieval call path,
parameters, and S1 preprocessing/batching semantics as
entity_resolution/predict.py (Step 2 "Indexing Test S2 and S3 records" and
the retrieval portion of Step 3), without loading the trained model or
computing pairwise features. Output format and row order are byte-for-byte
identical to what predict.py writes to output/candidate_pairs.tsv.

IMPORTANT: candidate order at the top-`max_candidates_per_entity` cutoff can
depend on Python's string-hash-seeded set/dict iteration order (via
Counter.most_common tie-breaking). Run this with the SAME PYTHONHASHSEED used
(or intended) for the predict.py run that produced output/matching_results.tsv
(e.g. `PYTHONHASHSEED=0`) for a byte-identical candidate_pairs.tsv.

This script does NOT modify output/matching_results.tsv and does NOT insert
matched IDs into the candidate lists it writes.
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from entity_resolution.config import DEFAULT_CONFIG, TEST_S1, TEST_S2, TEST_S3, OUTPUT_DIR
from entity_resolution.blocking import CandidateRetriever


def generate_candidate_pairs(
    output_dir=None,
    s1_path=None,
    s2_path=None,
    s3_path=None,
    batch_size: int = 50000,
    work_batch_size: int = 5000,
) -> Path:
    """
    Build candidate_pairs.tsv using exactly the same indexing and retrieval
    calls predict.py makes, skipping model loading and feature generation.
    """
    t_start = time.time()
    if batch_size <= 0 or work_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    out_dir = Path(output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_file = out_dir / "candidate_pairs.tsv"

    p_s1 = Path(s1_path or TEST_S1)
    p_s2 = Path(s2_path or TEST_S2)
    p_s3 = Path(s3_path or TEST_S3)

    print("============================================================")
    print("Blocking-only Candidate Pair Reconstruction")
    print(f"Test S1: {p_s1}")
    print(f"Candidate Pairs Output: {candidate_file}")
    print("============================================================")

    # Build candidate indexes from Test S2 and S3, identical to predict.py's
    # Step 2 (same chunk size, same assume_strings=True path, same
    # finalize_indexes(release_accumulators=True) call).
    print("\n[Step 1/2] Indexing Test S2 and S3 records...")
    indexing_start = time.perf_counter()
    retriever = CandidateRetriever(config=DEFAULT_CONFIG)

    for p, name in [(p_s2, "Test Source 2"), (p_s3, "Test Source 3")]:
        print(f"  Indexing {name} ({p})...")
        for chunk in pd.read_csv(p, sep="\t", chunksize=DEFAULT_CONFIG["chunk_size"], dtype=str, keep_default_na=False):
            retriever.index_chunk(chunk, assume_strings=True)
    retriever.finalize_indexes(release_accumulators=True)
    indexing_time = time.perf_counter() - indexing_start
    print(f"  S2/S3 indexing: {indexing_time:.1f}s")

    # Stream S1 with the same two-level batching predict.py uses: an outer
    # read chunksize of --batch-size, subdivided into --work-batch-size rows
    # per retrieval call, to keep memory use comparable.
    print("\n[Step 2/2] Streaming Test S1 records and retrieving candidates...")

    def work_batches():
        for read_chunk in pd.read_csv(p_s1, sep="\t", chunksize=batch_size, dtype=str, keep_default_na=False):
            for start in range(0, len(read_chunk), work_batch_size):
                yield read_chunk.iloc[start:start + work_batch_size]

    total_s1_processed = 0
    with open(candidate_file, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for batch_number, s1_chunk in enumerate(work_batches(), 1):
            batch_start = time.perf_counter()
            chunk_s1_ids = s1_chunk["entity_id"].tolist()

            cands_map = retriever.retrieve_candidates(
                s1_chunk, max_candidates=DEFAULT_CONFIG["max_candidates_per_entity"], assume_strings=True,
            )

            candidate_lines = []
            for s1_id in chunk_s1_ids:
                c_dict = cands_map.get(s1_id, {})
                cands_list = list(c_dict.keys())
                cand_str = ",".join(cands_list) if cands_list else ""
                candidate_lines.append(f"{s1_id}\t{cand_str}\n")
            f_cand.writelines(candidate_lines)
            del cands_map, candidate_lines

            total_s1_processed += len(chunk_s1_ids)
            batch_elapsed = time.perf_counter() - batch_start
            print(f"  Batch {batch_number}: {len(chunk_s1_ids):,} S1 in {batch_elapsed:.1f}s "
                  f"({len(chunk_s1_ids) / max(batch_elapsed, 1e-9):,.0f}/s)")

    print(f"\nTotal S1 rows written: {total_s1_processed:,}")
    print(f"Candidate Pairs: {candidate_file.resolve()}")
    print(f"Completed in {time.time() - t_start:.1f}s.")
    return candidate_file


def _read_id_order(path: Path, id_col: str) -> List[str]:
    """Read just the first column of a TSV in file order (streaming, low memory)."""
    ids: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        assert header[0] == id_col, f"Unexpected header in {path}: {header}"
        for row in reader:
            if row:
                ids.append(row[0])
    return ids


def _read_id_to_list(path: Path, list_col_index: int = 1) -> Dict[str, List[str]]:
    """Read {s1_id: [comma-separated list entries]} from a two-column TSV, streaming."""
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            raw = row[list_col_index] if len(row) > list_col_index else ""
            out[s1_id] = raw.split(",") if raw else []
    return out


def compare_with_matching_results(candidate_file: Path, matching_file: Path) -> None:
    """
    Report (a) whether S1 ID sets/order match between candidate_pairs.tsv and
    matching_results.tsv, and (b) matched IDs that are not a subset of that
    row's candidates (subset violations), with up to 10 examples.

    Read-only: does not modify matching_results.tsv and does not alter
    candidate lists.
    """
    if not matching_file.exists():
        print(f"\n[Compare] {matching_file} not found; skipping comparison.")
        return

    print(f"\n[Compare] Reading {candidate_file.name} and {matching_file.name}...")
    cand_ids = _read_id_order(candidate_file, "source1_entity_id")
    match_ids = _read_id_order(matching_file, "source1_entity_id")

    same_length = len(cand_ids) == len(match_ids)
    same_order = cand_ids == match_ids
    same_set = set(cand_ids) == set(match_ids)

    print(f"  candidate_pairs.tsv S1 rows: {len(cand_ids):,}")
    print(f"  matching_results.tsv S1 rows: {len(match_ids):,}")
    print(f"  Same row count: {same_length}")
    print(f"  Same ID order: {same_order}")
    print(f"  Same ID set (order-independent): {same_set}")
    if not same_set:
        only_in_cand = set(cand_ids) - set(match_ids)
        only_in_match = set(match_ids) - set(cand_ids)
        print(f"  IDs only in candidate_pairs.tsv: {len(only_in_cand):,} (e.g. {list(only_in_cand)[:5]})")
        print(f"  IDs only in matching_results.tsv: {len(only_in_match):,} (e.g. {list(only_in_match)[:5]})")

    print("\n[Compare] Checking matched IDs are a subset of candidates per row...")
    candidates_by_id = _read_id_to_list(candidate_file, list_col_index=1)
    matches_by_id = _read_id_to_list(matching_file, list_col_index=1)

    violation_count = 0
    examples = []
    for s1_id, matched_list in matches_by_id.items():
        if not matched_list:
            continue
        cand_set = set(candidates_by_id.get(s1_id, []))
        bad = [m for m in matched_list if m not in cand_set]
        if bad:
            violation_count += 1
            if len(examples) < 10:
                examples.append((s1_id, bad, sorted(cand_set)[:10]))

    print(f"  Rows with matched IDs NOT present in that row's candidates: {violation_count:,}")
    for s1_id, bad, cand_sample in examples:
        print(f"    {s1_id}: missing_from_candidates={bad} candidates_sample={cand_sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct candidate_pairs.tsv via blocking-only reproduction of predict.py")
    parser.add_argument("--test-dir", type=Path, default=None,
                         help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv "
                              "(defaults to the same TEST_DIR entity_resolution.config resolves)")
    parser.add_argument("--batch-size", type=int, default=50000, help="Outer S1 read chunk size (matches predict.py default)")
    parser.add_argument("--work-batch-size", type=int, default=5000, help="S1 rows per retrieval call (matches predict.py default)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for candidate_pairs.tsv (defaults to OUTPUT_DIR)")
    parser.add_argument("--matching", type=Path, default=None,
                         help="Path to matching_results.tsv to compare against (default: <output>/matching_results.tsv)")
    args = parser.parse_args()

    if args.test_dir is not None:
        s1_path = args.test_dir / "test_source1.tsv"
        s2_path = args.test_dir / "test_source2.tsv"
        s3_path = args.test_dir / "test_source3.tsv"
    else:
        s1_path = s2_path = s3_path = None

    out_path = generate_candidate_pairs(
        output_dir=args.output,
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        batch_size=args.batch_size,
        work_batch_size=args.work_batch_size,
    )

    matching_path = args.matching or (Path(args.output or OUTPUT_DIR) / "matching_results.tsv")
    compare_with_matching_results(out_path, matching_path)
