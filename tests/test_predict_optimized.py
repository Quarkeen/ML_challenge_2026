"""Regression checks for inference-only feature preparation and selection."""

import numpy as np
import pandas as pd

from entity_resolution.blocking import CandidateRetriever
from entity_resolution.features import FeatureGenerator, PREPARED_FEATURE_NAMES, prepare_record
from entity_resolution.normalize import (
    compressed_name, prepared_name_signatures, punct_normalize,
    strip_legal_suffix, stripped_from_punct,
)
from entity_resolution.selection import SetSelector


def test_prepared_features_match_reference():
    fg = FeatureGenerator()
    rows = [
        ("S2-1", "Atlas Logistics LLC", "500 Market St, Denver, CO 80202", "US"),
        ("S3-2", "atlaslogistics.com", "500 Market Street, Denver 80202", "USA"),
        ("S2-3", "Bharat Bio Tech Pvt Ltd", "Plot 24, Hyderabad 500081", "India"),
        ("S3-4", "Café & Co", "", "FR"),
        ("S2-5", "", "none", ""),
    ]
    for s1_id, s1_name, s1_address, s1_country in rows:
        s1 = {"entity_id": s1_id, "business_name": s1_name,
              "business_address": s1_address, "country": s1_country}
        for cid, name, address, country in rows:
            candidate = {"entity_id": cid, "business_name": name,
                         "business_address": address, "country": country}
            meta = {"retrieval_score": 14.5,
                    "retrieval_methods": "exact_name|comp_name|token|addr"}
            expected = fg.compute_pair_features(s1, candidate, meta)
            actual = fg.compute_prepared_pair_features(
                prepare_record(s1_name, s1_address, s1_country),
                prepare_record(name, address, country), cid, meta,
            )
            assert expected == actual
            values = fg.compute_prepared_pair_values(
                prepare_record(s1_name, s1_address, s1_country),
                prepare_record(name, address, country), cid, meta,
            )
            assert values == tuple(expected[key] for key in PREPARED_FEATURE_NAMES)


def test_reused_name_normalization_matches_reference():
    for name in ("", "Café & Co", "Atlas Logistics LLC", "www.atlas.com",
                 "Bharat Bio Tech Pvt Ltd", "Müller + Söhne GmbH", "भारत प्राइवेट लिमिटेड"):
        normalized, compressed = prepared_name_signatures(name)
        assert normalized == punct_normalize(name)
        assert compressed == compressed_name(name)
        assert stripped_from_punct(normalized) == strip_legal_suffix(name)


def test_feature_worker_block_matches_reference():
    import entity_resolution.predict as prediction

    original_positions = prediction._WORKER_FEATURE_POSITIONS
    original_prepared = prediction._WORKER_PREPARED
    original_group_records = prediction._WORKER_GROUP_RECORDS
    try:
        records = {
            "S2-1": ("Atlas Logistics LLC", "500 Market St", "US"),
            "S3-2": ("Atlas Logistics", "500 Market Street", "US"),
        }
        prediction._init_feature_worker(10, tuple(range(len(PREPARED_FEATURE_NAMES))))
        meta = {"retrieval_score": 12.0, "retrieval_methods": "exact_name|token"}
        candidates = {"S2-1": meta, "S3-2": meta}
        block = prediction._feature_group(([
            ("S1-1", ("Atlas Logistics", "500 Market St", "US"), candidates),
        ], records))
        fg = FeatureGenerator()
        source = {"business_name": "Atlas Logistics", "business_address": "500 Market St", "country": "US"}
        expected = [
            fg.compute_pair_features(source, {
                "entity_id": cid, "business_name": name, "business_address": address, "country": country,
            }, meta)
            for cid, (name, address, country) in records.items()
        ]
        np.testing.assert_array_equal(
            block, np.asarray([[row[key] for key in PREPARED_FEATURE_NAMES] for row in expected], dtype=np.float32),
        )
    finally:
        prediction._WORKER_FEATURE_POSITIONS = original_positions
        prediction._WORKER_PREPARED = original_prepared
        prediction._WORKER_GROUP_RECORDS = original_group_records


def test_spawned_worker_receives_only_group_records():
    from multiprocessing import get_context
    import entity_resolution.predict as prediction

    records = {"S2-1": ("Atlas Logistics LLC", "500 Market St", "US")}
    meta = {"retrieval_score": 12.0, "retrieval_methods": "exact_name"}
    job = ([("S1-1", ("Atlas Logistics", "500 Market St", "US"), {"S2-1": meta})], records)
    with get_context("spawn").Pool(
        2, initializer=prediction._init_feature_worker,
        initargs=(10, tuple(range(len(PREPARED_FEATURE_NAMES)))),
    ) as pool:
        blocks = pool.map(prediction._feature_group, [job, job])
    assert len(blocks) == 2
    np.testing.assert_array_equal(blocks[0], blocks[1])


def test_string_input_fast_path_preserves_candidates():
    candidates = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Atlas Logistics LLC",
         "business_address": "500 Market St", "country": "US"},
        {"entity_id": "S3-2", "business_name": "Atlas Logistics",
         "business_address": "500 Market Street", "country": "US"},
    ])
    source = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Atlas Logistics",
         "business_address": "500 Market St", "country": "US"},
    ])
    reference = CandidateRetriever()
    reference.index_chunk(candidates)
    reference.finalize_indexes()
    optimized = CandidateRetriever()
    optimized.index_chunk(candidates, assume_strings=True)
    optimized.finalize_indexes()
    assert optimized.retrieve_candidates(source, assume_strings=True) == reference.retrieve_candidates(source)


def test_scored_slices_match_reference_selector():
    selector = SetSelector(match_threshold=0.5, empty_threshold=0.35,
                           relative_gap=0.2, max_matches=2)
    candidates = {"S1-1": ["S2-1", "S3-2", "S2-3"], "S1-2": [],
                  "S1-3": ["S3-4", "S2-5"]}
    scores = np.array([0.91, 0.81, 0.19, 0.34, 0.9], dtype=np.float32)
    reference = pd.DataFrame([
        ("S1-1", "S2-1", scores[0]), ("S1-1", "S3-2", scores[1]),
        ("S1-1", "S2-3", scores[2]), ("S1-3", "S3-4", scores[3]),
        ("S1-3", "S2-5", scores[4]),
    ], columns=["source1_entity_id", "candidate_id", "score"])
    expected = selector.predict_sets(reference, all_s1_ids=list(candidates))
    actual = {sid: set() for sid in candidates}
    offset = 0
    for sid, ids in candidates.items():
        if ids:
            actual[sid] = set(selector.select_for_entity(zip(ids, scores[offset:offset + len(ids)])))
        offset += len(ids)
    assert actual == expected
