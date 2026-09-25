"""
Unit tests for the exact official Macro F_0.5 Scorer.
"""

import pytest
from entity_resolution.scorer import entity_f05, macro_f05, _parse_ids, detailed_evaluation


def test_entity_f05_empty_t_empty_s():
    """Rule 1: If T is empty and S is empty -> 1.0"""
    assert entity_f05(set(), set()) == 1.0


def test_entity_f05_empty_t_nonempty_s():
    """Rule 2: If T is empty and S is nonempty -> 0.0"""
    assert entity_f05(set(), {"S2-100"}) == 0.0


def test_entity_f05_nonempty_t_empty_s():
    """Rule 3: If T is nonempty and S is empty -> 0.0"""
    assert entity_f05({"S2-100"}, set()) == 0.0


def test_entity_f05_perfect_match():
    """Rule 4: Perfect match -> 1.0"""
    assert entity_f05({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0


def test_entity_f05_readme_example():
    """
    Official README example:
    Predicted = [S2-00047, S2-00193, S3-00812]  (len S = 3)
    Truth     = [S2-00047, S3-00812]             (len T = 2)
    TP = 2
    F_0.5 = 1.25 * 2 / (0.25 * 2 + 3) = 2.5 / 3.5 = 5/7 ≈ 0.7142857
    """
    t = {"S2-00047", "S3-00812"}
    s = {"S2-00047", "S2-00193", "S3-00812"}
    score = entity_f05(t, s)
    expected = 2.5 / 3.5
    assert abs(score - expected) < 1e-6


def test_parse_ids():
    assert _parse_ids("S2-1, S3-2, ") == {"S2-1", "S3-2"}
    assert _parse_ids("") == set()
    assert _parse_ids(None) == set()
    assert _parse_ids("nan") == set()
    assert _parse_ids(["S2-5", "S3-6"]) == {"S2-5", "S3-6"}


def test_macro_f05():
    gt = {
        "S1-1": {"S2-1", "S3-2"},
        "S1-2": set(),  # singleton
    }
    # Case 1: both correct
    pred_perfect = {
        "S1-1": {"S2-1", "S3-2"},
        "S1-2": set(),
    }
    assert macro_f05(gt, pred_perfect) == 1.0

    # Case 2: S1-2 falsely predicted nonempty
    pred_bad_singleton = {
        "S1-1": {"S2-1", "S3-2"},
        "S1-2": {"S2-99"},
    }
    # S1-1 = 1.0, S1-2 = 0.0 -> average = 0.5
    assert macro_f05(gt, pred_bad_singleton) == 0.5
