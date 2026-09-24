import pytest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer import entity_f05, macro_f05, _parse_ids

def test_entity_f05_empty_t_empty_s():
    assert entity_f05(set(), set()) == 1.0

def test_entity_f05_empty_t_nonempty_s():
    assert entity_f05(set(), {'S2-123'}) == 0.0

def test_entity_f05_perfect_match():
    assert entity_f05({'S2-123', 'S3-456'}, {'S2-123', 'S3-456'}) == 1.0

def test_entity_f05_readme_example():
    truth = {'S2-00047', 'S3-00812'}
    pred = {'S2-00047', 'S2-00193', 'S3-00812'}
    score = entity_f05(truth, pred)
    assert abs(score - (5/7)) < 1e-6
    
def test_partial_match_precision_recall():
    # T = {'A', 'B'}, S = {'A', 'C'}
    truth = {'A', 'B'}
    pred = {'A', 'C'}
    # TP = 1. len(T)=2, len(S)=2. score = 1.25 * 1 / (0.5 + 2) = 1.25 / 2.5 = 0.5
    score = entity_f05(truth, pred)
    assert abs(score - 0.5) < 1e-6

def test_all_empty_baseline():
    gt = {'S1-001': set(), 'S1-002': set()}
    pred = {'S1-001': set(), 'S1-002': set()}
    assert macro_f05(gt, pred) == 1.0

def test_macro_f05():
    gt = {
        'S1-001': set(),
        'S1-002': {'S2-001', 'S3-001'},
        'S1-003': {'S2-002'}
    }
    pred = {
        'S1-001': set(), # 1.0
        'S1-002': {'S2-001', 'S2-00193', 'S3-001'}, # 5/7
        'S1-003': set() # 0.0
    }
    assert abs(macro_f05(gt, pred) - (4/7)) < 1e-6

def test_parse_ids():
    assert _parse_ids('') == set()
    assert _parse_ids(None) == set()
    import numpy as np
    assert _parse_ids(np.nan) == set()
    assert _parse_ids('S2-123, S3-456') == {'S2-123', 'S3-456'}
    assert _parse_ids(' S2-123 ,  S3-456  ') == {'S2-123', 'S3-456'}
