import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (
    raw_name, unicode_normalize, lowercase_normalize, punct_normalize,
    alphanum_only, tokenize, strip_legal_suffix, name_tokens_set,
    normalize_address, parse_address_fields, address_tokens, numeric_tokens,
    rare_address_tokens, normalize_dataframe
)

def test_name_normalization():
    assert lowercase_normalize("HELLO  WORLD") == "hello world"
    assert punct_normalize("A & B Co., Ltd!") == "a and b co ltd"
    assert alphanum_only("A & B Co., Ltd!") == "a b co ltd"
    assert tokenize("A & B Co., Ltd!") == ["a", "b", "co", "ltd"]
    
    stripped, unstripped = strip_legal_suffix("Apple Inc.")
    assert stripped == "apple"
    assert unstripped == "apple inc."

    stripped, unstripped = strip_legal_suffix("Reliance Pvt Ltd")
    assert stripped == "reliance"
    
    stripped, unstripped = strip_legal_suffix("Café Paris SARL")
    # unicode normalized by lowercase_normalize
    assert stripped == "cafe paris"

def test_address_normalization():
    assert normalize_address("123 Main St.") == "123 main street." # St. -> street, . is kept unless alphanum used
    
    us_addr = parse_address_fields("1234 Elm Street NY 10001", "US")
    assert us_addr['house_number'] == "1234"
    assert us_addr['postal_code'] == "10001"
    
    in_addr = parse_address_fields("Plot 5, Sector 12, Delhi 110001", "IN")
    assert in_addr['postal_code'] == "110001"
    
    fr_addr = parse_address_fields("10 Rue de la Paix, 75001 Paris", "FR")
    assert fr_addr['postal_code'] == "75001"

    assert numeric_tokens("10 Rue 75001") == ["10", "75001"]

def test_dataframe_normalization():
    df = pd.DataFrame({
        'business_name': ['Apple Inc.', 'Banana Corp', None],
        'business_address': ['123 Main St NY 10001', '456 Market Blvd', ''],
        'country': ['US', 'US', 'US']
    })
    
    res = normalize_dataframe(df)
    assert len(res) == 3
    assert res['name_no_suffix'].iloc[0] == 'apple'
    assert res['postal_code'].iloc[0] == '10001'
    assert res['is_addr_missing'].iloc[2] == True
    assert res['is_name_missing'].iloc[2] == True
