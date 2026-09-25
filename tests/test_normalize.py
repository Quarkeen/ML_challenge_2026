"""
Unit tests for multi-representation normalization and address parsing.
"""

from entity_resolution.normalize import (
    unicode_normalize,
    punct_normalize,
    alphanum_only,
    strip_legal_suffix,
    compressed_name,
    parse_address_fields,
    extract_postal_code,
)


def test_unicode_and_punct_normalize():
    assert unicode_normalize("Café Léarning") == "Cafe Learning"
    assert punct_normalize("Avan & Brothers, Inc.") == "avan and brothers inc"
    assert alphanum_only("Quick-Fix #12!") == "quickfix12"


def test_strip_legal_suffix():
    stripped, suffix = strip_legal_suffix("Empire Alliance LLC")
    assert stripped == "empire alliance"
    assert suffix.lower() == "llc"

    stripped_in, suffix_in = strip_legal_suffix("Porur Vyapar Private Limited")
    assert "private limited" in suffix_in.lower() or "limited" in suffix_in.lower()
    assert "porur vyapar" in stripped_in.lower()


def test_compressed_domain_name():
    assert compressed_name("moorebitwise.com") == "moorebitwise"
    assert compressed_name("https://www.company.in") == "company"
    assert compressed_name("Optimal Data Enterprises, Inc.") == "optimaldataenterprises"


def test_parse_address_fields():
    # US address
    us_parsed = parse_address_fields("4204 Applegate Lane, Suitland, MD 20746", country="US")
    assert us_parsed["house_number"] == "4204"
    assert us_parsed["postal_code"] == "20746"
    assert us_parsed["is_missing"] is False

    # India address
    in_parsed = parse_address_fields("Door No 183, 41St Cross, Bengaluru 560078", country="India")
    assert in_parsed["house_number"] == "183"
    assert in_parsed["postal_code"] == "560078"

    # Missing address
    empty_parsed = parse_address_fields("", country="US")
    assert empty_parsed["is_missing"] is True
