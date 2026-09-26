"""
Multi-representation Normalization & Conservative Address Parsing.
Retains both raw and normalized values without loss of distinguishing information.
"""

import re
import unicodedata
from typing import Dict, Any, List, Set, Tuple
import pandas as pd


LEGAL_SUFFIXES = {
    # English / International
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "pvt", "private", "llp", "gmbh", "sa", "sarl", "plc",
    "lp", "group", "holdings", "enterprises", "services", "solutions",
    # India variations
    "pvt ltd", "private limited", "opc",
}

# Precompiled regex for legal suffix stripping
LEGAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b\.?",
    re.IGNORECASE,
)

# Common domain extensions
DOMAIN_PATTERN = re.compile(
    r"(https?://)?(www\.)?([a-zA-Z0-9\-\.]+)\.(com|in|org|net|co\.in|co|us|gov|edu|io|info|biz|fr)\b",
    re.IGNORECASE,
)


def raw_name(text: Any) -> str:
    """Return raw string value, preserving original formatting."""
    if text is None or pd.isna(text):
        return ""
    return str(text).strip()


def unicode_normalize(text: Any) -> str:
    """Apply NFKD unicode normalization, stripping non-spacing combining characters."""
    raw = raw_name(text)
    if not raw:
        return ""
    nfkd = unicodedata.normalize("NFKD", raw)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def lowercase_normalize(text: Any) -> str:
    """Convert to lowercase and collapse multiple whitespace characters."""
    u = unicode_normalize(text)
    return re.sub(r"\s+", " ", u.lower()).strip()


def punct_normalize(text: Any) -> str:
    """Replace punctuation and special characters with spaces, preserving alphanumeric text."""
    low = lowercase_normalize(text)
    if not low:
        return ""
    # Standardize ampersands and special tokens
    s = low.replace("&", " and ").replace("+", " plus ")
    s = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\u0980-\u09FF\u0C00-\u0C7F\u0D00-\u0D7F\u0B80-\u0BFF\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def alphanum_only(text: Any) -> str:
    """Extract strictly alphanumeric characters without whitespace or punctuation."""
    return re.sub(r"[^a-zA-Z0-9]", "", lowercase_normalize(text))


def tokenize(text: Any) -> List[str]:
    """Tokenize normalized text into word tokens."""
    p = punct_normalize(text)
    return p.split() if p else []


def name_tokens_set(text: Any, min_len: int = 2) -> Set[str]:
    """Return set of distinctive tokens, excluding single-character noise."""
    tokens = tokenize(text)
    return {t for t in tokens if len(t) >= min_len}


def strip_legal_suffix(text: Any) -> Tuple[str, str]:
    """
    Conservatively strip legal suffixes (e.g. Inc, LLC, Pvt Ltd).
    Returns: (stripped_name, removed_suffix)
    """
    raw = punct_normalize(text)
    if not raw:
        return "", ""

    match = LEGAL_PATTERN.search(raw)
    suffix = match.group(0).strip() if match else ""
    stripped = LEGAL_PATTERN.sub(" ", raw)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return (stripped if stripped else raw, suffix)


def clean_domain_name(text: Any) -> str:
    """Extract and clean domain-style names (e.g. 'company.com' -> 'company')."""
    raw = lowercase_normalize(text)
    m = DOMAIN_PATTERN.search(raw)
    if m:
        name_part = m.group(3)
        return re.sub(r"[^a-zA-Z0-9]", "", name_part)
    return ""


def compressed_name(text: Any) -> str:
    """Compute dense alphanumeric signature, removing domain wrappers and legal suffixes."""
    dom = clean_domain_name(text)
    if dom and len(dom) >= 4:
        return dom
    stripped, _ = strip_legal_suffix(text)
    return alphanum_only(stripped)


def prepared_name_signatures(text: Any) -> Tuple[str, str]:
    """Return the existing punct and compressed forms without changing either definition."""
    normalized = punct_normalize(text)
    return normalized, compressed_name(text)


def stripped_from_punct(normalized: str) -> Tuple[str, str]:
    """Apply strip_legal_suffix's existing regex to an already normalized name."""
    if not normalized:
        return "", ""
    match = LEGAL_PATTERN.search(normalized)
    suffix = match.group(0).strip() if match else ""
    stripped = LEGAL_PATTERN.sub(" ", normalized)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return (stripped if stripped else normalized, suffix)


def parse_address_fields(address_text: Any, country: str = "") -> Dict[str, Any]:
    """
    Conservatively parse address into structured components without external geocoding:
    - house_number
    - street_text
    - unit_suite
    - postal_code
    - missing flags
    """
    addr_raw = raw_name(address_text)
    if not addr_raw:
        return {
            "house_number": "",
            "street_text": "",
            "unit_suite": "",
            "postal_code": "",
            "is_missing": True,
            "raw_address": "",
            "normalized_address": "",
        }

    addr_norm = punct_normalize(addr_raw)
    c_upper = str(country).strip().upper()

    # 1. House number (leading or distinctive number sequence)
    hn_match = re.search(r"\b(\d{1,6}[a-zA-Z]?)\b", addr_norm)
    house_num = hn_match.group(1) if hn_match else ""

    # 2. Unit / Suite / Flat
    unit_match = re.search(r"\b(unit|suite|apt|apartment|flat|fl|floor|room|block|kh no|plot no)\s*([a-zA-Z0-9\-]+)\b", addr_norm)
    unit_suite = f"{unit_match.group(1)} {unit_match.group(2)}" if unit_match else ""

    # 3. Country-aware postal code extraction
    postal_code = ""
    if c_upper in ("US", "USA"):
        # US ZIP code: 5 digits (or 5+4)
        m_pc = re.search(r"\b(\d{5})(?:-\d{4})?\b", addr_raw)
        if m_pc:
            postal_code = m_pc.group(1)
    elif c_upper in ("INDIA", "IN"):
        # Indian PIN code: 6 digits usually starting with 1-8
        m_pc = re.search(r"\b([1-8]\d{5})\b", addr_raw)
        if m_pc:
            postal_code = m_pc.group(1)
    elif c_upper in ("FRANCE", "FR"):
        # French postal code: 5 digits
        m_pc = re.search(r"\b(\d{5})\b", addr_raw)
        if m_pc:
            postal_code = m_pc.group(1)
    else:
        # Generic postal code fallback
        m_pc = re.search(r"\b(\d{5,6})\b", addr_raw)
        if m_pc:
            postal_code = m_pc.group(1)

    # 4. Street text tokens (excluding common stop words)
    tokens = [t for t in addr_norm.split() if len(t) >= 3]
    common_addr_stops = {
        "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
        "court", "ct", "suite", "ste", "unit", "floor", "fl", "apartment", "apt",
        "post", "office", "box", "po", "city", "state", "near", "opp", "block", "plot"
    }
    street_tokens = [t for t in tokens if t not in common_addr_stops and not t.isdigit()]

    return {
        "house_number": house_num,
        "street_text": " ".join(street_tokens[:5]),
        "unit_suite": unit_suite,
        "postal_code": postal_code,
        "is_missing": False,
        "raw_address": addr_raw,
        "normalized_address": addr_norm,
    }


def normalize_text(text: Any) -> str:
    """Standard alias for punctuation normalized text."""
    return punct_normalize(text)


def extract_postal_code(address_text: Any, country: str = "US") -> str:
    """Convenience helper to extract postal code from address string."""
    return parse_address_fields(address_text, country).get("postal_code", "")


def normalize_dataframe(df: pd.DataFrame, source_prefix: str = "") -> pd.DataFrame:
    """
    Augment DataFrame with multi-representation normalized fields in-place or returned.
    Memory-efficient column assignment.
    """
    res = df.copy()
    
    # Process business name
    names = res["business_name"].fillna("").astype(str).values
    res["name_punct"] = [punct_normalize(n) for n in names]
    res["name_compressed"] = [compressed_name(n) for n in names]
    
    suffixes_info = [strip_legal_suffix(n) for n in names]
    res["name_stripped"] = [s[0] for s in suffixes_info]
    res["legal_suffix"] = [s[1] for s in suffixes_info]
    
    # Process address & missingness flags
    addresses = res["business_address"].fillna("").astype(str).values
    countries = res["country"].fillna("").astype(str).values if "country" in res.columns else [""] * len(res)
    
    res["addr_missing"] = [not bool(a.strip()) or a.strip().lower() in ("nan", "none") for a in addresses]
    
    parsed = [parse_address_fields(a, c) for a, c in zip(addresses, countries)]
    res["addr_punct"] = [p["normalized_address"] for p in parsed]
    res["house_number"] = [p["house_number"] for p in parsed]
    res["postal_code"] = [p["postal_code"] for p in parsed]
    
    return res
