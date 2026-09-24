import re
import unicodedata
import pandas as pd
import numpy as np

def raw_name(name):
    return str(name) if pd.notnull(name) else ""

def unicode_normalize(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    # NFKD normalizes unicode, e.g., separates accents, but we might want to keep Devanagari intact.
    # However, unicodedata.normalize works fine for Devanagari as well.
    return unicodedata.normalize('NFKD', text)

def lowercase_normalize(text):
    text = unicode_normalize(text)
    # lowercase and collapse whitespace
    text = text.lower()
    return re.sub(r'\s+', ' ', text).strip()

def punct_normalize(text):
    text = lowercase_normalize(text)
    # normalize & to and
    text = text.replace('&', ' and ')
    # remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def alphanum_only(text):
    text = lowercase_normalize(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    text = alphanum_only(text)
    tokens = [t for t in text.split() if t]
    return sorted(tokens)

# Legal suffixes
SUFFIXES = {
    'US': [r'\binc\b', r'\bllc\b', r'\bcorp\b', r'\bcorporation\b', r'\bco\b', r'\bcompany\b', r'\bltd\b', r'\blimited\b', r'\blp\b', r'\bllp\b', r'\bpc\b', r'\bpa\b', r'\bpllc\b', r'\bdba\b'],
    'IN': [r'\bpvt\s+ltd\b', r'\bprivate\s+limited\b', r'\bpvt\b', r'\bprivate\b', r'\bltd\b', r'\blimited\b', r'\bllp\b', r'\bpvte\b'],
    'FR': [r'\bsa\b', r'\bsas\b', r'\bsarl\b', r'\beurl\b', r'\bsci\b', r'\bsnc\b', r'\bsasu\b']
}

def strip_legal_suffix(name):
    """Strips common legal suffixes and returns both stripped and unstripped."""
    if pd.isnull(name):
        return "", ""
    name_str = str(name)
    norm = lowercase_normalize(name_str)
    
    # Try all suffixes (greedy first for combinations)
    all_suffixes = SUFFIXES['IN'] + SUFFIXES['US'] + SUFFIXES['FR']
    
    # Clean up punctuation before suffix matching
    clean_norm = re.sub(r'[^\w\s]', ' ', norm)
    clean_norm = re.sub(r'\s+', ' ', clean_norm).strip()
    
    stripped = clean_norm
    for suffix in all_suffixes:
        # Match suffix at the end of the string
        pattern = suffix + r'$'
        if re.search(pattern, stripped):
            stripped = re.sub(pattern, '', stripped).strip()
            # If we stripped a multi-word like "pvt ltd", let's break,
            # or continue to strip more? Let's just do one pass at the end.
            break
            
    return stripped, norm

STOPWORDS = {'and', 'the', 'of', 'in', 'for', 'a', 'an', 'to', 'at', 'co'}

def name_tokens_set(name):
    text = alphanum_only(name)
    tokens = {t for t in text.split() if len(t) >= 2 and t not in STOPWORDS}
    return tokens

# Address normalization

def normalize_address(addr):
    if pd.isnull(addr):
        return ""
    text = lowercase_normalize(str(addr))
    # Basic abbreviations
    abbrevs = {
        r'\bst\b': 'street', r'\bave\b': 'avenue', r'\brd\b': 'road',
        r'\bdr\b': 'drive', r'\bblvd\b': 'boulevard', r'\bln\b': 'lane',
        r'\bct\b': 'court', r'\bpl\b': 'place', r'\bste\b': 'suite',
        r'\bapt\b': 'apartment', r'\bfl\b': 'floor'
    }
    for pat, repl in abbrevs.items():
        text = re.sub(pat, repl, text)
    return text

def parse_address_fields(addr, country):
    if pd.isnull(addr):
        return {'house_number': '', 'street_text': '', 'city': '', 'region': '', 'postal_code': '', 'country': country}
    
    addr_str = normalize_address(addr)
    
    # Extremely basic and conservative extraction
    house_number = ""
    postal_code = ""
    
    # House number: leading digits
    hn_match = re.match(r'^(\d+)[a-z]?\b', addr_str)
    if hn_match:
        house_number = hn_match.group(1)
        
    # Postal code
    if country == 'US':
        pc_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', addr_str)
        if pc_match:
            postal_code = pc_match.group(1)
    elif country == 'IN': # India
        pc_match = re.search(r'\b(\d{6})\b', addr_str)
        if pc_match:
            postal_code = pc_match.group(1)
    elif country == 'FR': # France
        pc_match = re.search(r'\b(\d{5})\b', addr_str)
        if pc_match:
            postal_code = pc_match.group(1)
            
    # For street, city, region, we would ideally need a parser, but we'll do best-effort or leave empty 
    # to avoid destructive extraction, unless obvious.
    # Since we shouldn't use external libs, we'll return the whole address as street_text roughly.
    return {
        'house_number': house_number,
        'street_text': addr_str,
        'city': '', 
        'region': '',
        'postal_code': postal_code,
        'country': country
    }

def address_tokens(addr):
    return tokenize(addr)

def numeric_tokens(addr):
    if pd.isnull(addr):
        return []
    text = str(addr)
    return re.findall(r'\d+', text)

def rare_address_tokens(addr):
    # Without a global dictionary, we just filter out common address words
    tokens = address_tokens(addr)
    common = {'street', 'road', 'avenue', 'drive', 'suite', 'floor', 'apartment', 'lane', 'court', 'boulevard', 'place', 'room', 'building', 'no', 'number'}
    return [t for t in tokens if t not in common]

def normalize_dataframe(df, country_col='country'):
    # Missingness flags
    df['is_name_missing'] = df['business_name'].isnull() | (df['business_name'] == '')
    df['is_addr_missing'] = df['business_address'].isnull() | (df['business_address'] == '')
    
    # Name normalizations
    df['name_raw'] = df['business_name'].apply(raw_name)
    df['name_lower'] = df['business_name'].apply(lowercase_normalize)
    df['name_normalized'] = df['business_name'].apply(punct_normalize)
    df['name_alphanum'] = df['business_name'].apply(alphanum_only)
    
    stripped_names = df['business_name'].apply(strip_legal_suffix)
    df['name_no_suffix'] = [x[0] for x in stripped_names]
    
    df['name_tokens'] = df['business_name'].apply(tokenize)
    
    # Address normalizations
    df['addr_raw'] = df['business_address'].apply(lambda x: str(x) if pd.notnull(x) else "")
    df['addr_lower'] = df['business_address'].apply(lowercase_normalize)
    df['addr_normalized'] = df['business_address'].apply(normalize_address)
    df['addr_tokens'] = df['business_address'].apply(address_tokens)
    df['addr_numerics'] = df['business_address'].apply(numeric_tokens)
    
    # Parse fields
    def apply_parse(row):
        c = row[country_col] if country_col in row else ''
        return parse_address_fields(row['business_address'], c)
        
    parsed = df.apply(apply_parse, axis=1)
    df['postal_code'] = [p['postal_code'] for p in parsed]
    df['city'] = [p['city'] for p in parsed]
    df['region'] = [p['region'] for p in parsed]
    df['house_number'] = [p['house_number'] for p in parsed]
    
    return df

def normalize_text(text):
    if pd.isnull(text) or not str(text).strip():
        return ""
    return punct_normalize(str(text))

def extract_postal_code(text, country='US'):
    if pd.isnull(text) or not str(text).strip():
        return ""
    parsed = parse_address_fields(str(text), country)
    return parsed.get('postal_code', '')

