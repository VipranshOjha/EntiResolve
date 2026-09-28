"""
common.py — Shared normalization primitives used by both the name and
address normalizers.

Everything in this module is deterministic and stateless.  No external
data, no learned parameters, no I/O.

Data-driven design note
-----------------------
All transformation tables were built by analyzing an 80,000-record sample
from each of train_source1/2/3.tsv.  Any new entry must be justified by
observed data and documented here.

Normalization philosophy
------------------------
We preserve the original value and produce multiple representations.
Nothing is discarded at this layer — callers choose which representation
to use for which purpose.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence


# ---------------------------------------------------------------------------
# Basic Unicode / string utilities
# ---------------------------------------------------------------------------

def normalize_unicode(text: str, form: str = "NFKD") -> str:
    """
    Apply Unicode normalization to *text*.

    Default form is NFKD (Compatibility Decomposition).

    IMPORTANT: NFKD decomposes accented characters (é → e + combining accent)
    but does NOT convert Devanagari / Gujarati / other Indic scripts to Latin.
    It is safe to apply to all text without losing script identity.

    Parameters
    ----------
    text:
        Input string.
    form:
        Unicode normalization form.  Supported: 'NFC', 'NFD', 'NFKC', 'NFKD'.

    Returns
    -------
    str
        Normalized string (may contain combining characters if NFKD/NFD).
    """
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Collapse any sequence of whitespace (incl. Unicode spaces) to one space."""
    return _RE_WHITESPACE.sub(" ", text).strip()


def to_lowercase(text: str) -> str:
    """Lowercase using Python's full Unicode casefold (handles non-Latin too)."""
    return text.casefold()


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Any Unicode whitespace sequence
_RE_WHITESPACE = re.compile(r"\s+")

# Unicode replacement character (U+FFFD) — encoding corruption marker
_RE_REPLACEMENT_CHAR = re.compile(r"\ufffd")

# Leading non-alphanumeric prefix noise observed in the data:
# >>, --, ***, ..., @, #, <<, [, (, etc.
# Strips any sequence of non-word characters at the very start of a name
# that is not a letter in ANY script or a digit.
# Strategy: strip all characters that are NOT a Unicode letter, digit,
# opening-quote apostrophe, or specific allowed punctuation.
_RE_LEADING_NOISE = re.compile(
    r"^[\s\*\.\-\>\<\#\@\[\]\(\)\!\?\&\+\=\~\^\|\\\/\,\;\:\"\'\`\_\{\}]+"
)

# Trailing punctuation / noise
_RE_TRAILING_NOISE = re.compile(r"[\s\.\,\;\:\-\*\+\=\~\^\|\\\/\!\?]+$")

# Comma inside an address — used for component splitting
_RE_COMMA_SPLIT = re.compile(r",+")

# Postal / PIN code candidates: 5-digit (US/France ZIP) or 6-digit (India PIN)
# Must be word-bounded to avoid matching house numbers inside longer strings.
_RE_POSTAL = re.compile(r"\b(\d{5,6})\b")

# House / building number candidates: one or more digits at the start of
# a token, possibly followed by letters (e.g., "123A", "4/5B")
_RE_HOUSE_NUMBER = re.compile(r"^\d+[A-Za-z]?(/\d+[A-Za-z]?)*$")

# PO Box detection (US/Indian/French patterns)
_RE_PO_BOX = re.compile(r"\bP\.?\s*O\.?\s*BOX\b", re.IGNORECASE)

# Standalone double-ampersand or ampersand token
_RE_AMPERSAND = re.compile(r"\s*&\s*")

# Website domain suffix (for compact representation)
_RE_TLD = re.compile(
    r"\.(com|net|org|in|co\.in|co|biz|info|gov|edu|fr|uk|io)\b$", re.IGNORECASE
)

# Dots used as abbreviation markers: "Pvt." → "Pvt"
_RE_ABBREV_DOT = re.compile(r"\.(?=\s|$)")


# ---------------------------------------------------------------------------
# Legal suffix normalisation table
# ---------------------------------------------------------------------------
# Source: observed data (80K sample from each training source).
# Only includes forms actually seen with significant frequency (n ≥ 50).
# Canonical form: full expanded English word.
# Both the original key and the normalised canonical are kept by callers.
#
# Usage:  LEGAL_SUFFIX_MAP[token.lower()] → canonical string
#         If token not in map → keep as-is.

LEGAL_SUFFIX_MAP: dict[str, str] = {
    # Limited / Ltd
    "ltd": "limited",
    "ltd.": "limited",
    "l.t.d.": "limited",
    # Private
    "pvt": "private",
    "pvt.": "private",
    "priv": "private",
    # Corporation / Corp
    "corp": "corporation",
    "corp.": "corporation",
    # Incorporated / Inc
    "inc": "incorporated",
    "inc.": "incorporated",
    # Limited Liability Company
    "llc": "llc",          # already canonical; keep for lookup
    "l.l.c.": "llc",
    "l.l.c": "llc",
    # Limited Liability Partnership
    "llp": "llp",
    "l.l.p.": "llp",
    # Professional Corporation / Company
    "pc": "pc",
    "p.c.": "pc",
    "pllc": "pllc",
    # Company / Co
    "co": "company",
    "co.": "company",
    # Limited Partnership
    "lp": "lp",
    "l.p.": "lp",
    # Brothers
    "bros": "brothers",
    "bros.": "brothers",
    # International
    "intl": "international",
    "intl.": "international",
    # Services (not normalised — already a full word)
    # French legal forms
    "sarl": "sarl",
    "sas": "sas",
    "s.a.s": "sas",
    "s.a.s.": "sas",
    "sa": "sa",
    "s.a.": "sa",
    "sci": "sci",
    "sasu": "sasu",
    "eurl": "eurl",
    "ei": "ei",
}

# Two-token legal suffix bigrams (matched as consecutive token pairs).
# Key: (tok1, tok2) both lowercase.
LEGAL_SUFFIX_BIGRAMS: dict[tuple[str, str], str] = {
    ("private", "limited"): "private_limited",
    ("pvt", "ltd"): "private_limited",
    ("pvt.", "ltd."): "private_limited",
    ("pvt.", "ltd"): "private_limited",
    ("pvt", "ltd."): "private_limited",
    ("private", "ltd"): "private_limited",
    ("pvt", "limited"): "private_limited",
    ("public", "limited"): "public_limited",
}


# ---------------------------------------------------------------------------
# Address abbreviation normalisation table
# ---------------------------------------------------------------------------
# Strategy: normalise to the EXPANDED form so that "Ave" == "Avenue".
# Only include unambiguous abbreviations that are very common in the data.
# Deliberately exclude "ST" (Street vs State) and "CO" (County vs Company).
# Ambiguous tokens are left as-is; feature engineering will handle them.

ADDRESS_ABBREV_MAP: dict[str, str] = {
    "ave": "avenue",
    "av": "avenue",
    "rd": "road",
    "dr": "drive",
    "blvd": "boulevard",
    "hwy": "highway",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "sq": "square",
    "ln": "lane",
    "terr": "terrace",
    "terrace": "terrace",
    "trl": "trail",
    "pkwy": "parkway",
    "fwy": "freeway",
    "apt": "apartment",
    "appt": "apartment",
    "ste": "suite",
    "flr": "floor",
    "fl": "floor",
    # Indian address tokens
    "hno": "house_no",
    "h.no": "house_no",
    "h.no.": "house_no",
    "no.": "number",
    "nagar": "nagar",    # pass-through (common enough to be canonical)
    "marg": "marg",
    # French
    "r.": "rue",
    "r": "rue",          # only when isolated
    "bd": "boulevard",
    "all": "allee",
    "av.": "avenue",
    "blv": "boulevard",
}

# DO NOT apply ambiguous single-letter abbreviations ("N", "S", "E", "W")
# to avoid direction tokens vs name abbreviations.


# ---------------------------------------------------------------------------
# Core normalisation functions
# ---------------------------------------------------------------------------

def remove_leading_noise(text: str) -> str:
    """
    Strip leading punctuation / symbol noise from a business name.

    Examples observed in data:
        '>> Holloway Peak Inc'    → 'Holloway Peak Inc'
        '-- Summit Inc'           → 'Summit Inc'
        '*** Acme Corp'           → 'Acme Corp'
        '... Corp'                → 'Corp'
        '#thompsonstaking'        → 'thompsonstaking'
        '<< Team Ecole'           → 'Team Ecole'
        '[Intermediates]'         → 'Intermediates]'  (only leading bracket)

    The function does NOT strip punctuation that is embedded in the middle of
    a name (e.g., "Bros. & Sisters").
    """
    cleaned = _RE_LEADING_NOISE.sub("", text)
    return cleaned.strip()


def normalize_abbreviation_dots(text: str) -> str:
    """
    Remove dots used purely as abbreviation markers at end-of-token.

    'Pvt. Ltd.'  → 'Pvt Ltd'
    'Corp.'      → 'Corp'
    'U.S.A'      → kept as-is (internal dots)

    Only strips a trailing dot that is followed by whitespace or end-of-string.
    """
    return _RE_ABBREV_DOT.sub("", text)


def normalize_ampersand(text: str, replacement: str = " and ") -> str:
    """Replace ' & ' (or bare '&') with the given replacement string."""
    return _RE_AMPERSAND.sub(replacement, text)


def tokenize(text: str) -> tuple[str, ...]:
    """
    Split normalised text into tokens on whitespace.
    Returns an empty tuple for empty/whitespace-only input.
    """
    parts = text.split()
    return tuple(parts) if parts else ()


def apply_suffix_normalization(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """
    Apply legal suffix normalisation to a token sequence.

    Processes from right to left so that bigram suffixes at the end
    (e.g., 'Private Limited') are caught before individual tokens.

    Parameters
    ----------
    tokens:
        Lowercase tokens.

    Returns
    -------
    tuple[str, ...]
        Token tuple with recognized suffix tokens replaced by canonical forms.
    """
    if not tokens:
        return tokens

    result = list(tokens)
    n = len(result)

    # Check bigrams first (last two tokens)
    if n >= 2:
        bigram_key = (result[-2], result[-1])
        if bigram_key in LEGAL_SUFFIX_BIGRAMS:
            canonical = LEGAL_SUFFIX_BIGRAMS[bigram_key]
            result = result[:-2] + [canonical]
            n = len(result)

    # Check single-token suffixes
    for i in range(n - 1, max(n - 3, -1), -1):  # check last 2 tokens
        tok = result[i]
        if tok in LEGAL_SUFFIX_MAP:
            result[i] = LEGAL_SUFFIX_MAP[tok]

    return tuple(result)


def make_compact(text: str, alnum_only: bool = False) -> str:
    """
    Create a compact representation by removing whitespace.

    Parameters
    ----------
    text:
        Already-lowercased normalised string.
    alnum_only:
        If True, strip everything except alphanumeric characters and digits.
        Useful for matching website-style names against business names:
        'jeewanjyotiappartment.com' and 'Jeewan Jyoti Appartment' both
        become 'jeewanjyotiappartment' with alnum_only=True.

    Returns
    -------
    str
        Compact string.
    """
    if alnum_only:
        return re.sub(r"[^a-z0-9\u0900-\uD7FF]", "", text)
    # Remove spaces but keep other characters (punctuation, dots etc.)
    return re.sub(r"\s+", "", text)


def strip_tld(text: str) -> str:
    """
    Remove a common TLD suffix from the end of a string.

    'jeewanjyotiappartment.com' → 'jeewanjyotiappartment'
    'mycompany.co.in'           → 'mycompany'
    'ordinary text'             → 'ordinary text'  (no change)
    """
    return _RE_TLD.sub("", text).strip()
