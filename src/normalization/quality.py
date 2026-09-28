"""
quality.py — Quality classification for business names and addresses.

Name quality is driven by script metadata + explicit corruption signals.
Address quality is simpler: missing or corrupted/non-Latin.

Quality labels are plain strings (not enums) for forward-compatibility —
new categories can be added without breaking existing code that just checks
string equality.

Observed patterns from training data analysis (80K sample):
  - Corrupted (U+FFFD) names: small % of S2 records
  - Non-Latin names: 9.2% of S2, 5.3% of S3
  - Mixed-script (Latin + Indic): 0.4% S2, 0.6% S3
  - Noise prefix (>>, --, ***, @, #, etc.): ~3% of S2/S3
  - Domain/website names (.com, .in, etc.): ~1-2% of S2/S3
"""

from __future__ import annotations

import re

from src.normalization.scripts import (
    ScriptInfo,
    DOM_CORRUPTED,
    DOM_EMPTY,
    DOM_LATIN,
    DOM_MIXED,
    DOM_NON_LATIN,
    DOM_NO_LETTERS,
)

# ---------------------------------------------------------------------------
# Name quality labels
# ---------------------------------------------------------------------------

NQ_GOOD = "GOOD"
"""Latin name with no anomalies."""

NQ_NON_LATIN = "NON_LATIN"
"""Name written entirely in a non-Latin script (e.g., Devanagari, Gujarati)."""

NQ_MIXED_SCRIPT = "MIXED_SCRIPT"
"""Name contains both Latin and non-Latin letter characters."""

NQ_CORRUPTED = "CORRUPTED"
"""Encoding corruption detected (>10% U+FFFD replacement characters)."""

NQ_WEBSITE = "WEBSITE"
"""Name looks like a domain/website (e.g., 'acme.com', 'myshop.co.in')."""

NQ_NOISE_PREFIX = "NOISE_PREFIX"
"""Name has a leading noise prefix stripped by normalization (>>, --, etc.).
   Can be combined with other labels; this flag alone doesn't imply bad quality."""

NQ_EMPTY = "EMPTY"
"""Empty or whitespace-only string."""

NQ_NO_LETTERS = "NO_LETTERS"
"""Non-empty but contains no Unicode letter characters."""

NQ_OTHER = "OTHER"
"""Any other unusual pattern not covered above."""


# ---------------------------------------------------------------------------
# Address quality labels
# ---------------------------------------------------------------------------

AQ_GOOD = "GOOD"
"""Normal address with text content."""

AQ_MISSING = "MISSING"
"""Empty or whitespace-only address (observed in 3.3% of S2/S3 training records)."""

AQ_NON_LATIN = "NON_LATIN"
"""Address primarily or entirely in a non-Latin script."""

AQ_MIXED_SCRIPT = "MIXED_SCRIPT"
"""Address contains both Latin and non-Latin characters."""

AQ_CORRUPTED = "CORRUPTED"
"""Encoding corruption in the address field."""

AQ_OTHER = "OTHER"
"""Any other anomaly."""


# ---------------------------------------------------------------------------
# Helper patterns
# ---------------------------------------------------------------------------

_RE_WEBSITE = re.compile(
    r"""
    (?:                          # common patterns for website-style names:
        \w+\.(?:com|net|org|in|co\.in|biz|info|gov|edu|fr|uk|io|co|us)
        | www\.\w+               # 'www.something'
        | \w+\.co\b              # 'name.co'
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_LEADING_NOISE_PRESENT = re.compile(
    r"^[\s\*\.\-\>\<\#\@\[\]\(\)\!\?\&\+\=\~\^\|\\\/\,\;\:\"\'\`\_\{\}]+"
)


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

def classify_name_quality(
    raw: str,
    script_info: ScriptInfo,
) -> str:
    """
    Classify the quality of a business name.

    Parameters
    ----------
    raw:
        The original (un-normalised) business name string.
    script_info:
        ScriptInfo computed for *raw*.

    Returns
    -------
    str
        One of the NQ_* constants.  The primary quality label.
        (A name can exhibit multiple quality issues, but only one primary
        label is returned, in priority order.)
    """
    # Empty
    if not raw.strip():
        return NQ_EMPTY

    # Corruption takes highest priority
    if script_info.is_corrupted:
        return NQ_CORRUPTED

    # No letters at all
    if script_info.dominant_script == DOM_NO_LETTERS:
        return NQ_NO_LETTERS

    # Non-Latin
    if script_info.is_non_latin:
        return NQ_NON_LATIN

    # Mixed script
    if script_info.is_mixed_script:
        return NQ_MIXED_SCRIPT

    # Website-style name (check against stripped, lowercased raw)
    if _RE_WEBSITE.search(raw.strip().lower()):
        return NQ_WEBSITE

    # Otherwise GOOD (Latin, no corruption)
    return NQ_GOOD


def has_noise_prefix(raw: str) -> bool:
    """
    Return True if the name has a leading noise prefix.

    Examples: '>> Acme Corp', '-- Holloway Peak Inc', '*** Ltd'
    """
    return bool(_RE_LEADING_NOISE_PRESENT.match(raw))


def classify_address_quality(raw: str, script_info: ScriptInfo) -> str:
    """
    Classify the quality of an address string.

    Parameters
    ----------
    raw:
        The original (un-normalised) address string.
    script_info:
        ScriptInfo computed for *raw*.

    Returns
    -------
    str
        One of the AQ_* constants.
    """
    if not raw.strip():
        return AQ_MISSING

    if script_info.is_corrupted:
        return AQ_CORRUPTED

    if script_info.is_non_latin:
        return AQ_NON_LATIN

    if script_info.is_mixed_script:
        return AQ_MIXED_SCRIPT

    return AQ_GOOD
