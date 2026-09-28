"""
scripts.py — Unicode script detection for business name strings.

Purpose
-------
Source 2 and Source 3 may contain business names written in non-Latin scripts
(Devanagari, Telugu, Kannada, Bengali, Tamil, Gujarati, Malayalam, Gurmukhi,
Oriya, and others).  Source 1 is always Latin.

Observed script distribution (80K sample from training data):
    S1: 100% Latin
    S2: 90.8% Latin, 5.0% Devanagari, 0.7% Telugu, 0.7% Kannada,
        0.6% Bengali, 0.6% Tamil, 0.6% Gujarati, 0.4% Mixed,
        0.4% Malayalam, 0.1% Oriya, 0.1% Gurmukhi
    S3: 94.7% Latin, 2.6% Devanagari, 0.6% Mixed, ≤0.4% others

Script metadata is used by downstream phases to:
  1. Flag pairs where name comparison is unreliable (non-Latin S2/S3 vs Latin S1)
  2. Build script-specific blocking strategies
  3. Up-weight address evidence when name evidence is unreliable
  4. Potentially learn cross-script correspondences from training-data pairs

Design principles
-----------------
* Pure code-point range checks — no external libraries
* O(n) in string length per call, fast enough for 23M records
* Generic: handles any Unicode script, not just the ones observed so far
* DOES NOT transliterate — it only detects and describes

Unicode block ranges used
--------------------------
Latin     : U+0000–U+024F  (Basic Latin + Latin Extended-A/B + IPA)
Devanagari: U+0900–U+097F
Bengali   : U+0980–U+09FF
Gurmukhi  : U+0A00–U+0A7F  (Punjabi)
Gujarati  : U+0A80–U+0AFF
Oriya     : U+0B00–U+0B7F
Tamil     : U+0B80–U+0BFF
Telugu    : U+0C00–U+0C7F
Kannada   : U+0C80–U+0CFF
Malayalam : U+0D00–U+0D7F
Arabic    : U+0600–U+06FF
Hebrew    : U+0590–U+05FF
Cyrillic  : U+0400–U+04FF
Greek     : U+0370–U+03FF
Thai      : U+0E00–U+0E7F
CJK       : U+4E00–U+9FFF  (CJK Unified Ideographs)
U+FFFD    : Unicode replacement character (corruption marker)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Counter as CounterType


# ---------------------------------------------------------------------------
# Script constants
# ---------------------------------------------------------------------------

SCRIPT_LATIN = "LATIN"
SCRIPT_DEVANAGARI = "DEVANAGARI"
SCRIPT_BENGALI = "BENGALI"
SCRIPT_GURMUKHI = "GURMUKHI"
SCRIPT_GUJARATI = "GUJARATI"
SCRIPT_ORIYA = "ORIYA"
SCRIPT_TAMIL = "TAMIL"
SCRIPT_TELUGU = "TELUGU"
SCRIPT_KANNADA = "KANNADA"
SCRIPT_MALAYALAM = "MALAYALAM"
SCRIPT_ARABIC = "ARABIC"
SCRIPT_HEBREW = "HEBREW"
SCRIPT_CYRILLIC = "CYRILLIC"
SCRIPT_GREEK = "GREEK"
SCRIPT_THAI = "THAI"
SCRIPT_CJK = "CJK"
SCRIPT_CORRUPTED = "CORRUPTED"  # U+FFFD replacement character
SCRIPT_OTHER = "OTHER"


# ---------------------------------------------------------------------------
# Dominant script labels (for ScriptInfo.dominant_script)
# ---------------------------------------------------------------------------

DOM_LATIN = "LATIN"
DOM_NON_LATIN = "NON_LATIN"     # pure non-Latin, single script
DOM_MIXED = "MIXED"             # both Latin and non-Latin letters
DOM_EMPTY = "EMPTY"             # no characters at all
DOM_NO_LETTERS = "NO_LETTERS"   # has characters but no Unicode letters
DOM_CORRUPTED = "CORRUPTED"     # significant U+FFFD proportion


# ---------------------------------------------------------------------------
# Code-point classification
# ---------------------------------------------------------------------------

def _classify_codepoint(cp: int) -> str:
    """
    Return the Unicode script name for a code point.

    Only classifies *letter* code points — punctuation, digits, and spaces
    are handled separately by the caller using unicodedata.category().

    Parameters
    ----------
    cp : int
        Unicode code point (``ord(character)``).

    Returns
    -------
    str
        One of the SCRIPT_* constants defined above.
    """
    # Replacement character — encoding corruption
    if cp == 0xFFFD:
        return SCRIPT_CORRUPTED
    # Basic Latin + Latin Extended-A, B, C, D, Additional
    if cp <= 0x024F:
        return SCRIPT_LATIN
    # General Punctuation / Currency / Letterlike Symbols — not letters
    if 0x2000 <= cp <= 0x2BFF:
        return SCRIPT_OTHER
    # Arabic (includes Arabic Extended-A / Extended-B)
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return SCRIPT_ARABIC
    # Hebrew
    if 0x0590 <= cp <= 0x05FF:
        return SCRIPT_HEBREW
    # Greek and Coptic
    if 0x0370 <= cp <= 0x03FF:
        return SCRIPT_GREEK
    # Cyrillic
    if 0x0400 <= cp <= 0x04FF:
        return SCRIPT_CYRILLIC
    # Devanagari
    if 0x0900 <= cp <= 0x097F:
        return SCRIPT_DEVANAGARI
    # Bengali
    if 0x0980 <= cp <= 0x09FF:
        return SCRIPT_BENGALI
    # Gurmukhi
    if 0x0A00 <= cp <= 0x0A7F:
        return SCRIPT_GURMUKHI
    # Gujarati
    if 0x0A80 <= cp <= 0x0AFF:
        return SCRIPT_GUJARATI
    # Oriya / Odia
    if 0x0B00 <= cp <= 0x0B7F:
        return SCRIPT_ORIYA
    # Tamil
    if 0x0B80 <= cp <= 0x0BFF:
        return SCRIPT_TAMIL
    # Telugu
    if 0x0C00 <= cp <= 0x0C7F:
        return SCRIPT_TELUGU
    # Kannada
    if 0x0C80 <= cp <= 0x0CFF:
        return SCRIPT_KANNADA
    # Malayalam
    if 0x0D00 <= cp <= 0x0D7F:
        return SCRIPT_MALAYALAM
    # Thai
    if 0x0E00 <= cp <= 0x0E7F:
        return SCRIPT_THAI
    # CJK Unified Ideographs (common Chinese/Japanese/Korean)
    if 0x4E00 <= cp <= 0x9FFF:
        return SCRIPT_CJK
    # CJK Extension A
    if 0x3400 <= cp <= 0x4DBF:
        return SCRIPT_CJK
    # Hangul (Korean)
    if 0xAC00 <= cp <= 0xD7AF:
        return SCRIPT_OTHER
    return SCRIPT_OTHER


# ---------------------------------------------------------------------------
# ScriptInfo dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ScriptInfo:
    """
    Per-string script composition metadata.

    All ratio fields are in [0.0, 1.0] and sum to 1.0 over the set
    {letter_ratio, digit_ratio, punct_ratio, space_ratio, other_char_ratio}.
    The letter-script ratios sum to letter_ratio.

    Attributes
    ----------
    dominant_script:
        High-level script label.  One of the DOM_* constants.
    primary_script:
        The most frequent letter script in the string (a SCRIPT_* constant).
        'NONE' if there are no Unicode letters.
    latin_ratio:
        Fraction of all characters that are Latin letters.
    non_latin_letter_ratio:
        Fraction of all characters that are non-Latin Unicode letters.
    digit_ratio:
        Fraction of all characters that are Unicode digits (Nd category).
    punct_ratio:
        Fraction of all characters that are punctuation or symbols.
    space_ratio:
        Fraction of all characters that are whitespace.
    replacement_ratio:
        Fraction of all characters that are U+FFFD (corruption marker).
    is_mixed_script:
        True when both Latin and non-Latin letters are present.
    is_non_latin:
        True when there are letters but none are Latin.
    is_corrupted:
        True when replacement_ratio > CORRUPTION_THRESHOLD.
    scripts_present:
        Tuple of unique letter-script names found in the string,
        sorted alphabetically.  Empty tuple for strings with no letters.
    char_count:
        Total number of characters.
    """

    dominant_script: str
    primary_script: str
    latin_ratio: float
    non_latin_letter_ratio: float
    digit_ratio: float
    punct_ratio: float
    space_ratio: float
    replacement_ratio: float
    is_mixed_script: bool
    is_non_latin: bool
    is_corrupted: bool
    scripts_present: tuple[str, ...]
    char_count: int


# Threshold above which a string is classified as CORRUPTED
_CORRUPTION_THRESHOLD: float = 0.10


def get_script_info(text: str) -> ScriptInfo:
    """
    Compute ScriptInfo for a string.

    Iterates over the string once (O(n) time).

    Parameters
    ----------
    text:
        Input string (raw, before any normalisation).

    Returns
    -------
    ScriptInfo
    """
    n = len(text)

    if n == 0:
        return ScriptInfo(
            dominant_script=DOM_EMPTY,
            primary_script="NONE",
            latin_ratio=0.0,
            non_latin_letter_ratio=0.0,
            digit_ratio=0.0,
            punct_ratio=0.0,
            space_ratio=0.0,
            replacement_ratio=0.0,
            is_mixed_script=False,
            is_non_latin=False,
            is_corrupted=False,
            scripts_present=(),
            char_count=0,
        )

    latin_count = 0
    non_latin_letter_count = 0
    digit_count = 0
    punct_count = 0
    space_count = 0
    replacement_count = 0
    script_counter: dict[str, int] = {}

    for ch in text:
        cp = ord(ch)
        cat = unicodedata.category(ch)

        if cp == 0xFFFD:
            replacement_count += 1
            # Count as "other", not as letter
            continue

        if cat.startswith("L"):          # Unicode letter
            script = _classify_codepoint(cp)
            script_counter[script] = script_counter.get(script, 0) + 1
            if script == SCRIPT_LATIN:
                latin_count += 1
            else:
                non_latin_letter_count += 1

        elif cat.startswith("N"):        # digit (includes U+0966 etc.)
            digit_count += 1

        elif cat.startswith("Z") or ch in " \t\n\r\f\v":
            space_count += 1

        elif cat.startswith("P") or cat.startswith("S"):
            punct_count += 1
        # else: control chars → ignored

    total = float(n)
    total_letters = latin_count + non_latin_letter_count

    # Remove CORRUPTED from script_counter (it's tracked separately)
    script_counter.pop(SCRIPT_CORRUPTED, None)

    # Build sorted tuple of present letter scripts
    scripts_present = tuple(sorted(script_counter.keys()))

    # Dominant and primary script
    if total_letters == 0 and replacement_count == 0 and n > 0:
        dominant = DOM_NO_LETTERS
        primary = "NONE"
    elif total_letters == 0 and n == 0:
        dominant = DOM_EMPTY
        primary = "NONE"
    else:
        replacement_ratio = replacement_count / total
        is_corrupted = replacement_ratio > _CORRUPTION_THRESHOLD

        is_mixed = latin_count > 0 and non_latin_letter_count > 0
        is_non_latin = total_letters > 0 and latin_count == 0

        if is_corrupted:
            dominant = DOM_CORRUPTED
        elif total_letters == 0:
            dominant = DOM_NO_LETTERS
        elif is_mixed:
            dominant = DOM_MIXED
        elif is_non_latin:
            dominant = DOM_NON_LATIN
        else:
            dominant = DOM_LATIN

        if script_counter:
            primary = max(script_counter, key=lambda k: script_counter[k])
        elif total_letters == 0:
            primary = "NONE"
        else:
            primary = SCRIPT_LATIN

    replacement_ratio_val = replacement_count / total
    is_corrupted_val = replacement_ratio_val > _CORRUPTION_THRESHOLD
    is_mixed_val = latin_count > 0 and non_latin_letter_count > 0
    is_non_latin_val = total_letters > 0 and latin_count == 0

    return ScriptInfo(
        dominant_script=dominant,
        primary_script=primary if script_counter else (
            SCRIPT_LATIN if latin_count > 0 else "NONE"
        ),
        latin_ratio=latin_count / total,
        non_latin_letter_ratio=non_latin_letter_count / total,
        digit_ratio=digit_count / total,
        punct_ratio=punct_count / total,
        space_ratio=space_count / total,
        replacement_ratio=replacement_ratio_val,
        is_mixed_script=is_mixed_val,
        is_non_latin=is_non_latin_val,
        is_corrupted=is_corrupted_val,
        scripts_present=scripts_present,
        char_count=n,
    )
