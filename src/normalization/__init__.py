"""
src/normalization/__init__.py

Public API for the normalization package.

Most callers should import from here rather than the sub-modules directly.
"""

from src.normalization.address import AddressNormalizer, AddressRecord, normalize_address
from src.normalization.common import (
    LEGAL_SUFFIX_MAP,
    LEGAL_SUFFIX_BIGRAMS,
    ADDRESS_ABBREV_MAP,
    normalize_unicode,
    normalize_whitespace,
    to_lowercase,
    remove_leading_noise,
    normalize_ampersand,
    normalize_abbreviation_dots,
    tokenize,
    apply_suffix_normalization,
    make_compact,
    strip_tld,
)
from src.normalization.name import NameNormalizer, NameRecord, normalize_name
from src.normalization.phone import PhoneNormalizer, PhoneRecord, normalize_phone
from src.normalization.quality import (
    classify_address_quality,
    classify_name_quality,
    has_noise_prefix,
    NQ_GOOD, NQ_NON_LATIN, NQ_MIXED_SCRIPT, NQ_CORRUPTED,
    NQ_WEBSITE, NQ_NOISE_PREFIX, NQ_EMPTY, NQ_NO_LETTERS, NQ_OTHER,
    AQ_GOOD, AQ_MISSING, AQ_NON_LATIN, AQ_MIXED_SCRIPT, AQ_CORRUPTED, AQ_OTHER,
)
from src.normalization.record import BusinessRecord, NormalizationPipeline, normalize_record
from src.normalization.scripts import (
    ScriptInfo,
    get_script_info,
    SCRIPT_LATIN, SCRIPT_DEVANAGARI, SCRIPT_GUJARATI, SCRIPT_BENGALI,
    SCRIPT_GURMUKHI, SCRIPT_ORIYA, SCRIPT_TAMIL, SCRIPT_TELUGU,
    SCRIPT_KANNADA, SCRIPT_MALAYALAM,
    DOM_LATIN, DOM_NON_LATIN, DOM_MIXED, DOM_EMPTY, DOM_NO_LETTERS,
    DOM_CORRUPTED,
)

__all__ = [
    # Records
    "BusinessRecord",
    "NameRecord",
    "AddressRecord",
    "PhoneRecord",
    "ScriptInfo",
    # Normalizers / pipeline
    "NormalizationPipeline",
    "NameNormalizer",
    "AddressNormalizer",
    "PhoneNormalizer",
    # Convenience functions
    "normalize_record",
    "normalize_name",
    "normalize_address",
    "normalize_phone",
    "get_script_info",
    # Quality labels
    "NQ_GOOD", "NQ_NON_LATIN", "NQ_MIXED_SCRIPT", "NQ_CORRUPTED",
    "NQ_WEBSITE", "NQ_NOISE_PREFIX", "NQ_EMPTY", "NQ_NO_LETTERS", "NQ_OTHER",
    "AQ_GOOD", "AQ_MISSING", "AQ_NON_LATIN", "AQ_MIXED_SCRIPT",
    "AQ_CORRUPTED", "AQ_OTHER",
    # Script constants
    "SCRIPT_LATIN", "SCRIPT_DEVANAGARI", "SCRIPT_GUJARATI", "SCRIPT_BENGALI",
    "DOM_LATIN", "DOM_NON_LATIN", "DOM_MIXED", "DOM_EMPTY",
    "DOM_NO_LETTERS", "DOM_CORRUPTED",
    # Lookup tables
    "LEGAL_SUFFIX_MAP", "LEGAL_SUFFIX_BIGRAMS", "ADDRESS_ABBREV_MAP",
]
