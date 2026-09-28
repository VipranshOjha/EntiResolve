"""
name.py — Multi-view business name normalizer.

Produces a NameRecord with multiple parallel representations of a single
raw business name string.  The original raw value is always preserved.

Representations produced
------------------------
raw                 Original string, unmodified.
normalized          Lowercased, whitespace-normalised, leading noise stripped,
                    abbreviation dots removed.
normalized_nfkd     NFKD-normalised form of 'normalized'.
tokens              Whitespace-split tokens of 'normalized'.
tokens_nfkd         Whitespace-split tokens of 'normalized_nfkd'.
tokens_suffix_norm  Tokens with recognised legal suffixes expanded to their
                    canonical long form (e.g., 'Pvt Ltd' → 'private_limited').
compact             'normalized' with all whitespace removed (no spaces).
compact_alnum       'normalized' keeping only alphanumeric + Indic script chars.
compact_no_tld      'compact_alnum' with common TLD suffix stripped
                    (useful for matching '.com' website names).
ampersand_norm      'normalized' with ' & ' replaced by ' and '.
ampersand_tokens    Whitespace-split tokens of 'ampersand_norm'.
script              ScriptInfo for the raw string.
quality             Primary quality label (NQ_* constant).
noise_prefix_stripped  True when a leading noise prefix was stripped.

Design principles
-----------------
1.  No information is destroyed.  Every transformation adds a new view;
    it never overwrites the previous one.
2.  All representations are computed in a single linear pass per record.
3.  The NameNormalizer is stateless — one instance is reusable across
    millions of records.
4.  The NameRecord is frozen (immutable) to prevent accidental mutation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.normalization.common import (
    apply_suffix_normalization,
    make_compact,
    normalize_ampersand,
    normalize_abbreviation_dots,
    normalize_unicode,
    normalize_whitespace,
    remove_leading_noise,
    strip_tld,
    to_lowercase,
    tokenize,
)
from src.normalization.quality import (
    classify_name_quality,
    has_noise_prefix,
)
from src.normalization.scripts import ScriptInfo, get_script_info


# ---------------------------------------------------------------------------
# NameRecord dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NameRecord:
    """
    All multi-view representations of a single business name.

    Fields
    ------
    raw:
        The original input string.  Never modified.

    normalized:
        Canonical working form:
        raw → strip leading noise → normalize whitespace → lowercase
            → remove abbreviation dots.
        Examples:
            '>> Urology Partners  Inc' → 'urology partners inc'
            'SAI TRADERS PRIVATE LIMITED' → 'sai traders private limited'

    normalized_nfkd:
        NFKD-normalised version of 'normalized'.  Decomposes accented
        characters (é → e + combining ́) for accent-insensitive comparison.
        Does NOT convert non-Latin scripts to Latin.

    tokens:
        Tuple of whitespace-split tokens from 'normalized'.

    tokens_nfkd:
        Tuple of whitespace-split tokens from 'normalized_nfkd'.

    tokens_suffix_norm:
        Tokens with legal suffixes replaced by canonical long forms.
        Examples:
            ('sai', 'traders', 'pvt', 'ltd') → ('sai', 'traders', 'private_limited')
            ('acme', 'corp') → ('acme', 'corporation')

    compact:
        'normalized' with all whitespace removed.
        Useful for TF-IDF character n-grams and prefix-key blocking.

    compact_alnum:
        'normalized' keeping only alphanumeric characters and Indic-script
        letters.  Useful for matching website-style names.
        'jeewan jyoti appartment' → 'jeewanjyotiappartment'

    compact_no_tld:
        'compact_alnum' with common TLD suffix removed.
        'jeewanjyotiappartmentcom' → 'jeewanjyotiappartment'

    ampersand_norm:
        'normalized' with ' & ' replaced by ' and '.
        Useful when one source uses '&' and another spells out 'and'.

    ampersand_tokens:
        Whitespace-split tokens of 'ampersand_norm'.

    script:
        ScriptInfo computed from 'raw'.  Describes script composition ratios
        and dominant script.

    quality:
        Primary quality label (NQ_GOOD, NQ_NON_LATIN, NQ_MIXED_SCRIPT,
        NQ_CORRUPTED, NQ_WEBSITE, NQ_EMPTY, NQ_NO_LETTERS, NQ_OTHER).

    noise_prefix_stripped:
        True when a leading noise prefix (>>, --, ***, #, @, etc.) was
        detected and stripped during normalization.
    """

    raw: str
    normalized: str
    normalized_nfkd: str
    tokens: tuple[str, ...]
    tokens_nfkd: tuple[str, ...]
    tokens_suffix_norm: tuple[str, ...]
    compact: str
    compact_alnum: str
    compact_no_tld: str
    ampersand_norm: str
    ampersand_tokens: tuple[str, ...]
    script: ScriptInfo
    quality: str
    noise_prefix_stripped: bool


# ---------------------------------------------------------------------------
# NameNormalizer
# ---------------------------------------------------------------------------

class NameNormalizer:
    """
    Stateless normalizer that converts a raw name string into a NameRecord.

    Instantiate once and reuse across millions of records.

    Example
    -------
    >>> normalizer = NameNormalizer()
    >>> rec = normalizer.normalize('>> Sai Traders Pvt. Ltd.')
    >>> rec.normalized
    'sai traders pvt ltd'
    >>> rec.tokens_suffix_norm
    ('sai', 'traders', 'private_limited')
    >>> rec.quality
    'GOOD'
    >>> rec.noise_prefix_stripped
    True
    """

    def normalize(self, raw: str) -> NameRecord:
        """
        Normalise a single raw business name into a NameRecord.

        Parameters
        ----------
        raw:
            The original business name string from the TSV file.

        Returns
        -------
        NameRecord
        """
        # --- Step 0: Script info on the raw string (before any modification) ---
        script = get_script_info(raw)

        # --- Step 1: Detect and strip leading noise prefix ---
        noise_present = has_noise_prefix(raw)
        stripped = remove_leading_noise(raw) if noise_present else raw

        # --- Step 2: Remove abbreviation dots (Pvt. → Pvt) ---
        no_abbr_dots = normalize_abbreviation_dots(stripped)

        # --- Step 3: Normalize whitespace ---
        ws_clean = normalize_whitespace(no_abbr_dots)

        # --- Step 4: Lowercase ---
        normalized = to_lowercase(ws_clean)

        # --- Step 5: NFKD normalized ---
        normalized_nfkd = normalize_unicode(normalized, form="NFKD")

        # --- Step 6: Tokenize ---
        tokens = tokenize(normalized)
        tokens_nfkd = tokenize(normalized_nfkd)

        # --- Step 7: Suffix normalization on tokens ---
        tokens_suffix_norm = apply_suffix_normalization(tokens)

        # --- Step 8: Compact representations ---
        compact = make_compact(normalized, alnum_only=False)
        compact_alnum = make_compact(normalized, alnum_only=True)
        # compact_no_tld: strip TLD BEFORE removing non-alnum chars, so the dot
        # is still present for the regex match.
        # 'jeewanjyotiappartment.com' → strip '.com' → 'jeewanjyotiappartment' → alnum
        normalized_no_tld = strip_tld(normalized)
        compact_no_tld = make_compact(normalized_no_tld, alnum_only=True)

        # --- Step 9: Ampersand normalization ---
        ampersand_norm = normalize_ampersand(normalized, replacement=" and ")
        # Re-normalize whitespace after substitution
        ampersand_norm = normalize_whitespace(ampersand_norm)
        ampersand_tokens = tokenize(ampersand_norm)

        # --- Step 10: Quality classification ---
        quality = classify_name_quality(raw, script)

        return NameRecord(
            raw=raw,
            normalized=normalized,
            normalized_nfkd=normalized_nfkd,
            tokens=tokens,
            tokens_nfkd=tokens_nfkd,
            tokens_suffix_norm=tokens_suffix_norm,
            compact=compact,
            compact_alnum=compact_alnum,
            compact_no_tld=compact_no_tld,
            ampersand_norm=ampersand_norm,
            ampersand_tokens=ampersand_tokens,
            script=script,
            quality=quality,
            noise_prefix_stripped=noise_present,
        )

    def normalize_batch(
        self, raws: list[str]
    ) -> list[NameRecord]:
        """
        Normalise a list of raw name strings.

        This is a simple loop; no parallelism.  For very large batches,
        call normalize() inside a streaming loop rather than materializing
        the entire batch in memory.

        Parameters
        ----------
        raws:
            List of raw name strings.

        Returns
        -------
        list[NameRecord]
        """
        return [self.normalize(r) for r in raws]


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_DEFAULT_NORMALIZER = NameNormalizer()


def normalize_name(raw: str) -> NameRecord:
    """
    Module-level convenience function using a shared NameNormalizer instance.

    Parameters
    ----------
    raw:
        Raw business name string.

    Returns
    -------
    NameRecord
    """
    return _DEFAULT_NORMALIZER.normalize(raw)
