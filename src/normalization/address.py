"""
address.py — Multi-view address normalizer.

Produces an AddressRecord with multiple parallel representations and
lightweight structural signals extracted from a single raw address string.

Representations produced
------------------------
raw                  Original string, unmodified.
normalized           Lowercased, whitespace-normalised, leading noise stripped.
normalized_nfkd      NFKD-normalised form of 'normalized'.
tokens               Whitespace-split tokens of 'normalized'.
tokens_nfkd          NFKD tokens.
compact              'normalized' with all whitespace removed.
compact_alnum        'normalized' keeping only alphanumeric chars.

Structural signals (extracted, not normalised)
----------------------------------------------
numeric_tokens       Tokens that are purely numeric (or numeric-slash patterns).
alpha_tokens         Tokens that are purely alphabetic.
postal_candidates    5- or 6-digit tokens (likely ZIP/PIN codes).
house_number_candidates  Leading numeric tokens (potential house/building numbers).
has_po_box           True when 'PO BOX' pattern detected.
component_count      Number of comma-separated address components.
token_count          Number of whitespace-split tokens.
quality              Primary quality label (AQ_* constant).
script               ScriptInfo for the raw address.

Address normalisation strategy
-------------------------------
Unlike name normalisation, we do NOT aggressively expand address abbreviations.

Rationale:
  1.  Address abbreviations are highly ambiguous (ST = Street OR State).
  2.  Structural signal extraction (numeric tokens, postal code, components)
      is more useful than lexical expansion at this stage.
  3.  Phase 4 feature engineering will use token overlap, which is already
      effective without full expansion.
  4.  We DO strip obvious noise (## prefix, excessive repeated symbols).

Observed common patterns from training data:
  - '##218 WENRICK STREET' — ## prefix noise
  - 'NEAR SBI ATM, ...' — landmark references (India)
  - 'H.NO. 37, SECTOR-3' — Indian house number notation
  - 'NO 10 ENKAY SQUARE, ...' — 'NO' as house number marker
  - 'C/O MANISH KUMAR, ...' — care-of notation (India)
  - '3343-3345 J F SCOTT DR' — number range in house number
  - '72573 LAKE VIEW DRIVE, JACKSON, MN' — US standard
  - 'KH NO. -570/13, NEW DELHI' — Indian plot notation
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.normalization.common import (
    make_compact,
    normalize_unicode,
    normalize_whitespace,
    to_lowercase,
    tokenize,
    _RE_PO_BOX,
    _RE_POSTAL,
    _RE_HOUSE_NUMBER,
)
from src.normalization.quality import classify_address_quality
from src.normalization.scripts import ScriptInfo, get_script_info


# ---------------------------------------------------------------------------
# Patterns specific to address normalisation
# ---------------------------------------------------------------------------

# Leading noise in addresses: ## prefix (observed in data)
_RE_ADDR_LEADING_NOISE = re.compile(r"^[\#\@\*\-\.\+\/\\]+\s*")

# Repeated punctuation / separators (more than 2 in a row)
_RE_REPEATED_PUNCT = re.compile(r"([-,./|\\])\1{2,}")

# Purely numeric token (allowing slashes, hyphens for ranges/Indian plots)
_RE_NUMERIC_TOKEN = re.compile(r"^\d+([/-]\d+)*[A-Za-z]?$")

# Strips leading and trailing punctuation from an address token before classification
_RE_TOKEN_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")

# Splits a token that has an embedded dash/slash number
# e.g., 'no.-37' → strips leading 'no.-' to expose '37'
_RE_EMBEDDED_NUMBER = re.compile(r"\d+([/-]\d+)*[A-Za-z]?$")

# Purely alphabetic token (checked after punctuation stripping)
_RE_ALPHA_TOKEN = re.compile(r"^[A-Za-z\u0900-\uD7FF]+$")


def _strip_token_punct(tok: str) -> str:
    """Strip leading and trailing punctuation from an address token."""
    return _RE_TOKEN_PUNCT.sub("", tok)


# ---------------------------------------------------------------------------
# AddressRecord dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AddressRecord:
    """
    All multi-view representations and structural signals for one address.

    Fields
    ------
    raw:
        Original address string from the TSV file.

    normalized:
        Working form: strip leading ## noise → normalize whitespace → lowercase.

    normalized_nfkd:
        NFKD-normalised version of 'normalized'.

    tokens:
        Whitespace-split tokens of 'normalized'.

    tokens_nfkd:
        Whitespace-split tokens of 'normalized_nfkd'.

    compact:
        'normalized' with all whitespace removed.

    compact_alnum:
        'normalized' keeping only alphanumeric + Indic-script characters.

    numeric_tokens:
        Tokens from 'normalized' that match a numeric pattern
        (e.g., '123', '4/5', '37-40', '123A').
        These often encode house numbers, plot numbers, etc.

    alpha_tokens:
        Tokens from 'normalized' that are purely alphabetic (any script).
        Useful for street/locality name comparison.

    postal_candidates:
        5- or 6-digit substrings found in the address — potential ZIP (5-digit)
        or PIN (6-digit) codes.  Not validated against any postal database.

    house_number_candidates:
        Leading numeric-pattern tokens (first 1–3 tokens of the address that
        match _RE_NUMERIC_TOKEN or begin with a digit).  In US addresses, the
        house number is usually the first token; in Indian addresses it may
        appear after 'NO', 'H.NO.', 'PLOT NO', etc.

    has_po_box:
        True when the address contains 'PO BOX', 'P.O. BOX', etc.

    component_count:
        Number of comma-separated address components.  Useful as a rough
        address complexity indicator.

    token_count:
        Number of whitespace-split tokens.

    quality:
        Primary quality label (AQ_GOOD, AQ_MISSING, AQ_NON_LATIN,
        AQ_MIXED_SCRIPT, AQ_CORRUPTED, AQ_OTHER).

    script:
        ScriptInfo for the raw address string.
    """

    raw: str
    normalized: str
    normalized_nfkd: str
    tokens: tuple[str, ...]
    tokens_nfkd: tuple[str, ...]
    compact: str
    compact_alnum: str
    numeric_tokens: tuple[str, ...]
    alpha_tokens: tuple[str, ...]
    postal_candidates: tuple[str, ...]
    house_number_candidates: tuple[str, ...]
    has_po_box: bool
    component_count: int
    token_count: int
    quality: str
    script: ScriptInfo


# ---------------------------------------------------------------------------
# AddressNormalizer
# ---------------------------------------------------------------------------

class AddressNormalizer:
    """
    Stateless normalizer that converts a raw address string into an AddressRecord.

    Instantiate once and reuse across millions of records.

    Example
    -------
    >>> norm = AddressNormalizer()
    >>> rec = norm.normalize('##218 WENRICK STREET, COVINGTON, OH')
    >>> rec.normalized
    '218 wenrick street, covington, oh'
    >>> rec.numeric_tokens
    ('218',)
    >>> rec.component_count
    3
    >>> rec.has_po_box
    False
    """

    def normalize(self, raw: str) -> AddressRecord:
        """
        Normalise a single raw address string into an AddressRecord.

        Parameters
        ----------
        raw:
            The original address string from the TSV file.

        Returns
        -------
        AddressRecord
        """
        # --- Step 0: Script info on raw (before any modification) ---
        script = get_script_info(raw)

        # --- Step 1: Quality classification (uses raw) ---
        quality = classify_address_quality(raw, script)

        # --- Step 2: Strip leading ## noise ---
        stripped = _RE_ADDR_LEADING_NOISE.sub("", raw)

        # --- Step 3: Normalize repeated punctuation ---
        stripped = _RE_REPEATED_PUNCT.sub(r"\1", stripped)

        # --- Step 4: Normalize whitespace ---
        ws_clean = normalize_whitespace(stripped)

        # --- Step 5: Lowercase ---
        normalized = to_lowercase(ws_clean)

        # --- Step 6: NFKD ---
        normalized_nfkd = normalize_unicode(normalized, form="NFKD")

        # --- Step 7: Tokenize ---
        tokens = tokenize(normalized)
        tokens_nfkd = tokenize(normalized_nfkd)

        # --- Step 8: Compact representations ---
        compact = make_compact(normalized, alnum_only=False)
        compact_alnum = make_compact(normalized, alnum_only=True)

        # --- Step 9: Structural signal extraction ---
        # Strip trailing punctuation from tokens before classification
        clean_tokens = tuple(_strip_token_punct(t) for t in tokens)
        clean_tokens = tuple(t for t in clean_tokens if t)  # remove empty

        numeric_tokens = tuple(
            t for t in clean_tokens if _RE_NUMERIC_TOKEN.match(t)
        )
        alpha_tokens = tuple(
            t for t in clean_tokens
            if _RE_ALPHA_TOKEN.match(t) and len(t) > 1
        )

        # Postal code candidates from the raw/normalized string
        postal_candidates = tuple(
            dict.fromkeys(_RE_POSTAL.findall(normalized))  # de-duplicate
        )

        # House number candidates: leading tokens that look numeric
        house_candidates = []
        for tok in clean_tokens[:4]:  # check only first 4 cleaned tokens
            if _RE_NUMERIC_TOKEN.match(tok) or (tok and tok[0].isdigit()):
                house_candidates.append(tok)
            else:
                break  # stop at first non-numeric leading token

        # Also check tokens immediately after 'no', 'no.', 'h.no', 'plot no', etc.
        house_markers = {"no", "no.", "h.no.", "h.no", "hno", "plot", "kh"}
        for i, tok in enumerate(clean_tokens[:-1]):
            if tok in house_markers:
                candidate = clean_tokens[i + 1]
                if candidate and (_RE_NUMERIC_TOKEN.match(candidate) or candidate[0].isdigit()):
                    if candidate not in house_candidates:
                        house_candidates.append(candidate)

        # Handle embedded number in token like 'no.-37' → extract '37'
        for tok in clean_tokens[:5]:
            m = _RE_EMBEDDED_NUMBER.search(tok)
            if m and not _RE_NUMERIC_TOKEN.match(tok):
                extracted = m.group(0)
                if extracted not in house_candidates:
                    house_candidates.append(extracted)

        house_number_candidates = tuple(house_candidates)

        # PO Box detection
        has_po_box = bool(_RE_PO_BOX.search(raw))

        # Component count (comma-based)
        component_count = normalized.count(",") + 1 if normalized else 0

        return AddressRecord(
            raw=raw,
            normalized=normalized,
            normalized_nfkd=normalized_nfkd,
            tokens=tokens,
            tokens_nfkd=tokens_nfkd,
            compact=compact,
            compact_alnum=compact_alnum,
            numeric_tokens=numeric_tokens,
            alpha_tokens=alpha_tokens,
            postal_candidates=postal_candidates,
            house_number_candidates=house_number_candidates,
            has_po_box=has_po_box,
            component_count=component_count,
            token_count=len(tokens),
            quality=quality,
            script=script,
        )

    def normalize_batch(self, raws: list[str]) -> list[AddressRecord]:
        """Normalise a list of raw address strings."""
        return [self.normalize(r) for r in raws]


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_DEFAULT_NORMALIZER = AddressNormalizer()


def normalize_address(raw: str) -> AddressRecord:
    """
    Module-level convenience using a shared AddressNormalizer instance.

    Parameters
    ----------
    raw:
        Raw address string.

    Returns
    -------
    AddressRecord
    """
    return _DEFAULT_NORMALIZER.normalize(raw)
