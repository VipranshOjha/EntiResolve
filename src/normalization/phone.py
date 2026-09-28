"""
phone.py — Phone number normalisation and representation.

Provides PhoneRecord and PhoneNormalizer to handle phone numbers if present
in records. Extracts raw digits, normalized suffixes (last 10, last 7 digits)
to handle international prefixes, trunk prefixes (0), and formatting characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RE_NON_DIGITS = re.compile(r"\D+")


@dataclass(frozen=True, slots=True)
class PhoneRecord:
    """
    Normalized representation of a phone number.

    Attributes
    ----------
    raw:
        Original raw phone string as provided.
    digits:
        All digits extracted from the raw string (no spaces, dashes, parentheses).
    last_10:
        Last 10 digits (standard national number in US, India, etc.), or all
        digits if fewer than 10.
    last_7:
        Last 7 digits (local subscriber number), or all digits if fewer than 7.
    has_phone:
        True if digits contains at least one digit.
    is_valid:
        True if number of digits is between 7 and 15 (standard ITU-T E.164 range).
    """

    raw: str
    digits: str
    last_10: str
    last_7: str
    has_phone: bool
    is_valid: bool


class PhoneNormalizer:
    """
    Stateless normalizer for phone strings.
    """

    def normalize(self, raw: str | None) -> PhoneRecord:
        """
        Normalize a raw phone number.

        Parameters
        ----------
        raw:
            Raw phone string, e.g. "+1 (800) 555-0199", "09876543210", or None.

        Returns
        -------
        PhoneRecord
        """
        if raw is None:
            raw = ""
        else:
            raw = str(raw).strip()

        digits = _RE_NON_DIGITS.sub("", raw)
        has_phone = len(digits) > 0
        last_10 = digits[-10:] if len(digits) >= 10 else digits
        last_7 = digits[-7:] if len(digits) >= 7 else digits
        is_valid = 7 <= len(digits) <= 15

        return PhoneRecord(
            raw=raw,
            digits=digits,
            last_10=last_10,
            last_7=last_7,
            has_phone=has_phone,
            is_valid=is_valid,
        )


_DEFAULT_PHONE_NORMALIZER = PhoneNormalizer()


def normalize_phone(raw: str | None) -> PhoneRecord:
    """
    Module-level convenience function for phone normalization.
    """
    return _DEFAULT_PHONE_NORMALIZER.normalize(raw)
