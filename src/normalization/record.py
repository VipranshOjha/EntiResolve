"""
record.py — BusinessRecord dataclass and NormalizationPipeline.

A BusinessRecord is the complete multi-view representation of one entity
from any source file (Source 1, 2, or 3).  It aggregates the NameRecord
and AddressRecord produced by their respective normalizers, along with
the preserved entity_id and country.

NormalizationPipeline is a lightweight wrapper that:
  1. Accepts raw TSV field values (entity_id, name, address, country).
  2. Delegates to NameNormalizer and AddressNormalizer.
  3. Returns a BusinessRecord.

The pipeline is stateless and reusable across millions of records.
It does NOT load data from disk; callers are responsible for reading TSV rows
and passing field values here.

Usage example
-------------
    from src.normalization.record import NormalizationPipeline

    pipeline = NormalizationPipeline()

    # Process a single record
    record = pipeline.normalize("S2-12345", "Sai Traders Pvt. Ltd.",
                                "House No.-37, Rewari, Haryana", "India")
    print(record.name.tokens_suffix_norm)
    print(record.address.postal_candidates)
    print(record.name.script.dominant_script)

    # Process a TSV file in streaming fashion (preferred for large files)
    for record in pipeline.stream_tsv(path_to_source2_tsv):
        # ... blocking / feature extraction ...
        pass
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator

from src.normalization.address import AddressNormalizer, AddressRecord
from src.normalization.name import NameNormalizer, NameRecord
from src.normalization.phone import PhoneNormalizer, PhoneRecord


# ---------------------------------------------------------------------------
# BusinessRecord dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BusinessRecord:
    """
    Complete multi-view representation of one business entity.

    Attributes
    ----------
    entity_id:
        The original entity ID from the source file (e.g., 'S1-123456').
        The source (S1, S2, S3) is encoded in the prefix.
    name:
        NameRecord — all representations of the business name.
    address:
        AddressRecord — all representations of the business address.
    country:
        Country label as supplied in the TSV file (open-set string).
        Preserved exactly; never hard-coded or filtered.
    phone:
        PhoneRecord — normalized phone representations (optional).
    """

    entity_id: str
    name: NameRecord
    address: AddressRecord
    country: str
    phone: PhoneRecord | None = None

    @property
    def source(self) -> str:
        """Source prefix in lowercase: 's1', 's2', 's3'."""
        return self.entity_id.split("-")[0].lower() if "-" in self.entity_id else ""

    @property
    def has_phone(self) -> bool:
        """True if record has a non-empty phone number."""
        return self.phone is not None and self.phone.has_phone

    @property
    def dominant_script(self) -> str:
        """Dominant script of the business name."""
        return self.name.script.dominant_script

    @property
    def is_mixed_script(self) -> bool:
        """True if the business name contains mixed scripts."""
        return self.name.script.is_mixed_script

    @property
    def is_complete_record(self) -> bool:
        """True if both name and address have valid non-empty content."""
        return (
            self.name.quality in ("GOOD", "NON_LATIN")
            and self.address.quality in ("GOOD", "NON_LATIN")
        )


# ---------------------------------------------------------------------------
# NormalizationPipeline
# ---------------------------------------------------------------------------

class NormalizationPipeline:
    """
    Stateless pipeline that converts raw TSV field values into BusinessRecords.

    Instantiate once per process; reuse across all records.

    The pipeline holds stateless normalizer instances (NameNormalizer,
    AddressNormalizer, PhoneNormalizer) that are called once per field per record.

    Parameters
    ----------
    None (configuration is embedded in the normalizer modules).
    """

    def __init__(self) -> None:
        self._name_norm = NameNormalizer()
        self._addr_norm = AddressNormalizer()
        self._phone_norm = PhoneNormalizer()

    def normalize(
        self,
        entity_id: str,
        raw_name: str,
        raw_address: str,
        country: str,
        raw_phone: str | None = None,
    ) -> BusinessRecord:
        """
        Produce a BusinessRecord from raw TSV field values.

        Parameters
        ----------
        entity_id:
            The entity_id column value (e.g., 'S2-987654').
        raw_name:
            The business_name column value (may be empty, garbled, non-Latin).
        raw_address:
            The business_address column value (may be empty).
        country:
            The country column value (open-set string, preserved verbatim).
        raw_phone:
            Optional raw phone string.

        Returns
        -------
        BusinessRecord
        """
        name_record = self._name_norm.normalize(raw_name)
        addr_record = self._addr_norm.normalize(raw_address)
        phone_record = self._phone_norm.normalize(raw_phone) if raw_phone is not None else None
        return BusinessRecord(
            entity_id=entity_id,
            name=name_record,
            address=addr_record,
            country=country.strip(),
            phone=phone_record,
        )

    def stream_tsv(
        self,
        path: Path,
        max_records: int | None = None,
    ) -> Generator[BusinessRecord, None, None]:
        """
        Lazily stream BusinessRecords from a source TSV file.

        Reads the file row-by-row without loading it all into memory.
        Suitable for use on files with tens of millions of records.

        Expected TSV columns (0-indexed):
            0: entity_id
            1: business_name
            2: business_address
            3: country

        Parameters
        ----------
        path:
            Path to a source TSV file (train_source1.tsv etc.).
        max_records:
            If set, stop after this many records (useful for sampling/testing).

        Yields
        ------
        BusinessRecord
            One per data row (header is skipped).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TSV not found: {path}")

        count = 0
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            next(reader, None)   # skip header
            for row in reader:
                if not row:
                    continue
                if max_records is not None and count >= max_records:
                    break
                # Safely unpack columns; pad with empty strings if short
                entity_id = row[0].strip() if len(row) > 0 else ""
                raw_name = row[1] if len(row) > 1 else ""
                raw_address = row[2] if len(row) > 2 else ""
                country = row[3].strip() if len(row) > 3 else ""
                yield self.normalize(entity_id, raw_name, raw_address, country)
                count += 1

    def stream_records_for_ids(
        self,
        path: Path,
        id_set: set[str],
    ) -> Generator[BusinessRecord, None, None]:
        """
        Stream BusinessRecords from a TSV, filtering to only the given entity IDs.

        Useful for loading only the training partition of Source 1 records
        without materializing the full file.

        Parameters
        ----------
        path:
            Path to a source TSV file.
        id_set:
            Set of entity IDs to include.

        Yields
        ------
        BusinessRecord
        """
        for record in self.stream_tsv(path):
            if record.entity_id in id_set:
                yield record


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_DEFAULT_PIPELINE = NormalizationPipeline()


def normalize_record(
    entity_id: str,
    raw_name: str,
    raw_address: str,
    country: str,
    raw_phone: str | None = None,
) -> BusinessRecord:
    """
    Module-level convenience using a shared NormalizationPipeline instance.

    Parameters
    ----------
    entity_id, raw_name, raw_address, country:
        Raw field values from a source TSV row.
    raw_phone:
        Optional raw phone number string.

    Returns
    -------
    BusinessRecord
    """
    return _DEFAULT_PIPELINE.normalize(entity_id, raw_name, raw_address, country, raw_phone)
