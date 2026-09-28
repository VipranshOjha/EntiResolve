"""
cross_script.py — Auxiliary cross-script correspondence learned strictly from training data.

STRICT CONSTRAINTS:
- No external transliteration libraries, dictionaries, or APIs.
- Learned ONLY from ground-truth training S1 ↔ Target pairs.
- Validation split is strictly excluded.
- Produces an auxiliary similarity score in [0.0, 1.0] for cross-script pairs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.normalization.record import BusinessRecord


class CrossScriptAligner:
    """
    Learns empirical token and n-gram associations between Latin and non-Latin names
    strictly from confirmed training ground-truth entity pairs.
    """

    def __init__(self, min_cooccurrence: int = 2, min_dice_threshold: float = 0.1) -> None:
        self.min_cooccurrence = min_cooccurrence
        self.min_dice_threshold = min_dice_threshold
        # Mapping from (latin_token, non_latin_token) -> Dice association score
        self.token_associations: dict[str, dict[str, float]] = {}
        self.fitted: bool = False

    def fit(
        self,
        true_pairs: Iterable[tuple[BusinessRecord, BusinessRecord]],
        disallowed_entity_ids: set[str] | None = None,
    ) -> CrossScriptAligner:
        """
        Fit associations on ground-truth training pairs.

        Parameters
        ----------
        true_pairs:
            Iterable of (s1_record, target_record) ground truth pairs from training split only.
        disallowed_entity_ids:
            Optional set of entity IDs (e.g. validation or test) that must never enter training.
            If any pair contains an ID in this set, a ValueError is raised.
        """
        if disallowed_entity_ids:
            disallowed_set = set(disallowed_entity_ids)
            for s1, target in true_pairs:
                if s1.entity_id in disallowed_set or target.entity_id in disallowed_set:
                    raise ValueError(
                        f"Validation leakage detected! Entity {s1.entity_id} or {target.entity_id} "
                        f"is present in disallowed_entity_ids."
                    )

        pair_counts: Counter[tuple[str, str]] = Counter()
        latin_counts: Counter[str] = Counter()
        non_latin_counts: Counter[str] = Counter()

        for s1, target in true_pairs:
            s1_non_lat = s1.name.script.is_non_latin or s1.name.script.is_mixed_script
            tgt_non_lat = target.name.script.is_non_latin or target.name.script.is_mixed_script

            # Only consider cross-script or non-Latin pairs
            if not (s1_non_lat or tgt_non_lat):
                continue

            # Identify Latin and non-Latin token sets
            if not s1_non_lat and tgt_non_lat:
                latin_tokens = set(s1.name.tokens)
                non_latin_tokens = set(target.name.tokens)
            elif s1_non_lat and not tgt_non_lat:
                latin_tokens = set(target.name.tokens)
                non_latin_tokens = set(s1.name.tokens)
            else:
                # Both contain non-Latin; also link their tokens
                latin_tokens = set(s1.name.tokens)
                non_latin_tokens = set(target.name.tokens)

            # Filter short tokens
            lat_valid = [t for t in latin_tokens if len(t) >= 2]
            non_lat_valid = [t for t in non_latin_tokens if len(t) >= 2]

            for lt in lat_valid:
                latin_counts[lt] += 1
            for nt in non_lat_valid:
                non_latin_counts[nt] += 1

            for lt in lat_valid:
                for nt in non_lat_valid:
                    pair_counts[(lt, nt)] += 1

        # Compute Dice coefficient: 2 * C(lt, nt) / (C(lt) + C(nt))
        associations: dict[str, dict[str, float]] = {}
        for (lt, nt), count in pair_counts.items():
            if count >= self.min_cooccurrence:
                dice = (2.0 * count) / (latin_counts[lt] + non_latin_counts[nt])
                if dice >= self.min_dice_threshold:
                    if lt not in associations:
                        associations[lt] = {}
                    associations[lt][nt] = round(dice, 4)

        self.token_associations = associations
        self.fitted = True
        return self

    def compute_similarity(self, s1: BusinessRecord, target: BusinessRecord) -> float:
        """
        Compute learned cross-script association score in [0.0, 1.0].
        If neither record has non-Latin script, returns 0.0.
        """
        if not self.fitted or not self.token_associations:
            return 0.0

        s1_non_lat = s1.name.script.is_non_latin or s1.name.script.is_mixed_script
        tgt_non_lat = target.name.script.is_non_latin or target.name.script.is_mixed_script

        # If both are purely Latin, cross-script is not applicable
        if not s1_non_lat and not tgt_non_lat:
            return 0.0

        if not s1_non_lat and tgt_non_lat:
            lat_tokens = s1.name.tokens
            non_lat_tokens = set(target.name.tokens)
        elif s1_non_lat and not tgt_non_lat:
            lat_tokens = target.name.tokens
            non_lat_tokens = set(s1.name.tokens)
        else:
            lat_tokens = s1.name.tokens
            non_lat_tokens = set(target.name.tokens)

        if not lat_tokens or not non_lat_tokens:
            return 0.0

        best_scores: list[float] = []
        for lt in lat_tokens:
            if lt in self.token_associations:
                matched_dice = [
                    self.token_associations[lt][nt]
                    for nt in non_lat_tokens
                    if nt in self.token_associations[lt]
                ]
                if matched_dice:
                    best_scores.append(max(matched_dice))

        if not best_scores:
            return 0.0

        return float(sum(best_scores) / max(len(lat_tokens), 1))

    def save(self, filepath: str | Path) -> None:
        """Serialize associations to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "min_cooccurrence": self.min_cooccurrence,
                    "min_dice_threshold": self.min_dice_threshold,
                    "associations": self.token_associations,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, filepath: str | Path) -> CrossScriptAligner:
        """Load associations from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        aligner = cls(
            min_cooccurrence=data["min_cooccurrence"],
            min_dice_threshold=data["min_dice_threshold"],
        )
        aligner.token_associations = data["associations"]
        aligner.fitted = True
        return aligner
