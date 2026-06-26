"""
Generate balanced train/val/test splits for scenario-tagged datasets.

Supports stratified splitting by scenario attributes, geographic splitting
to prevent location leakage, and temporal splitting by recording timestamps.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from .database import ScenarioDatabase


class SplitGenerator:
    """
    Generator for balanced dataset splits.

    Provides multiple splitting strategies to ensure representative
    train/val/test partitions while avoiding data leakage.
    """

    def generate_balanced_split(
        self,
        database: ScenarioDatabase,
        ratios: Dict[str, float],
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """
        Generate a stratified split balancing scenario attributes across partitions.

        Uses iterative stratification: for each recording, assigns it to the split
        that is most underrepresented for its scenario tags.

        Args:
            database: The scenario database to split.
            ratios: Dictionary mapping split names to desired proportions
                    (e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15}).
            seed: Random seed for reproducibility.

        Returns:
            Dictionary mapping split names to lists of recording IDs.
        """
        rng = random.Random(seed)

        all_recordings = database.get_all_recordings()
        if not all_recordings:
            return {name: [] for name in ratios}

        # Normalize ratios
        total_ratio = sum(ratios.values())
        normalized = {name: r / total_ratio for name, r in ratios.items()}
        split_names = list(normalized.keys())

        # Build tag profile for each recording
        recording_ids = [rec["id"] for rec in all_recordings]
        recording_tags: Dict[str, Set[str]] = {}
        for rec_id in recording_ids:
            tags = database.get_tags_for_recording(rec_id)
            recording_tags[rec_id] = {tag["node_id"] for tag in tags}

        # Collect all unique tags
        all_tags = set()
        for tags in recording_tags.values():
            all_tags.update(tags)

        # Target counts per tag per split
        tag_recording_counts: Dict[str, int] = defaultdict(int)
        for tags in recording_tags.values():
            for tag in tags:
                tag_recording_counts[tag] += 1

        # Initialize splits
        splits: Dict[str, List[str]] = {name: [] for name in split_names}
        split_tag_counts: Dict[str, Dict[str, int]] = {
            name: defaultdict(int) for name in split_names
        }
        split_sizes: Dict[str, int] = {name: 0 for name in split_names}

        # Target sizes
        n = len(recording_ids)
        target_sizes = {name: int(n * normalized[name]) for name in split_names}
        # Distribute remainder to ensure all recordings are assigned
        remainder = n - sum(target_sizes.values())
        for i, name in enumerate(split_names):
            if i < remainder:
                target_sizes[name] += 1

        # Shuffle recordings for randomness
        shuffled_ids = recording_ids.copy()
        rng.shuffle(shuffled_ids)

        # Assign recordings using stratified approach
        for rec_id in shuffled_ids:
            tags = recording_tags[rec_id]

            # Calculate imbalance score for each split
            best_split = None
            best_score = float("inf")

            for name in split_names:
                if split_sizes[name] >= target_sizes[name]:
                    continue

                # Score: how much this split needs this recording's tags
                # Lower score = more needed
                score = 0.0
                for tag in tags:
                    if tag_recording_counts[tag] > 0:
                        current_ratio = (
                            split_tag_counts[name][tag] / tag_recording_counts[tag]
                        )
                        target_ratio = normalized[name]
                        # How far above target is this split for this tag
                        score += current_ratio - target_ratio
                    else:
                        score += 0.0

                # Also factor in overall size balance
                size_ratio = split_sizes[name] / max(target_sizes[name], 1)
                score += size_ratio * 0.1

                if score < best_score:
                    best_score = score
                    best_split = name

            # Fallback: assign to the split with most remaining capacity
            if best_split is None:
                best_split = max(
                    split_names,
                    key=lambda s: target_sizes[s] - split_sizes[s],
                )

            splits[best_split].append(rec_id)
            split_sizes[best_split] += 1
            for tag in tags:
                split_tag_counts[best_split][tag] += 1

        return splits

    def generate_geographic_split(
        self,
        database: ScenarioDatabase,
        location_clusters: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        Generate splits by assigning entire geographic locations to splits.

        This prevents location leakage: no location appears in more than one split.

        Args:
            database: The scenario database.
            location_clusters: Dictionary mapping split names to lists of location
                              identifiers. Each location's recordings go to that split.
                              Example: {'train': ['munich', 'berlin'], 'val': ['hamburg'],
                                        'test': ['frankfurt']}

        Returns:
            Dictionary mapping split names to lists of recording IDs.
        """
        all_recordings = database.get_all_recordings()

        # Build location -> recording_id mapping
        location_to_recordings: Dict[Optional[str], List[str]] = defaultdict(list)
        for rec in all_recordings:
            location_to_recordings[rec["location"]].append(rec["id"])

        splits: Dict[str, List[str]] = {name: [] for name in location_clusters}

        for split_name, locations in location_clusters.items():
            for loc in locations:
                # Match case-insensitively
                for db_loc, rec_ids in location_to_recordings.items():
                    if db_loc is not None and db_loc.lower() == loc.lower():
                        splits[split_name].extend(rec_ids)

        # Sort for determinism
        for name in splits:
            splits[name].sort()

        return splits

    def generate_temporal_split(
        self, database: ScenarioDatabase
    ) -> Dict[str, List[str]]:
        """
        Generate splits based on recording timestamps.

        Sorts recordings chronologically and assigns the earliest portion
        to train, middle to val, and latest to test. This simulates a
        realistic deployment scenario where training data is older than
        evaluation data.

        Default ratios: train=70%, val=15%, test=15%.

        Args:
            database: The scenario database.

        Returns:
            Dictionary mapping 'train', 'val', 'test' to lists of recording IDs.
        """
        all_recordings = database.get_all_recordings()

        # Separate recordings with and without timestamps
        with_timestamp = [
            rec for rec in all_recordings if rec["timestamp"] is not None
        ]
        without_timestamp = [
            rec for rec in all_recordings if rec["timestamp"] is None
        ]

        # Sort by timestamp
        with_timestamp.sort(key=lambda r: r["timestamp"])

        # Combine: timestamped first (sorted), then non-timestamped (assigned to train)
        ordered_ids = [rec["id"] for rec in with_timestamp]

        n = len(ordered_ids)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        splits: Dict[str, List[str]] = {
            "train": ordered_ids[:train_end],
            "val": ordered_ids[train_end:val_end],
            "test": ordered_ids[val_end:],
        }

        # Add recordings without timestamps to train
        splits["train"].extend(rec["id"] for rec in without_timestamp)

        return splits

    def validate_split(
        self,
        split: Dict[str, List[str]],
        database: ScenarioDatabase,
    ) -> Dict[str, Any]:
        """
        Validate a split for quality: balance, overlap, and coverage.

        Checks:
        - No recording appears in multiple splits (overlap)
        - All recordings are assigned (coverage)
        - Tag distribution is approximately proportional (balance)

        Args:
            split: Dictionary mapping split names to lists of recording IDs.
            database: The scenario database for reference.

        Returns:
            Dictionary with validation results including:
            - 'valid': bool, whether all checks pass
            - 'overlap': list of recording IDs appearing in multiple splits
            - 'coverage': float, fraction of all recordings that are assigned
            - 'unassigned': list of recording IDs not in any split
            - 'split_sizes': dict of split name -> count
            - 'tag_balance': dict showing tag distribution per split
        """
        all_recordings = database.get_all_recordings()
        all_ids = {rec["id"] for rec in all_recordings}
        total = len(all_ids)

        # Check overlap
        seen: Dict[str, str] = {}
        overlapping: List[str] = []
        for split_name, rec_ids in split.items():
            for rec_id in rec_ids:
                if rec_id in seen:
                    overlapping.append(rec_id)
                else:
                    seen[rec_id] = split_name

        # Check coverage
        assigned_ids = set(seen.keys())
        unassigned = sorted(all_ids - assigned_ids)
        coverage = len(assigned_ids & all_ids) / max(total, 1)

        # Split sizes
        split_sizes = {name: len(ids) for name, ids in split.items()}

        # Tag balance analysis
        tag_balance: Dict[str, Dict[str, int]] = {}
        for split_name, rec_ids in split.items():
            tag_counts: Dict[str, int] = defaultdict(int)
            for rec_id in rec_ids:
                tags = database.get_tags_for_recording(rec_id)
                for tag in tags:
                    tag_counts[tag["node_id"]] += 1
            tag_balance[split_name] = dict(tag_counts)

        is_valid = len(overlapping) == 0 and coverage >= 1.0

        return {
            "valid": is_valid,
            "overlap": sorted(set(overlapping)),
            "coverage": coverage,
            "unassigned": unassigned,
            "split_sizes": split_sizes,
            "tag_balance": tag_balance,
        }

    def assign_splits(
        self,
        database: ScenarioDatabase,
        split: Dict[str, List[str]],
        version: str,
    ) -> None:
        """
        Store split assignments in the database.

        Args:
            database: The scenario database.
            split: Dictionary mapping split names to lists of recording IDs.
            version: Version identifier for this split assignment.
        """
        for split_name, rec_ids in split.items():
            for rec_id in rec_ids:
                database.add_split_assignment(rec_id, split_name, version)
