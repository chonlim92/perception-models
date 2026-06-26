"""
Coverage analysis for scenario annotations.

Analyzes which scenario combinations are represented in the dataset,
identifies gaps, and suggests data collection priorities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from ..taxonomy.scenario_schema import ScenarioAnnotation, ScenarioTag


@dataclass
class CoverageGap:
    """Represents a missing or underrepresented scenario combination."""

    attributes: Dict[str, str]
    expected: bool
    actual_count: int
    description: str


@dataclass
class CollectionPriority:
    """A ranked priority for additional data collection."""

    rank: int
    attributes: Dict[str, str]
    current_count: int
    target_count: int
    reason: str
    priority_score: float


def _extract_layer(node_id: str) -> int:
    """Extract layer number from a node_id like 'L4.3.1' -> 4."""
    if node_id.startswith("L") and len(node_id) > 1:
        digit_chars = ""
        for ch in node_id[1:]:
            if ch.isdigit():
                digit_chars += ch
            else:
                break
        if digit_chars:
            return int(digit_chars)
    return 0


def _extract_category(node_id: str) -> str:
    """
    Extract the category-level node ID.

    E.g., 'L4.3.1' -> 'L4.3' (the parent category).
    """
    parts = node_id.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return node_id


def _get_attribute_value(annotation: ScenarioAnnotation, attr: str) -> List[str]:
    """
    Get attribute values from an annotation based on attribute specification.

    Attributes can be:
    - 'layer_N' : extract all node_ids at layer N
    - 'node_prefix_X' : extract node_ids starting with prefix X
    - Direct node_id patterns
    """
    values: List[str] = []

    if attr.startswith("layer_"):
        try:
            layer_num = int(attr.split("_")[1])
        except (IndexError, ValueError):
            return values
        for tag in annotation.tags:
            if _extract_layer(tag.node_id) == layer_num:
                values.append(tag.node_id)
    elif attr.startswith("category_"):
        prefix = attr.replace("category_", "")
        for tag in annotation.tags:
            if tag.node_id.startswith(prefix):
                values.append(tag.node_id)
    else:
        # Treat attr as a layer number or node prefix directly
        for tag in annotation.tags:
            if tag.node_id.startswith(attr):
                values.append(tag.node_id)

    return values


class CoverageAnalyzer:
    """
    Analyze scenario coverage and identify gaps in driving data annotations.

    Provides cross-tabulation of scenario attributes, gap identification,
    and collection priority suggestions.
    """

    def compute_coverage_matrix(
        self,
        annotations: List[ScenarioAnnotation],
        dim1_attr: str,
        dim2_attr: str,
    ) -> pd.DataFrame:
        """
        Compute a cross-tabulation matrix of two scenario attributes.

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            All annotations to analyze.
        dim1_attr : str
            First dimension attribute specification (rows).
            Can be 'layer_N', 'category_X', or a node_id prefix.
        dim2_attr : str
            Second dimension attribute specification (columns).

        Returns
        -------
        pd.DataFrame
            Cross-tabulation with counts of recordings matching each
            combination. Rows are dim1 values, columns are dim2 values.
        """
        # Collect all (dim1_value, dim2_value) pairs
        pairs: List[Tuple[str, str]] = []

        for ann in annotations:
            dim1_values = _get_attribute_value(ann, dim1_attr)
            dim2_values = _get_attribute_value(ann, dim2_attr)

            # Generate all combinations of values found
            for v1 in dim1_values:
                for v2 in dim2_values:
                    pairs.append((v1, v2))

        if not pairs:
            return pd.DataFrame()

        # Build the cross-tabulation
        df = pd.DataFrame(pairs, columns=["dim1", "dim2"])
        matrix = pd.crosstab(df["dim1"], df["dim2"])
        matrix.index.name = dim1_attr
        matrix.columns.name = dim2_attr

        return matrix

    def find_gaps(
        self,
        annotations: List[ScenarioAnnotation],
        required_attributes: List[str],
    ) -> List[CoverageGap]:
        """
        Identify missing combinations of required attributes.

        For each pair of required attributes, checks which combinations
        have zero recordings. Returns all gaps found.

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            All annotations to analyze.
        required_attributes : list of str
            Attribute specifications that should all be cross-covered.

        Returns
        -------
        list of CoverageGap
            All identified gaps (missing combinations).
        """
        if len(required_attributes) < 2:
            return []

        # Collect the unique values for each attribute
        attr_values: Dict[str, Set[str]] = defaultdict(set)
        for ann in annotations:
            for attr in required_attributes:
                values = _get_attribute_value(ann, attr)
                for v in values:
                    attr_values[attr].add(v)

        # Collect actual observed combinations
        observed_combos: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}

        for attr_a, attr_b in combinations(required_attributes, 2):
            combo_key = (attr_a, attr_b)
            observed_combos[combo_key] = set()

            for ann in annotations:
                values_a = _get_attribute_value(ann, attr_a)
                values_b = _get_attribute_value(ann, attr_b)
                for va in values_a:
                    for vb in values_b:
                        observed_combos[combo_key].add((va, vb))

        # Find gaps (expected but unobserved combinations)
        gaps: List[CoverageGap] = []

        for attr_a, attr_b in combinations(required_attributes, 2):
            combo_key = (attr_a, attr_b)
            all_possible = set(product(attr_values[attr_a], attr_values[attr_b]))
            missing = all_possible - observed_combos[combo_key]

            for va, vb in sorted(missing):
                gaps.append(CoverageGap(
                    attributes={attr_a: va, attr_b: vb},
                    expected=True,
                    actual_count=0,
                    description=(
                        f"No recordings with {attr_a}={va} AND {attr_b}={vb}"
                    ),
                ))

        return gaps

    def compute_layer_coverage(
        self,
        annotations: List[ScenarioAnnotation],
        layer: int,
    ) -> Dict[str, int]:
        """
        Count recordings per node at a specific layer of the scenario tree.

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            All annotations to analyze.
        layer : int
            Layer number (1-6) to analyze.

        Returns
        -------
        dict
            Mapping from node_id to count of recordings containing that node.
        """
        counts: Counter = Counter()

        for ann in annotations:
            # Use a set to avoid double-counting the same node in one recording
            seen_nodes: Set[str] = set()
            for tag in ann.tags:
                if _extract_layer(tag.node_id) == layer:
                    seen_nodes.add(tag.node_id)
            for node_id in seen_nodes:
                counts[node_id] += 1

        return dict(counts)

    def suggest_collection_priorities(
        self,
        annotations: List[ScenarioAnnotation],
    ) -> List[CollectionPriority]:
        """
        Rank underrepresented scenarios to suggest data collection priorities.

        Analyzes node frequency across all layers and identifies nodes that
        are significantly below the average representation level.

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            All annotations to analyze.

        Returns
        -------
        list of CollectionPriority
            Ranked priorities for additional data collection, sorted by
            priority score (highest priority first).
        """
        if not annotations:
            return []

        # Count occurrences of each node across all annotations
        node_counts: Counter = Counter()
        total_annotations = len(annotations)

        for ann in annotations:
            seen_nodes: Set[str] = set()
            for tag in ann.tags:
                seen_nodes.add(tag.node_id)
            for node_id in seen_nodes:
                node_counts[node_id] += 1

        if not node_counts:
            return []

        # Compute statistics per layer
        layer_stats: Dict[int, Dict[str, float]] = {}
        for layer in range(1, 7):
            layer_nodes = {
                nid: count for nid, count in node_counts.items()
                if _extract_layer(nid) == layer
            }
            if layer_nodes:
                mean_count = sum(layer_nodes.values()) / len(layer_nodes)
                layer_stats[layer] = {
                    "mean": mean_count,
                    "max": max(layer_nodes.values()),
                }

        # Identify underrepresented nodes
        priorities: List[CollectionPriority] = []

        for node_id, count in node_counts.items():
            layer = _extract_layer(node_id)
            if layer not in layer_stats:
                continue

            mean_count = layer_stats[layer]["mean"]
            max_count = layer_stats[layer]["max"]

            # Priority score: how far below average this node is
            if mean_count > 0:
                deficit_ratio = 1.0 - (count / mean_count)
            else:
                deficit_ratio = 0.0

            if deficit_ratio <= 0:
                continue  # Not underrepresented

            target = int(mean_count * 1.5)  # Target 1.5x average as collection goal
            priority_score = deficit_ratio * (max_count / max(count, 1))

            priorities.append(CollectionPriority(
                rank=0,  # Will be set after sorting
                attributes={"node_id": node_id, "layer": str(layer)},
                current_count=count,
                target_count=target,
                reason=(
                    f"Node {node_id} has {count} recordings vs layer average "
                    f"of {mean_count:.0f} ({deficit_ratio*100:.0f}% below average)"
                ),
                priority_score=priority_score,
            ))

        # Sort by priority score descending
        priorities.sort(key=lambda p: p.priority_score, reverse=True)

        # Assign ranks
        for i, p in enumerate(priorities):
            p.rank = i + 1

        return priorities

    def generate_coverage_report(
        self,
        annotations: List[ScenarioAnnotation],
    ) -> str:
        """
        Generate a text summary of overall coverage.

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            All annotations to analyze.

        Returns
        -------
        str
            Human-readable coverage report.
        """
        if not annotations:
            return "No annotations available for coverage analysis."

        total = len(annotations)
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("SCENARIO COVERAGE REPORT")
        lines.append("=" * 60)
        lines.append(f"Total recordings analyzed: {total}")
        lines.append("")

        # Per-layer analysis
        for layer in range(1, 7):
            layer_names = {
                1: "Road Topology",
                2: "Traffic Infrastructure",
                3: "Temporary Modifications",
                4: "Dynamic Objects",
                5: "Environment",
                6: "Digital Information",
            }
            layer_coverage = self.compute_layer_coverage(annotations, layer)

            lines.append(f"--- Layer {layer}: {layer_names.get(layer, 'Unknown')} ---")

            if not layer_coverage:
                lines.append("  No tags found for this layer.")
            else:
                # Sort by count descending
                sorted_nodes = sorted(
                    layer_coverage.items(), key=lambda x: x[1], reverse=True
                )
                lines.append(f"  Unique nodes: {len(sorted_nodes)}")
                lines.append(f"  Total tags: {sum(layer_coverage.values())}")

                # Show top 5 and bottom 5
                lines.append("  Top represented:")
                for node_id, count in sorted_nodes[:5]:
                    pct = count / total * 100
                    lines.append(f"    {node_id}: {count} ({pct:.1f}%)")

                if len(sorted_nodes) > 5:
                    lines.append("  Least represented:")
                    for node_id, count in sorted_nodes[-5:]:
                        pct = count / total * 100
                        lines.append(f"    {node_id}: {count} ({pct:.1f}%)")

            lines.append("")

        # Coverage completeness estimate
        all_nodes: Set[str] = set()
        for ann in annotations:
            for tag in ann.tags:
                all_nodes.add(tag.node_id)

        lines.append("--- Overall Statistics ---")
        lines.append(f"  Unique scenario nodes observed: {len(all_nodes)}")
        lines.append(f"  Average tags per recording: "
                     f"{sum(len(a.tags) for a in annotations) / total:.1f}")

        # Identify completely missing layers
        covered_layers = {_extract_layer(nid) for nid in all_nodes}
        missing_layers = set(range(1, 7)) - covered_layers
        if missing_layers:
            lines.append(f"  WARNING: No coverage for layers: "
                         f"{sorted(missing_layers)}")

        # Collection priorities summary
        priorities = self.suggest_collection_priorities(annotations)
        if priorities:
            lines.append("")
            lines.append("--- Top Collection Priorities ---")
            for p in priorities[:10]:
                lines.append(f"  #{p.rank}: {p.reason}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
