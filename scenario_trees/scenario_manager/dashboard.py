"""
Text-based dashboard for visualizing dataset statistics.

Renders tables, coverage matrices, split compositions, and difficulty
distributions using plain text formatting (no external TUI libraries).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .database import ScenarioDatabase


class ScenarioDashboard:
    """
    Text-based dashboard for scenario dataset statistics.

    All render methods return formatted strings suitable for terminal display.
    """

    def __init__(self, database: ScenarioDatabase) -> None:
        """
        Initialize the dashboard.

        Args:
            database: The ScenarioDatabase instance to visualize.
        """
        self.database = database

    def render_statistics(self) -> str:
        """
        Render a text table with overall dataset statistics.

        Shows total recordings, total duration, counts per layer, tag source
        breakdown, and location distribution.

        Returns:
            Formatted text table string.
        """
        stats = self.database.get_statistics()
        total_recordings = stats["total_recordings"]
        total_duration = stats["total_duration_seconds"]
        tag_counts = stats["tag_counts"]
        source_counts = stats["source_counts"]
        location_counts = stats["location_counts"]

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DATASET STATISTICS")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append(f"  Total Recordings:  {total_recordings}")
        hours = total_duration / 3600.0
        lines.append(f"  Total Duration:    {total_duration:.1f}s ({hours:.2f}h)")
        lines.append(f"  Total Tag Types:   {len(tag_counts)}")
        lines.append("")

        # Per-layer tag counts
        lines.append("-" * 60)
        lines.append("  TAGS PER LAYER")
        lines.append("-" * 60)

        layer_aggregates: Dict[int, int] = defaultdict(int)
        for node_id, count in tag_counts.items():
            layer = _extract_layer(node_id)
            layer_aggregates[layer] += count

        layer_names = {
            1: "Road Topology",
            2: "Traffic Infrastructure",
            3: "Temporary Modifications",
            4: "Dynamic Objects",
            5: "Environment",
            6: "Digital Information",
        }

        for layer_num in sorted(layer_aggregates.keys()):
            name = layer_names.get(layer_num, f"Layer {layer_num}")
            count = layer_aggregates[layer_num]
            bar = _bar_char(count, max(layer_aggregates.values(), default=1))
            lines.append(f"  L{layer_num} {name:<25} {count:>6}  {bar}")

        lines.append("")

        # Source breakdown
        lines.append("-" * 60)
        lines.append("  TAG SOURCES")
        lines.append("-" * 60)
        for source, count in sorted(source_counts.items()):
            lines.append(f"  {source:<10} {count:>8}")
        lines.append("")

        # Location distribution
        if location_counts:
            lines.append("-" * 60)
            lines.append("  LOCATIONS")
            lines.append("-" * 60)
            for loc, count in sorted(
                location_counts.items(), key=lambda x: -x[1]
            ):
                bar = _bar_char(count, max(location_counts.values()))
                lines.append(f"  {loc:<20} {count:>6}  {bar}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def render_coverage_matrix(
        self, layer1_attr: str, layer2_attr: str
    ) -> str:
        """
        Render a text-based coverage matrix showing co-occurrence between
        two layers/attributes.

        Shows how many recordings have tags in both the row and column categories.

        Args:
            layer1_attr: First attribute for rows (e.g., 'L1' for Road Topology nodes).
            layer2_attr: Second attribute for columns (e.g., 'L5' for Environment nodes).

        Returns:
            Formatted text matrix string.
        """
        all_recordings = self.database.get_all_recordings()

        # Collect which recordings have which node_ids per layer prefix
        recording_tags_map: Dict[str, Dict[str, set]] = {}
        for rec in all_recordings:
            rec_id = rec["id"]
            tags = self.database.get_tags_for_recording(rec_id)
            row_tags: set = set()
            col_tags: set = set()
            for tag in tags:
                node_id = tag["node_id"]
                if node_id.startswith(layer1_attr):
                    row_tags.add(node_id)
                if node_id.startswith(layer2_attr):
                    col_tags.add(node_id)
            recording_tags_map[rec_id] = {"rows": row_tags, "cols": col_tags}

        # Get unique row and column values
        all_row_values: set = set()
        all_col_values: set = set()
        for data in recording_tags_map.values():
            all_row_values.update(data["rows"])
            all_col_values.update(data["cols"])

        row_values = sorted(all_row_values)
        col_values = sorted(all_col_values)

        if not row_values or not col_values:
            return f"No coverage data for {layer1_attr} x {layer2_attr}"

        # Build co-occurrence matrix
        matrix: Dict[str, Dict[str, int]] = {
            r: {c: 0 for c in col_values} for r in row_values
        }

        for data in recording_tags_map.values():
            for r in data["rows"]:
                for c in data["cols"]:
                    matrix[r][c] += 1

        # Render the matrix
        lines: List[str] = []
        lines.append(f"Coverage Matrix: {layer1_attr} (rows) x {layer2_attr} (cols)")
        lines.append("=" * (16 + len(col_values) * 8))

        # Header
        header = f"{'':>14} |"
        for col in col_values:
            # Use short label (last part of node_id)
            short = col.split(".")[-1] if "." in col else col
            header += f" {short:>5} |"
        lines.append(header)
        lines.append("-" * len(header))

        # Data rows
        for row in row_values:
            short_row = row.split(".")[-1] if "." in row else row
            row_str = f"  {row:<12}|"
            for col in col_values:
                count = matrix[row][col]
                if count > 0:
                    row_str += f" {count:>5} |"
                else:
                    row_str += f"     . |"
            lines.append(row_str)

        lines.append("-" * len(header))
        return "\n".join(lines)

    def render_split_composition(self, version: str) -> str:
        """
        Render the composition of a specific split version.

        Shows the number of recordings per split, proportion, and per-split
        tag distribution summary.

        Args:
            version: The split version identifier to display.

        Returns:
            Formatted text showing split breakdown.
        """
        stats = self.database.get_statistics()
        split_counts = stats.get("split_counts", {})
        total_recordings = stats["total_recordings"]

        version_splits = split_counts.get(version, {})
        if not version_splits:
            return f"No split data found for version '{version}'"

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(f"  SPLIT COMPOSITION (version: {version})")
        lines.append("=" * 60)
        lines.append("")

        total_in_splits = sum(version_splits.values())
        lines.append(f"  Total recordings in splits: {total_in_splits} / {total_recordings}")
        lines.append("")

        lines.append(f"  {'Split':<12} {'Count':>8} {'Ratio':>8} {'Bar'}")
        lines.append(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*20}")

        max_count = max(version_splits.values(), default=1)
        for split_name in sorted(version_splits.keys()):
            count = version_splits[split_name]
            ratio = count / max(total_in_splits, 1)
            bar = _bar_char(count, max_count, width=20)
            lines.append(f"  {split_name:<12} {count:>8} {ratio:>7.1%} {bar}")

        # Per-split tag composition
        lines.append("")
        lines.append("-" * 60)
        lines.append("  PER-SPLIT TAG SUMMARY")
        lines.append("-" * 60)

        for split_name in sorted(version_splits.keys()):
            rec_ids = self.database.get_split_recordings(split_name, version)
            tag_counts: Dict[str, int] = defaultdict(int)
            for rec_id in rec_ids:
                tags = self.database.get_tags_for_recording(rec_id)
                for tag in tags:
                    tag_counts[tag["node_id"]] += 1

            lines.append(f"\n  [{split_name}] ({len(rec_ids)} recordings, "
                        f"{len(tag_counts)} unique tags)")
            # Show top 5 tags
            top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]
            for node_id, count in top_tags:
                lines.append(f"    {node_id:<15} {count:>4}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def render_difficulty_distribution(self) -> str:
        """
        Render a histogram of recording difficulty scores.

        Difficulty is approximated by the number of distinct scenario tags
        per recording (more tags = more complex scenario = higher difficulty).

        Returns:
            Text-based histogram string.
        """
        all_recordings = self.database.get_all_recordings()
        if not all_recordings:
            return "No recordings in database."

        # Calculate difficulty score per recording
        difficulty_scores: List[int] = []
        for rec in all_recordings:
            tags = self.database.get_tags_for_recording(rec["id"])
            unique_tags = {tag["node_id"] for tag in tags}
            difficulty_scores.append(len(unique_tags))

        if not difficulty_scores:
            return "No tag data available for difficulty estimation."

        # Create histogram buckets
        max_score = max(difficulty_scores)
        min_score = min(difficulty_scores)

        # Use fixed bucket ranges
        if max_score == min_score:
            buckets = {str(min_score): len(difficulty_scores)}
        else:
            num_buckets = min(10, max_score - min_score + 1)
            bucket_size = max(1, (max_score - min_score + 1) // num_buckets)
            buckets: Dict[str, int] = {}
            for score in difficulty_scores:
                bucket_idx = (score - min_score) // bucket_size
                bucket_start = min_score + bucket_idx * bucket_size
                bucket_end = bucket_start + bucket_size - 1
                label = f"{bucket_start}-{bucket_end}" if bucket_start != bucket_end else str(bucket_start)
                buckets[label] = buckets.get(label, 0) + 1

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DIFFICULTY DISTRIBUTION (tags per recording)")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Recordings: {len(difficulty_scores)}")
        lines.append(f"  Min tags:   {min_score}")
        lines.append(f"  Max tags:   {max_score}")
        avg_score = sum(difficulty_scores) / len(difficulty_scores)
        lines.append(f"  Avg tags:   {avg_score:.1f}")
        lines.append("")

        max_count = max(buckets.values(), default=1)
        lines.append(f"  {'Range':<10} {'Count':>6}  Histogram")
        lines.append(f"  {'-'*10} {'-'*6}  {'-'*30}")

        for label, count in sorted(buckets.items(), key=lambda x: x[0]):
            bar_len = int((count / max_count) * 30)
            bar = "#" * bar_len
            lines.append(f"  {label:<10} {count:>6}  {bar}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def render_summary(self) -> str:
        """
        Render a full dashboard combining all visualizations.

        Returns:
            Complete dashboard string with statistics, difficulty distribution,
            and available split compositions.
        """
        sections: List[str] = []

        # Main statistics
        sections.append(self.render_statistics())
        sections.append("")

        # Difficulty distribution
        sections.append(self.render_difficulty_distribution())
        sections.append("")

        # Show all available split versions
        stats = self.database.get_statistics()
        split_counts = stats.get("split_counts", {})
        for version in sorted(split_counts.keys()):
            sections.append(self.render_split_composition(version))
            sections.append("")

        return "\n".join(sections)


def _extract_layer(node_id: str) -> int:
    """Extract layer number from a node ID like 'L4.3.1' -> 4."""
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


def _bar_char(value: int, max_value: int, width: int = 20) -> str:
    """Create a simple text bar of proportional length."""
    if max_value <= 0:
        return ""
    bar_len = int((value / max_value) * width)
    return "#" * max(bar_len, 1) if value > 0 else ""
