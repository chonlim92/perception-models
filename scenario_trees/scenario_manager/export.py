"""
Export filtered subsets of scenario-tagged recordings for model training.

Provides methods to export file lists, filtered subsets, mini datasets,
and metadata CSV files.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from ..taxonomy.scenario_schema import ScenarioQuery
from .database import ScenarioDatabase
from .query_engine import ScenarioQueryEngine


class ScenarioExporter:
    """
    Exporter for scenario-tagged recording datasets.

    Provides methods to export filtered views of the dataset in various
    formats suitable for model training pipelines.
    """

    def __init__(self, database: ScenarioDatabase) -> None:
        """
        Initialize the exporter.

        Args:
            database: The ScenarioDatabase instance to export from.
        """
        self.database = database
        self._query_engine = ScenarioQueryEngine(database)

    def export_file_list(
        self,
        recording_ids: List[str],
        output_path: str,
        format: str = "txt",
    ) -> None:
        """
        Write a list of file paths for the given recordings.

        Args:
            recording_ids: List of recording IDs to include.
            output_path: Path to the output file.
            format: Output format - 'txt' (one path per line), 'json' (JSON array),
                    or 'jsonl' (one JSON object per line).
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Resolve paths from database
        entries: List[Dict[str, str]] = []
        for rec_id in recording_ids:
            rec = self.database.get_recording(rec_id)
            if rec is not None:
                entries.append({"id": rec["id"], "path": rec["path"]})

        if format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(entry["path"] + "\n")

        elif format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [entry["path"] for entry in entries],
                    f,
                    indent=2,
                )

        elif format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

        else:
            raise ValueError(
                f"Unsupported format: '{format}'. Supported: txt, json, jsonl"
            )

    def export_filtered_subset(
        self, query: ScenarioQuery, output_dir: str
    ) -> None:
        """
        Export recordings matching a ScenarioQuery to an output directory.

        Creates the output directory structure with:
        - file_list.txt: paths of matching recordings
        - metadata.json: query parameters and recording metadata
        - tags/: per-recording tag JSON files

        Args:
            query: ScenarioQuery specifying which recordings to export.
            output_dir: Directory to write the exported subset to.
        """
        # Find matching recordings
        matching_ids = self._query_engine.query(query)

        os.makedirs(output_dir, exist_ok=True)
        tags_dir = os.path.join(output_dir, "tags")
        os.makedirs(tags_dir, exist_ok=True)

        # Write file list
        file_list_path = os.path.join(output_dir, "file_list.txt")
        self.export_file_list(matching_ids, file_list_path, format="txt")

        # Write metadata
        metadata = {
            "query": {
                "required_tags": query.required_tags,
                "excluded_tags": query.excluded_tags,
                "min_confidence": query.min_confidence,
                "layer_filters": query.layer_filters,
            },
            "num_recordings": len(matching_ids),
            "recordings": [],
        }

        for rec_id in matching_ids:
            rec = self.database.get_recording(rec_id)
            if rec is None:
                continue

            # Serialize datetime for JSON
            rec_data = {
                "id": rec["id"],
                "path": rec["path"],
                "timestamp": rec["timestamp"].isoformat() if rec["timestamp"] else None,
                "duration": rec["duration"],
                "location": rec["location"],
            }
            metadata["recordings"].append(rec_data)

            # Write per-recording tag file
            tags = self.database.get_tags_for_recording(rec_id)
            tag_file = os.path.join(tags_dir, f"{rec_id}.json")
            with open(tag_file, "w", encoding="utf-8") as f:
                json.dump(tags, f, indent=2)

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def export_mini_dataset(
        self,
        scenario_filter: Dict[str, Any],
        max_recordings: int,
        output_dir: str,
    ) -> None:
        """
        Create a small debugging dataset by sampling from filtered recordings.

        The scenario_filter dict supports:
        - 'required_tags': list of node_ids that must be present
        - 'excluded_tags': list of node_ids that must not be present
        - 'min_confidence': minimum confidence threshold
        - 'locations': list of locations to include
        - 'max_duration': maximum recording duration in seconds

        Args:
            scenario_filter: Dictionary specifying filter criteria.
            max_recordings: Maximum number of recordings to include.
            output_dir: Directory to write the mini dataset to.
        """
        # Build a query from the filter dict
        required_tags = scenario_filter.get("required_tags", [])
        excluded_tags = scenario_filter.get("excluded_tags", [])
        min_confidence = scenario_filter.get("min_confidence", 0.0)
        locations = scenario_filter.get("locations", [])
        max_duration = scenario_filter.get("max_duration", None)

        query = ScenarioQuery(
            required_tags=required_tags,
            excluded_tags=excluded_tags,
            min_confidence=min_confidence,
        )

        matching_ids = self._query_engine.query(query)

        # Apply additional filters not in ScenarioQuery
        if locations or max_duration is not None:
            filtered_ids: List[str] = []
            for rec_id in matching_ids:
                rec = self.database.get_recording(rec_id)
                if rec is None:
                    continue
                # Location filter
                if locations:
                    if rec["location"] is None or rec["location"].lower() not in [
                        loc.lower() for loc in locations
                    ]:
                        continue
                # Duration filter
                if max_duration is not None:
                    if rec["duration"] is not None and rec["duration"] > max_duration:
                        continue
                filtered_ids.append(rec_id)
            matching_ids = filtered_ids

        # Limit to max_recordings
        selected_ids = matching_ids[:max_recordings]

        # Export
        os.makedirs(output_dir, exist_ok=True)

        # Write file list
        file_list_path = os.path.join(output_dir, "file_list.txt")
        self.export_file_list(selected_ids, file_list_path, format="txt")

        # Write summary
        summary = {
            "filter": scenario_filter,
            "max_recordings": max_recordings,
            "actual_recordings": len(selected_ids),
            "recording_ids": selected_ids,
        }
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Write metadata CSV
        csv_path = os.path.join(output_dir, "metadata.csv")
        self.export_metadata_csv(selected_ids, csv_path)

    def export_split(
        self, split_name: str, version: str, output_path: str
    ) -> None:
        """
        Export all recordings from a specific split to a file list.

        Args:
            split_name: Name of the split (e.g., 'train', 'val', 'test').
            version: Split version identifier.
            output_path: Path to write the file list to.
        """
        rec_ids = self.database.get_split_recordings(split_name, version)
        self.export_file_list(rec_ids, output_path, format="txt")

    def export_metadata_csv(
        self, recording_ids: List[str], output_path: str
    ) -> None:
        """
        Export recording metadata and tags as a CSV file.

        Each row represents one recording with columns for metadata fields
        and a semicolon-separated list of tag node_ids.

        Args:
            recording_ids: List of recording IDs to export.
            output_path: Path to the output CSV file.
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fieldnames = [
            "recording_id",
            "path",
            "timestamp",
            "duration",
            "location",
            "tags",
            "tag_confidences",
            "tag_sources",
            "num_tags",
        ]

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for rec_id in recording_ids:
                rec = self.database.get_recording(rec_id)
                if rec is None:
                    continue

                tags = self.database.get_tags_for_recording(rec_id)

                tag_node_ids = [t["node_id"] for t in tags]
                tag_confidences = [f"{t['confidence']:.3f}" for t in tags]
                tag_sources = [t["source"] for t in tags]

                row = {
                    "recording_id": rec["id"],
                    "path": rec["path"],
                    "timestamp": (
                        rec["timestamp"].isoformat() if rec["timestamp"] else ""
                    ),
                    "duration": rec["duration"] if rec["duration"] is not None else "",
                    "location": rec["location"] or "",
                    "tags": ";".join(tag_node_ids),
                    "tag_confidences": ";".join(tag_confidences),
                    "tag_sources": ";".join(tag_sources),
                    "num_tags": len(tags),
                }
                writer.writerow(row)
