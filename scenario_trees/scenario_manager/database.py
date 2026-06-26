"""
SQLite/SQLAlchemy backend for scenario metadata storage.

Stores recordings, scenario tags, split assignments, and model evaluation results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
    or_,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..taxonomy.scenario_schema import ScenarioTag


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all tables."""
    pass


class RecordingRow(Base):
    """Table storing recording metadata."""

    __tablename__ = "recordings"

    id = Column(String, primary_key=True)
    path = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    location = Column(String, nullable=True)


class ScenarioTagRow(Base):
    """Table storing scenario tags linked to recordings."""

    __tablename__ = "scenario_tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recording_id = Column(String, nullable=False, index=True)
    node_id = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    source = Column(String, nullable=False)
    frame_start = Column(Integer, nullable=True)
    frame_end = Column(Integer, nullable=True)


class SplitRow(Base):
    """Table storing train/val/test split assignments."""

    __tablename__ = "splits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recording_id = Column(String, nullable=False, index=True)
    split_name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False, index=True)


class ModelResultRow(Base):
    """Table storing model evaluation metrics per recording."""

    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recording_id = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False, index=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)


class ScenarioDatabase:
    """
    SQLAlchemy-backed scenario metadata database.

    Manages recordings, tags, splits, and model results using a local
    SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the database engine and create all tables.

        Args:
            db_path: Path to the SQLite database file. Use ':memory:' for in-memory DB.
        """
        if db_path == ":memory:":
            url = "sqlite:///:memory:"
        else:
            url = f"sqlite:///{db_path}"
        self.engine = create_engine(url, echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def _session(self) -> Session:
        """Create a new session."""
        return self._Session()

    def add_recording(
        self,
        recording_id: str,
        path: str,
        timestamp: Optional[datetime] = None,
        duration: Optional[float] = None,
        location: Optional[str] = None,
    ) -> None:
        """
        Add a recording to the database.

        Args:
            recording_id: Unique identifier for the recording.
            path: File path to the recording data.
            timestamp: When the recording was captured.
            duration: Duration in seconds.
            location: Geographic location descriptor.
        """
        with self._session() as session:
            row = RecordingRow(
                id=recording_id,
                path=path,
                timestamp=timestamp,
                duration=duration,
                location=location,
            )
            session.merge(row)
            session.commit()

    def add_tags(self, recording_id: str, tags: List[ScenarioTag]) -> None:
        """
        Add scenario tags for a recording.

        Args:
            recording_id: The recording these tags belong to.
            tags: List of ScenarioTag objects to store.
        """
        with self._session() as session:
            for tag in tags:
                row = ScenarioTagRow(
                    recording_id=recording_id,
                    node_id=tag.node_id,
                    confidence=tag.confidence,
                    source=tag.source,
                    frame_start=None,
                    frame_end=None,
                )
                session.add(row)
            session.commit()

    def add_tags_with_frames(
        self,
        recording_id: str,
        tags: List[ScenarioTag],
        frame_ranges: Optional[List[tuple]] = None,
    ) -> None:
        """
        Add scenario tags with optional frame range information.

        Args:
            recording_id: The recording these tags belong to.
            tags: List of ScenarioTag objects.
            frame_ranges: Optional list of (frame_start, frame_end) tuples aligned with tags.
        """
        with self._session() as session:
            for i, tag in enumerate(tags):
                frame_start = None
                frame_end = None
                if frame_ranges and i < len(frame_ranges):
                    frame_start, frame_end = frame_ranges[i]
                row = ScenarioTagRow(
                    recording_id=recording_id,
                    node_id=tag.node_id,
                    confidence=tag.confidence,
                    source=tag.source,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
                session.add(row)
            session.commit()

    def get_recording(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a recording's metadata by ID.

        Args:
            recording_id: The recording identifier.

        Returns:
            Dictionary with recording fields, or None if not found.
        """
        with self._session() as session:
            row = session.get(RecordingRow, recording_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "path": row.path,
                "timestamp": row.timestamp,
                "duration": row.duration,
                "location": row.location,
            }

    def get_tags_for_recording(self, recording_id: str) -> List[Dict[str, Any]]:
        """
        Get all scenario tags for a given recording.

        Args:
            recording_id: The recording identifier.

        Returns:
            List of dictionaries with tag fields.
        """
        with self._session() as session:
            rows = (
                session.query(ScenarioTagRow)
                .filter(ScenarioTagRow.recording_id == recording_id)
                .all()
            )
            return [
                {
                    "id": row.id,
                    "recording_id": row.recording_id,
                    "node_id": row.node_id,
                    "confidence": row.confidence,
                    "source": row.source,
                    "frame_start": row.frame_start,
                    "frame_end": row.frame_end,
                }
                for row in rows
            ]

    def get_recordings_with_tag(
        self, node_id: str, min_confidence: float = 0.0
    ) -> List[str]:
        """
        Find all recording IDs that have a specific tag at or above a confidence threshold.

        Args:
            node_id: The scenario tree node ID to search for.
            min_confidence: Minimum confidence score (inclusive).

        Returns:
            List of recording IDs matching the criteria.
        """
        with self._session() as session:
            rows = (
                session.query(ScenarioTagRow.recording_id)
                .filter(
                    ScenarioTagRow.node_id == node_id,
                    ScenarioTagRow.confidence >= min_confidence,
                )
                .distinct()
                .all()
            )
            return [row[0] for row in rows]

    def get_all_recordings(self) -> List[Dict[str, Any]]:
        """
        Get all recordings in the database.

        Returns:
            List of dictionaries with recording metadata.
        """
        with self._session() as session:
            rows = session.query(RecordingRow).all()
            return [
                {
                    "id": row.id,
                    "path": row.path,
                    "timestamp": row.timestamp,
                    "duration": row.duration,
                    "location": row.location,
                }
                for row in rows
            ]

    def add_split_assignment(
        self, recording_id: str, split_name: str, version: str
    ) -> None:
        """
        Assign a recording to a data split.

        Args:
            recording_id: The recording identifier.
            split_name: Name of the split (e.g., 'train', 'val', 'test').
            version: Version identifier for this split scheme.
        """
        with self._session() as session:
            row = SplitRow(
                recording_id=recording_id,
                split_name=split_name,
                version=version,
            )
            session.add(row)
            session.commit()

    def add_model_result(
        self,
        recording_id: str,
        model_name: str,
        metric_name: str,
        value: float,
    ) -> None:
        """
        Store a model evaluation result for a recording.

        Args:
            recording_id: The recording identifier.
            model_name: Name of the evaluated model.
            metric_name: Name of the metric (e.g., 'mAP', 'recall').
            value: The metric value.
        """
        with self._session() as session:
            row = ModelResultRow(
                recording_id=recording_id,
                model_name=model_name,
                metric_name=metric_name,
                metric_value=value,
            )
            session.add(row)
            session.commit()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics about the dataset.

        Returns:
            Dictionary with counts per tag, per split, per source, total recordings,
            total duration, and location breakdown.
        """
        with self._session() as session:
            total_recordings = session.query(func.count(RecordingRow.id)).scalar() or 0
            total_duration = (
                session.query(func.sum(RecordingRow.duration)).scalar() or 0.0
            )

            # Counts per tag (node_id)
            tag_counts_rows = (
                session.query(
                    ScenarioTagRow.node_id,
                    func.count(ScenarioTagRow.id),
                )
                .group_by(ScenarioTagRow.node_id)
                .all()
            )
            tag_counts: Dict[str, int] = {row[0]: row[1] for row in tag_counts_rows}

            # Counts per split (across all versions)
            split_counts_rows = (
                session.query(
                    SplitRow.split_name,
                    SplitRow.version,
                    func.count(SplitRow.id),
                )
                .group_by(SplitRow.split_name, SplitRow.version)
                .all()
            )
            split_counts: Dict[str, Dict[str, int]] = {}
            for split_name, version, count in split_counts_rows:
                if version not in split_counts:
                    split_counts[version] = {}
                split_counts[version][split_name] = count

            # Counts per source
            source_counts_rows = (
                session.query(
                    ScenarioTagRow.source,
                    func.count(ScenarioTagRow.id),
                )
                .group_by(ScenarioTagRow.source)
                .all()
            )
            source_counts: Dict[str, int] = {row[0]: row[1] for row in source_counts_rows}

            # Location breakdown
            location_counts_rows = (
                session.query(
                    RecordingRow.location,
                    func.count(RecordingRow.id),
                )
                .filter(RecordingRow.location.isnot(None))
                .group_by(RecordingRow.location)
                .all()
            )
            location_counts: Dict[str, int] = {
                row[0]: row[1] for row in location_counts_rows
            }

            return {
                "total_recordings": total_recordings,
                "total_duration_seconds": total_duration,
                "tag_counts": tag_counts,
                "split_counts": split_counts,
                "source_counts": source_counts,
                "location_counts": location_counts,
            }

    def search(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Full-text search on recording paths and locations.

        Uses SQL LIKE for substring matching on path and location fields.

        Args:
            query_text: Search string to match against paths and locations.

        Returns:
            List of matching recording dictionaries.
        """
        pattern = f"%{query_text}%"
        with self._session() as session:
            rows = (
                session.query(RecordingRow)
                .filter(
                    or_(
                        RecordingRow.path.like(pattern),
                        RecordingRow.location.like(pattern),
                    )
                )
                .all()
            )
            return [
                {
                    "id": row.id,
                    "path": row.path,
                    "timestamp": row.timestamp,
                    "duration": row.duration,
                    "location": row.location,
                }
                for row in rows
            ]

    def get_split_recordings(
        self, split_name: str, version: str
    ) -> List[str]:
        """
        Get all recording IDs assigned to a specific split and version.

        Args:
            split_name: Name of the split (train/val/test).
            version: Version identifier.

        Returns:
            List of recording IDs in that split.
        """
        with self._session() as session:
            rows = (
                session.query(SplitRow.recording_id)
                .filter(
                    SplitRow.split_name == split_name,
                    SplitRow.version == version,
                )
                .all()
            )
            return [row[0] for row in rows]

    def get_all_node_ids(self) -> List[str]:
        """
        Get all distinct node IDs that appear in the scenario_tags table.

        Returns:
            Sorted list of unique node IDs.
        """
        with self._session() as session:
            rows = (
                session.query(ScenarioTagRow.node_id)
                .distinct()
                .order_by(ScenarioTagRow.node_id)
                .all()
            )
            return [row[0] for row in rows]

    def get_model_results(
        self, recording_id: Optional[str] = None, model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query model results with optional filters.

        Args:
            recording_id: Filter by recording (optional).
            model_name: Filter by model name (optional).

        Returns:
            List of model result dictionaries.
        """
        with self._session() as session:
            query = session.query(ModelResultRow)
            if recording_id is not None:
                query = query.filter(ModelResultRow.recording_id == recording_id)
            if model_name is not None:
                query = query.filter(ModelResultRow.model_name == model_name)
            rows = query.all()
            return [
                {
                    "id": row.id,
                    "recording_id": row.recording_id,
                    "model_name": row.model_name,
                    "metric_name": row.metric_name,
                    "metric_value": row.metric_value,
                }
                for row in rows
            ]
