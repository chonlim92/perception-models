# [IMPLEMENTED BY CLAUDE - was missing]
"""Extended database schema for FST versioning, KPIs, scripts, and suggestions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Boolean,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class TreeVersionRow(Base):
    __tablename__ = "tree_versions"

    id = Column(String, primary_key=True)
    version = Column(String, nullable=False, index=True)
    tree_data = Column(Text, nullable=False)  # JSON
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String, nullable=False, default="system")
    change_description = Column(Text, default="")
    parent_version_id = Column(String, nullable=True)
    is_current = Column(Boolean, default=False)


class NodeKPIRow(Base):
    __tablename__ = "node_kpis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String, nullable=False, index=True)
    metric_name = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)
    direction = Column(String, nullable=False, default="above")
    warning_margin = Column(Float, default=0.1)


class EvaluationScriptRow(Base):
    __tablename__ = "evaluation_scripts"

    id = Column(String, primary_key=True)
    node_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    script_content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    last_run = Column(DateTime, nullable=True)


class EvaluationResultRow(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String, nullable=False, index=True)
    recording_id = Column(String, nullable=False, index=True)
    script_id = Column(String, nullable=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    evaluated_at = Column(DateTime, nullable=False)


class NodeRecordingRow(Base):
    __tablename__ = "node_recordings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String, nullable=False, index=True)
    recording_id = Column(String, nullable=False, index=True)
    attached_at = Column(DateTime, nullable=False)
    attached_by = Column(String, default="system")


class SuggestionRow(Base):
    __tablename__ = "suggestions"

    id = Column(String, primary_key=True)
    node_id = Column(String, nullable=False, index=True)
    suggestion_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    evidence = Column(Text, default="{}")  # JSON
    proposed_changes = Column(Text, default="{}")  # JSON
    confidence = Column(Float, default=0.0)
    impact_estimate = Column(String, default="medium")
    status = Column(String, default="pending", index=True)
    created_at = Column(DateTime, nullable=False)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    notes = Column(Text, default="")


class RecordingMetadataRow(Base):
    __tablename__ = "recording_metadata"

    id = Column(String, primary_key=True)
    path = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    location = Column(String, nullable=True)
    attributes = Column(Text, default="{}")  # JSON - for pattern analysis


class FSTDatabase:
    """Database manager for the FST API with versioning and analysis support."""

    def __init__(self, db_path: str = "fst.db") -> None:
        if db_path == ":memory:":
            url = "sqlite:///:memory:"
        else:
            url = f"sqlite:///{db_path}"
        self.engine = create_engine(url, echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def _session(self) -> Session:
        return self._Session()

    # --- Tree Versions ---

    def save_tree_version(
        self,
        tree_data: Dict[str, Any],
        version: str,
        created_by: str = "system",
        change_description: str = "",
        parent_version_id: Optional[str] = None,
    ) -> str:
        version_id = str(uuid.uuid4())
        with self._session() as session:
            session.query(TreeVersionRow).update({"is_current": False})
            row = TreeVersionRow(
                id=version_id,
                version=version,
                tree_data=json.dumps(tree_data),
                created_at=datetime.utcnow(),
                created_by=created_by,
                change_description=change_description,
                parent_version_id=parent_version_id,
                is_current=True,
            )
            session.add(row)
            session.commit()
        return version_id

    def get_current_tree_version(self) -> Optional[Dict[str, Any]]:
        with self._session() as session:
            row = session.query(TreeVersionRow).filter(
                TreeVersionRow.is_current == True
            ).first()
            if row is None:
                return None
            return {
                "id": row.id,
                "version": row.version,
                "tree_data": json.loads(row.tree_data),
                "created_at": row.created_at,
                "created_by": row.created_by,
                "change_description": row.change_description,
                "parent_version_id": row.parent_version_id,
            }

    def get_tree_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        with self._session() as session:
            row = session.get(TreeVersionRow, version_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "version": row.version,
                "tree_data": json.loads(row.tree_data),
                "created_at": row.created_at,
                "created_by": row.created_by,
                "change_description": row.change_description,
                "parent_version_id": row.parent_version_id,
            }

    def list_tree_versions(self) -> List[Dict[str, Any]]:
        with self._session() as session:
            rows = session.query(TreeVersionRow).order_by(
                TreeVersionRow.created_at.desc()
            ).all()
            return [
                {
                    "id": row.id,
                    "version": row.version,
                    "created_at": row.created_at,
                    "created_by": row.created_by,
                    "change_description": row.change_description,
                    "parent_version_id": row.parent_version_id,
                    "is_current": row.is_current,
                }
                for row in rows
            ]

    # --- Node KPIs ---

    def set_kpi(
        self,
        node_id: str,
        metric_name: str,
        threshold: float,
        direction: str = "above",
        warning_margin: float = 0.1,
    ) -> None:
        with self._session() as session:
            existing = session.query(NodeKPIRow).filter(
                NodeKPIRow.node_id == node_id,
                NodeKPIRow.metric_name == metric_name,
            ).first()
            if existing:
                existing.threshold = threshold
                existing.direction = direction
                existing.warning_margin = warning_margin
            else:
                row = NodeKPIRow(
                    node_id=node_id,
                    metric_name=metric_name,
                    threshold=threshold,
                    direction=direction,
                    warning_margin=warning_margin,
                )
                session.add(row)
            session.commit()

    def get_kpis(self, node_id: str) -> List[Dict[str, Any]]:
        with self._session() as session:
            rows = session.query(NodeKPIRow).filter(
                NodeKPIRow.node_id == node_id
            ).all()
            return [
                {
                    "metric_name": row.metric_name,
                    "threshold": row.threshold,
                    "direction": row.direction,
                    "warning_margin": row.warning_margin,
                }
                for row in rows
            ]

    # --- Evaluation Scripts ---

    def add_script(
        self, node_id: str, name: str, script_content: str
    ) -> str:
        script_id = str(uuid.uuid4())
        with self._session() as session:
            row = EvaluationScriptRow(
                id=script_id,
                node_id=node_id,
                name=name,
                script_content=script_content,
                created_at=datetime.utcnow(),
            )
            session.add(row)
            session.commit()
        return script_id

    def get_scripts(self, node_id: str) -> List[Dict[str, Any]]:
        with self._session() as session:
            rows = session.query(EvaluationScriptRow).filter(
                EvaluationScriptRow.node_id == node_id
            ).all()
            return [
                {
                    "id": row.id,
                    "node_id": row.node_id,
                    "name": row.name,
                    "script_content": row.script_content,
                    "created_at": row.created_at,
                    "last_run": row.last_run,
                }
                for row in rows
            ]

    # --- Evaluation Results ---

    def add_evaluation_result(
        self,
        node_id: str,
        recording_id: str,
        metric_name: str,
        metric_value: float,
        script_id: Optional[str] = None,
    ) -> None:
        with self._session() as session:
            row = EvaluationResultRow(
                node_id=node_id,
                recording_id=recording_id,
                script_id=script_id,
                metric_name=metric_name,
                metric_value=metric_value,
                evaluated_at=datetime.utcnow(),
            )
            session.add(row)
            session.commit()

    def get_evaluation_results(
        self,
        node_id: str,
        metric_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._session() as session:
            query = session.query(EvaluationResultRow).filter(
                EvaluationResultRow.node_id == node_id
            )
            if metric_name:
                query = query.filter(
                    EvaluationResultRow.metric_name == metric_name
                )
            rows = query.order_by(EvaluationResultRow.evaluated_at.desc()).all()
            return [
                {
                    "recording_id": row.recording_id,
                    "metric_name": row.metric_name,
                    "metric_value": row.metric_value,
                    "script_id": row.script_id,
                    "evaluated_at": row.evaluated_at,
                }
                for row in rows
            ]

    def get_node_metrics_summary(self, node_id: str) -> Dict[str, Any]:
        """Compute aggregated metrics for a node across all its recordings."""
        with self._session() as session:
            results = session.query(EvaluationResultRow).filter(
                EvaluationResultRow.node_id == node_id
            ).all()

            if not results:
                return {"metrics": {}, "total_recordings": 0}

            from collections import defaultdict
            import statistics

            metrics_by_name: Dict[str, List[float]] = defaultdict(list)
            recording_ids = set()
            for r in results:
                metrics_by_name[r.metric_name].append(r.metric_value)
                recording_ids.add(r.recording_id)

            summary = {}
            for name, values in metrics_by_name.items():
                summary[name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

            return {
                "metrics": summary,
                "total_recordings": len(recording_ids),
            }

    # --- Node Recordings ---

    def attach_recording(
        self, node_id: str, recording_id: str, attached_by: str = "system"
    ) -> None:
        with self._session() as session:
            existing = session.query(NodeRecordingRow).filter(
                NodeRecordingRow.node_id == node_id,
                NodeRecordingRow.recording_id == recording_id,
            ).first()
            if not existing:
                row = NodeRecordingRow(
                    node_id=node_id,
                    recording_id=recording_id,
                    attached_at=datetime.utcnow(),
                    attached_by=attached_by,
                )
                session.add(row)
                session.commit()

    def detach_recording(self, node_id: str, recording_id: str) -> None:
        with self._session() as session:
            session.query(NodeRecordingRow).filter(
                NodeRecordingRow.node_id == node_id,
                NodeRecordingRow.recording_id == recording_id,
            ).delete()
            session.commit()

    def get_node_recordings(self, node_id: str) -> List[str]:
        with self._session() as session:
            rows = session.query(NodeRecordingRow.recording_id).filter(
                NodeRecordingRow.node_id == node_id
            ).all()
            return [row[0] for row in rows]

    # --- Recording Metadata ---

    def upsert_recording(
        self,
        recording_id: str,
        path: str,
        timestamp: Optional[datetime] = None,
        duration: Optional[float] = None,
        location: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._session() as session:
            row = session.get(RecordingMetadataRow, recording_id)
            if row:
                row.path = path
                if timestamp:
                    row.timestamp = timestamp
                if duration:
                    row.duration = duration
                if location:
                    row.location = location
                if attributes:
                    row.attributes = json.dumps(attributes)
            else:
                row = RecordingMetadataRow(
                    id=recording_id,
                    path=path,
                    timestamp=timestamp,
                    duration=duration,
                    location=location,
                    attributes=json.dumps(attributes or {}),
                )
                session.add(row)
            session.commit()

    def get_recording(self, recording_id: str) -> Optional[Dict[str, Any]]:
        with self._session() as session:
            row = session.get(RecordingMetadataRow, recording_id)
            if not row:
                return None
            return {
                "id": row.id,
                "path": row.path,
                "timestamp": row.timestamp,
                "duration": row.duration,
                "location": row.location,
                "attributes": json.loads(row.attributes) if row.attributes else {},
            }

    def get_recordings_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        with self._session() as session:
            rec_ids = session.query(NodeRecordingRow.recording_id).filter(
                NodeRecordingRow.node_id == node_id
            ).all()
            rec_ids = [r[0] for r in rec_ids]

            if not rec_ids:
                return []

            rows = session.query(RecordingMetadataRow).filter(
                RecordingMetadataRow.id.in_(rec_ids)
            ).all()
            return [
                {
                    "id": row.id,
                    "path": row.path,
                    "timestamp": row.timestamp,
                    "duration": row.duration,
                    "location": row.location,
                    "attributes": json.loads(row.attributes) if row.attributes else {},
                }
                for row in rows
            ]

    # --- Suggestions ---

    def add_suggestion(
        self,
        node_id: str,
        suggestion_type: str,
        title: str,
        description: str,
        evidence: Dict[str, Any],
        proposed_changes: Dict[str, Any],
        confidence: float,
        impact_estimate: str = "medium",
    ) -> str:
        suggestion_id = str(uuid.uuid4())
        with self._session() as session:
            row = SuggestionRow(
                id=suggestion_id,
                node_id=node_id,
                suggestion_type=suggestion_type,
                title=title,
                description=description,
                evidence=json.dumps(evidence),
                proposed_changes=json.dumps(proposed_changes),
                confidence=confidence,
                impact_estimate=impact_estimate,
                status="pending",
                created_at=datetime.utcnow(),
            )
            session.add(row)
            session.commit()
        return suggestion_id

    def get_suggestions(
        self, node_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._session() as session:
            query = session.query(SuggestionRow)
            if node_id:
                query = query.filter(SuggestionRow.node_id == node_id)
            if status:
                query = query.filter(SuggestionRow.status == status)
            rows = query.order_by(SuggestionRow.created_at.desc()).all()
            return [
                {
                    "id": row.id,
                    "node_id": row.node_id,
                    "suggestion_type": row.suggestion_type,
                    "title": row.title,
                    "description": row.description,
                    "evidence": json.loads(row.evidence),
                    "proposed_changes": json.loads(row.proposed_changes),
                    "confidence": row.confidence,
                    "impact_estimate": row.impact_estimate,
                    "status": row.status,
                    "created_at": row.created_at,
                    "reviewed_by": row.reviewed_by,
                    "reviewed_at": row.reviewed_at,
                    "notes": row.notes,
                }
                for row in rows
            ]

    def update_suggestion_status(
        self,
        suggestion_id: str,
        status: str,
        reviewed_by: str = "",
        notes: str = "",
    ) -> bool:
        with self._session() as session:
            row = session.get(SuggestionRow, suggestion_id)
            if not row:
                return False
            row.status = status
            row.reviewed_by = reviewed_by
            row.reviewed_at = datetime.utcnow()
            row.notes = notes
            session.commit()
            return True
