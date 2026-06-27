# [IMPLEMENTED BY CLAUDE - was missing]
"""Pydantic models for the FST API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TreeNodeResponse(BaseModel):
    """A single node in the FST tree response."""
    id: str
    name: str
    layer: int
    description: str = ""
    detection_method: str = ""
    parent_id: Optional[str] = None
    children: List["TreeNodeResponse"] = Field(default_factory=list)
    kpi_status: Optional[str] = None  # "pass", "warn", "fail", None
    recording_count: int = 0
    metrics_summary: Optional[Dict[str, float]] = None


class TreeVersionResponse(BaseModel):
    """A tree version entry."""
    id: str
    version: str
    created_at: datetime
    created_by: str
    change_description: str
    parent_version_id: Optional[str] = None


class TreeVersionDetailResponse(TreeVersionResponse):
    """Full tree version with data."""
    tree_data: Dict[str, Any]


class CreateVersionRequest(BaseModel):
    """Request to create a new tree version."""
    version: Optional[str] = None  # Auto-increment if not specified
    change_description: str = ""
    created_by: str = "system"


class UpdateNodeRequest(BaseModel):
    """Request to update a node's properties."""
    name: Optional[str] = None
    description: Optional[str] = None
    detection_method: Optional[str] = None


class AddChildRequest(BaseModel):
    """Request to add a child node."""
    id: str
    name: str
    layer: int
    description: str = ""
    detection_method: str = ""


class SplitNodeRequest(BaseModel):
    """Request to split a node into sub-branches."""
    split_criteria: str  # e.g., "has_bicycle"
    branch_names: List[str]  # e.g., ["with_bicycle", "without_bicycle"]
    auto_reassign: bool = True  # Auto-reassign recordings based on criteria


class RecordingResponse(BaseModel):
    """A recording/measurement attached to a node."""
    id: str
    path: str
    timestamp: Optional[datetime] = None
    duration: Optional[float] = None
    location: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metrics: Optional[Dict[str, float]] = None


class AttachRecordingRequest(BaseModel):
    """Request to attach a recording to a node."""
    recording_id: str
    path: Optional[str] = None


class BulkImportRequest(BaseModel):
    """Request to bulk import recordings."""
    recordings: List[Dict[str, Any]]
    node_id: Optional[str] = None


class MetricsSummaryResponse(BaseModel):
    """Metrics summary for a node."""
    node_id: str
    total_recordings: int
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max, pass_rate}
    kpi_status: str  # "pass", "warn", "fail"
    failing_recordings: List[str] = Field(default_factory=list)
    trend: Optional[Dict[str, List[float]]] = None  # metric_name -> time series


class KPIConfigRequest(BaseModel):
    """Request to set KPI thresholds."""
    metric_name: str
    threshold: float
    direction: Literal["above", "below"] = "above"
    warning_margin: float = 0.1  # Warn at threshold +/- margin


class KPIConfigResponse(BaseModel):
    """KPI configuration for a node."""
    node_id: str
    configs: List[Dict[str, Any]]


class EvaluationScriptResponse(BaseModel):
    """An evaluation script attached to a node."""
    id: str
    node_id: str
    name: str
    script_content: str
    created_at: datetime
    last_run: Optional[datetime] = None
    last_result: Optional[Dict[str, Any]] = None


class CreateScriptRequest(BaseModel):
    """Request to create/update an evaluation script."""
    name: str
    script_content: str


class RunEvaluationRequest(BaseModel):
    """Request to run evaluation on a node."""
    script_id: Optional[str] = None  # Run specific script, or all if None
    recording_ids: Optional[List[str]] = None  # Specific recordings, or all if None


class RootCauseAnalysisResponse(BaseModel):
    """Result of root cause analysis for a failing node."""
    node_id: str
    analysis_timestamp: datetime
    failing_count: int
    total_count: int
    failure_rate: float
    patterns: List[PatternResult] = Field(default_factory=list)
    suggestions: List[SuggestionResponse] = Field(default_factory=list)


class PatternResult(BaseModel):
    """A discovered pattern in failing recordings."""
    attribute: str  # e.g., "has_bicycle", "weather_rain"
    prevalence_in_failures: float  # % of failures with this attribute
    prevalence_overall: float  # % of all recordings with this attribute
    lift: float  # prevalence_in_failures / prevalence_overall
    confidence: float  # Statistical confidence
    sample_recordings: List[str] = Field(default_factory=list)


class SuggestionResponse(BaseModel):
    """A suggested action based on root cause analysis."""
    id: str
    node_id: str
    suggestion_type: Literal["split", "reassign", "adjust_threshold", "investigate"]
    title: str
    description: str
    evidence: Dict[str, Any]
    proposed_changes: Dict[str, Any]
    confidence: float
    impact_estimate: str  # "high", "medium", "low"
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class ApproveSuggestionRequest(BaseModel):
    """Request to approve/reject a suggestion."""
    reviewed_by: str = "developer"
    notes: str = ""


# Rebuild forward refs for recursive models
TreeNodeResponse.model_rebuild()
