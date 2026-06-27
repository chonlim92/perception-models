# [IMPLEMENTED BY CLAUDE - was missing]
"""
FastAPI application for the FST (Functional Scenario Tree) interactive system.

Run with: uvicorn scenario_trees.api.app:app --reload --port 8000
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..taxonomy.scenario_tree import build_default_tree, ScenarioTreeNode
from ..taxonomy.scenario_schema import ScenarioTreeModel
from .database import FSTDatabase
from .analysis_engine import RootCauseAnalyzer
from .models import (
    AddChildRequest,
    ApproveSuggestionRequest,
    AttachRecordingRequest,
    BulkImportRequest,
    CreateScriptRequest,
    CreateVersionRequest,
    KPIConfigRequest,
    KPIConfigResponse,
    RunEvaluationRequest,
    SplitNodeRequest,
    UpdateNodeRequest,
)

app = FastAPI(
    title="FST - Functional Scenario Tree API",
    description="Interactive scenario tree management with metrics, versioning, and root cause analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = FSTDatabase("fst.db")
analyzer = RootCauseAnalyzer(db)


def _ensure_initial_version() -> None:
    """Create initial tree version if none exists."""
    current = db.get_current_tree_version()
    if current is None:
        tree = build_default_tree()
        tree_model = ScenarioTreeModel.from_tree_node(tree)
        tree_data = tree_model.to_dict()
        db.save_tree_version(
            tree_data=tree_data,
            version="v1.0.0",
            created_by="system",
            change_description="Initial PEGASUS/ASAM 6-layer functional scenario tree",
        )


_ensure_initial_version()


# --- Tree Endpoints ---


@app.get("/api/tree")
def get_current_tree() -> Dict[str, Any]:
    """Get the current version of the scenario tree."""
    version = db.get_current_tree_version()
    if not version:
        raise HTTPException(status_code=404, detail="No tree version found")
    return version


@app.get("/api/tree/versions")
def list_tree_versions() -> List[Dict[str, Any]]:
    """List all tree versions with metadata."""
    return db.list_tree_versions()


@app.get("/api/tree/versions/{version_id}")
def get_tree_version(version_id: str) -> Dict[str, Any]:
    """Get a specific tree version by ID."""
    version = db.get_tree_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    return version


@app.post("/api/tree/versions")
def create_tree_version(request: CreateVersionRequest) -> Dict[str, Any]:
    """Create a new tree version (snapshot current state)."""
    current = db.get_current_tree_version()
    if not current:
        raise HTTPException(status_code=404, detail="No current tree to snapshot")

    version_str = request.version
    if not version_str:
        versions = db.list_tree_versions()
        if versions:
            last = versions[0]["version"]
            parts = last.replace("v", "").split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            version_str = "v" + ".".join(parts)
        else:
            version_str = "v1.0.0"

    version_id = db.save_tree_version(
        tree_data=current["tree_data"],
        version=version_str,
        created_by=request.created_by,
        change_description=request.change_description,
        parent_version_id=current["id"],
    )
    return {"id": version_id, "version": version_str}


@app.put("/api/tree/nodes/{node_id}")
def update_node(node_id: str, request: UpdateNodeRequest) -> Dict[str, Any]:
    """Update a node's properties in the current tree."""
    current = db.get_current_tree_version()
    if not current:
        raise HTTPException(status_code=404, detail="No current tree")

    tree_data = current["tree_data"]
    node = _find_node_in_tree(tree_data["root"], node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    if request.name is not None:
        node["name"] = request.name
    if request.description is not None:
        node["description"] = request.description
    if request.detection_method is not None:
        node["detection_method"] = request.detection_method

    version_id = db.save_tree_version(
        tree_data=tree_data,
        version=_next_patch_version(current["version"]),
        created_by="api",
        change_description=f"Updated node {node_id}",
        parent_version_id=current["id"],
    )
    return {"status": "updated", "version_id": version_id}


@app.post("/api/tree/nodes/{parent_id}/children")
def add_child_node(parent_id: str, request: AddChildRequest) -> Dict[str, Any]:
    """Add a new child node to an existing parent."""
    current = db.get_current_tree_version()
    if not current:
        raise HTTPException(status_code=404, detail="No current tree")

    tree_data = current["tree_data"]
    parent = _find_node_in_tree(tree_data["root"], parent_id)
    if not parent:
        raise HTTPException(status_code=404, detail=f"Parent {parent_id} not found")

    new_child = {
        "id": request.id,
        "name": request.name,
        "layer": request.layer,
        "description": request.description,
        "detection_method": request.detection_method,
        "parent_id": parent_id,
        "children": [],
    }
    parent["children"].append(new_child)

    version_id = db.save_tree_version(
        tree_data=tree_data,
        version=_next_minor_version(current["version"]),
        created_by="api",
        change_description=f"Added child '{request.name}' to {parent_id}",
        parent_version_id=current["id"],
    )
    return {"status": "created", "node_id": request.id, "version_id": version_id}


@app.delete("/api/tree/nodes/{node_id}")
def delete_node(node_id: str) -> Dict[str, Any]:
    """Remove a node from the tree (and all its children)."""
    current = db.get_current_tree_version()
    if not current:
        raise HTTPException(status_code=404, detail="No current tree")

    tree_data = current["tree_data"]
    if not _remove_node_from_tree(tree_data["root"], node_id):
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    version_id = db.save_tree_version(
        tree_data=tree_data,
        version=_next_minor_version(current["version"]),
        created_by="api",
        change_description=f"Removed node {node_id}",
        parent_version_id=current["id"],
    )
    return {"status": "deleted", "version_id": version_id}


@app.post("/api/tree/nodes/{node_id}/split")
def split_node(node_id: str, request: SplitNodeRequest) -> Dict[str, Any]:
    """Split a node into sub-branches."""
    current = db.get_current_tree_version()
    if not current:
        raise HTTPException(status_code=404, detail="No current tree")

    tree_data = current["tree_data"]
    parent = _find_node_in_tree(tree_data["root"], node_id)
    if not parent:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    new_children = []
    for branch_name in request.branch_names:
        child_id = f"{node_id}.{branch_name}"
        child = {
            "id": child_id,
            "name": branch_name.replace("_", " ").title(),
            "layer": parent["layer"],
            "description": f"Split by {request.split_criteria}: {branch_name}",
            "detection_method": parent.get("detection_method", ""),
            "parent_id": node_id,
            "children": [],
        }
        parent["children"].append(child)
        new_children.append(child_id)

    if request.auto_reassign:
        recordings = db.get_node_recordings(node_id)
        for rec_id in recordings:
            rec = db.get_recording(rec_id)
            if not rec:
                continue
            attrs = rec.get("attributes", {})
            criteria_value = attrs.get(request.split_criteria)

            if criteria_value and str(criteria_value).lower() in [
                b.lower() for b in request.branch_names
            ]:
                target_branch = f"{node_id}.{criteria_value.lower()}"
                db.attach_recording(target_branch, rec_id)
            else:
                target_branch = f"{node_id}.{request.branch_names[-1]}"
                db.attach_recording(target_branch, rec_id)

    version_id = db.save_tree_version(
        tree_data=tree_data,
        version=_next_minor_version(current["version"]),
        created_by="api",
        change_description=f"Split node {node_id} by {request.split_criteria}",
        parent_version_id=current["id"],
    )
    return {
        "status": "split",
        "new_children": new_children,
        "version_id": version_id,
    }


# --- Recording Endpoints ---


@app.get("/api/nodes/{node_id}/recordings")
def get_node_recordings(node_id: str) -> List[Dict[str, Any]]:
    """Get all recordings attached to a node."""
    return db.get_recordings_for_node(node_id)


@app.post("/api/nodes/{node_id}/recordings")
def attach_recording(node_id: str, request: AttachRecordingRequest) -> Dict[str, Any]:
    """Attach a recording to a node."""
    if request.path:
        db.upsert_recording(request.recording_id, request.path)
    db.attach_recording(node_id, request.recording_id)
    return {"status": "attached", "node_id": node_id, "recording_id": request.recording_id}


@app.delete("/api/nodes/{node_id}/recordings/{recording_id}")
def detach_recording(node_id: str, recording_id: str) -> Dict[str, Any]:
    """Detach a recording from a node."""
    db.detach_recording(node_id, recording_id)
    return {"status": "detached"}


@app.post("/api/recordings/bulk-import")
def bulk_import_recordings(request: BulkImportRequest) -> Dict[str, Any]:
    """Bulk import recordings with optional node attachment."""
    imported = 0
    for rec in request.recordings:
        rec_id = rec.get("id", rec.get("recording_id", ""))
        if not rec_id:
            continue
        db.upsert_recording(
            recording_id=rec_id,
            path=rec.get("path", ""),
            timestamp=rec.get("timestamp"),
            duration=rec.get("duration"),
            location=rec.get("location"),
            attributes=rec.get("attributes"),
        )
        if request.node_id:
            db.attach_recording(request.node_id, rec_id)
        imported += 1
    return {"status": "imported", "count": imported}


# --- Metrics & Evaluation Endpoints ---


@app.get("/api/nodes/{node_id}/metrics")
def get_node_metrics(node_id: str) -> Dict[str, Any]:
    """Get aggregated metrics for a node."""
    summary = db.get_node_metrics_summary(node_id)
    kpis = db.get_kpis(node_id)

    kpi_status = "pass"
    failing_metrics = []
    for kpi in kpis:
        metric_name = kpi["metric_name"]
        if metric_name in summary.get("metrics", {}):
            mean_val = summary["metrics"][metric_name]["mean"]
            if kpi["direction"] == "above" and mean_val < kpi["threshold"]:
                kpi_status = "fail"
                failing_metrics.append(metric_name)
            elif kpi["direction"] == "below" and mean_val > kpi["threshold"]:
                kpi_status = "fail"
                failing_metrics.append(metric_name)
            elif kpi["direction"] == "above" and mean_val < kpi["threshold"] + kpi["warning_margin"]:
                if kpi_status != "fail":
                    kpi_status = "warn"

    summary["kpi_status"] = kpi_status
    summary["failing_metrics"] = failing_metrics
    summary["kpi_configs"] = kpis
    return summary


@app.post("/api/nodes/{node_id}/evaluate")
def run_evaluation(node_id: str, request: RunEvaluationRequest) -> Dict[str, Any]:
    """
    Run evaluation scripts on a node's recordings.

    Note: In production, this would execute Python scripts in a sandbox.
    For this implementation, it stores mock results for demonstration.
    """
    scripts = db.get_scripts(node_id)
    if not scripts:
        raise HTTPException(status_code=404, detail="No evaluation scripts for this node")

    recordings = db.get_node_recordings(node_id)
    if request.recording_ids:
        recordings = [r for r in recordings if r in request.recording_ids]

    if not recordings:
        raise HTTPException(status_code=404, detail="No recordings to evaluate")

    results_count = 0
    for script in scripts:
        if request.script_id and script["id"] != request.script_id:
            continue
        for rec_id in recordings:
            db.add_evaluation_result(
                node_id=node_id,
                recording_id=rec_id,
                metric_name="evaluation_pending",
                metric_value=0.0,
                script_id=script["id"],
            )
            results_count += 1

    return {
        "status": "evaluation_queued",
        "scripts_run": len(scripts),
        "recordings_evaluated": len(recordings),
        "results_count": results_count,
    }


@app.get("/api/nodes/{node_id}/kpi")
def get_kpi_config(node_id: str) -> Dict[str, Any]:
    """Get KPI configuration for a node."""
    configs = db.get_kpis(node_id)
    return {"node_id": node_id, "configs": configs}


@app.put("/api/nodes/{node_id}/kpi")
def set_kpi_config(node_id: str, request: KPIConfigRequest) -> Dict[str, Any]:
    """Set or update KPI threshold for a node."""
    db.set_kpi(
        node_id=node_id,
        metric_name=request.metric_name,
        threshold=request.threshold,
        direction=request.direction,
        warning_margin=request.warning_margin,
    )
    return {"status": "kpi_set", "node_id": node_id, "metric": request.metric_name}


@app.get("/api/nodes/{node_id}/scripts")
def get_scripts(node_id: str) -> List[Dict[str, Any]]:
    """Get evaluation scripts for a node."""
    return db.get_scripts(node_id)


@app.post("/api/nodes/{node_id}/scripts")
def add_script(node_id: str, request: CreateScriptRequest) -> Dict[str, Any]:
    """Add an evaluation script to a node."""
    script_id = db.add_script(
        node_id=node_id,
        name=request.name,
        script_content=request.script_content,
    )
    return {"status": "created", "script_id": script_id}


# --- Root Cause Analysis Endpoints ---


@app.post("/api/nodes/{node_id}/analyze")
def trigger_analysis(node_id: str) -> Dict[str, Any]:
    """Trigger root cause analysis for a failing node."""
    result = analyzer.analyze_node(node_id)
    return result


@app.get("/api/nodes/{node_id}/suggestions")
def get_suggestions(
    node_id: str,
    status: Optional[str] = Query(None, description="Filter by status"),
) -> List[Dict[str, Any]]:
    """Get pending suggestions for a node."""
    return db.get_suggestions(node_id=node_id, status=status)


@app.post("/api/suggestions/{suggestion_id}/approve")
def approve_suggestion(
    suggestion_id: str, request: ApproveSuggestionRequest
) -> Dict[str, Any]:
    """Approve a suggestion and apply changes."""
    suggestions = db.get_suggestions()
    suggestion = next(
        (s for s in suggestions if s["id"] == suggestion_id), None
    )
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    db.update_suggestion_status(
        suggestion_id=suggestion_id,
        status="approved",
        reviewed_by=request.reviewed_by,
        notes=request.notes,
    )

    if suggestion["suggestion_type"] == "split":
        current = db.get_current_tree_version()
        if current:
            analyzer.apply_split_suggestion(
                suggestion_id, suggestion["node_id"], current["tree_data"]
            )

    return {"status": "approved", "suggestion_id": suggestion_id}


@app.post("/api/suggestions/{suggestion_id}/reject")
def reject_suggestion(
    suggestion_id: str, request: ApproveSuggestionRequest
) -> Dict[str, Any]:
    """Reject a suggestion."""
    success = db.update_suggestion_status(
        suggestion_id=suggestion_id,
        status="rejected",
        reviewed_by=request.reviewed_by,
        notes=request.notes,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    return {"status": "rejected", "suggestion_id": suggestion_id}


# --- Helper Functions ---


def _find_node_in_tree(node: Dict[str, Any], target_id: str) -> Optional[Dict[str, Any]]:
    """Recursively find a node by ID in the tree dict."""
    if node.get("id") == target_id:
        return node
    for child in node.get("children", []):
        result = _find_node_in_tree(child, target_id)
        if result:
            return result
    return None


def _remove_node_from_tree(node: Dict[str, Any], target_id: str) -> bool:
    """Recursively remove a node by ID. Returns True if found and removed."""
    children = node.get("children", [])
    for i, child in enumerate(children):
        if child.get("id") == target_id:
            children.pop(i)
            return True
        if _remove_node_from_tree(child, target_id):
            return True
    return False


def _next_patch_version(current: str) -> str:
    """Increment patch version: v1.0.3 -> v1.0.4"""
    parts = current.replace("v", "").split(".")
    if len(parts) == 3:
        parts[2] = str(int(parts[2]) + 1)
    return "v" + ".".join(parts)


def _next_minor_version(current: str) -> str:
    """Increment minor version: v1.2.3 -> v1.3.0"""
    parts = current.replace("v", "").split(".")
    if len(parts) == 3:
        parts[1] = str(int(parts[1]) + 1)
        parts[2] = "0"
    return "v" + ".".join(parts)
