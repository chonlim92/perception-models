# FST API - Functional Scenario Tree Backend

## Overview

FastAPI backend powering the FST interactive visualization system. Provides REST endpoints for tree management, metrics evaluation, recording attachment, and semi-automated root cause analysis.

## Installation

```bash
pip install fastapi uvicorn sqlalchemy pydantic pyyaml
```

## Running

```bash
# From the repository root
uvicorn scenario_trees.api.app:app --reload --port 8000

# API docs available at:
# http://localhost:8000/docs  (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

## Architecture

```
api/
├── __init__.py           # Package init
├── app.py               # FastAPI application (routes + startup)
├── models.py            # Pydantic request/response schemas
├── database.py          # SQLAlchemy ORM (versioning, KPIs, scripts, suggestions)
├── analysis_engine.py   # Root cause analysis + suggestion generation
└── README.md            # This file
```

## Key Components

### Tree Versioning
Every tree modification creates a new immutable version snapshot. The `is_current` flag tracks the active version. Versions form a linked list via `parent_version_id`.

### Root Cause Analysis Engine
When triggered on a failing node:
1. Classifies recordings as pass/fail based on KPI configs
2. Collects attributes from failing vs passing recordings
3. Computes statistical lift for each attribute value
4. Generates suggestions with confidence scores

### Suggestion Types
- **split**: Propose splitting a node into sub-branches based on a discriminating attribute
- **investigate**: Flag a correlation that warrants manual review
- **adjust_threshold**: Suggest relaxing a KPI when no pattern is found
- **reassign**: Move recordings to a more appropriate node

## Database Schema

Tables:
- `tree_versions` - Immutable tree snapshots with version strings
- `node_kpis` - KPI threshold configurations per node
- `evaluation_scripts` - Python scripts attached to nodes
- `evaluation_results` - Metric values per recording per node
- `node_recordings` - Many-to-many recording-node attachments
- `suggestions` - Generated and reviewed suggestions
- `recording_metadata` - Recording attributes for pattern mining

## Example Usage

```python
import requests

BASE = "http://localhost:8000/api"

# Get current tree
tree = requests.get(f"{BASE}/tree").json()

# Set a KPI
requests.put(f"{BASE}/nodes/L4.vehicle/kpi", json={
    "metric_name": "mAP",
    "threshold": 0.7,
    "direction": "above",
})

# Submit evaluation results from external evaluator
requests.post(f"{BASE}/nodes/L4.vehicle/results", json={
    "recording_id": "rec_001",
    "metrics": {"mAP": 0.72, "NDS": 0.68}
})

# Trigger root cause analysis
analysis = requests.post(f"{BASE}/nodes/L4.vehicle/analyze").json()
print(analysis["patterns"])
print(analysis["suggestions"])

# Approve a suggestion
requests.post(f"{BASE}/suggestions/{suggestion_id}/approve", json={
    "reviewed_by": "developer",
    "notes": "Looks good, bicycle pattern is clear"
})
```
