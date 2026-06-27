# FST Interactive System — Complete Guide

## What is the FST System?

The Functional Scenario Tree (FST) Interactive System is a web-based platform for managing, visualizing, and analyzing the PEGASUS/ASAM-based scenario taxonomy used in autonomous driving perception validation.

It answers the question: **"Where does our perception system fail, and why?"**

## System Components

### 1. Backend API (`scenario_trees/api/`)

A FastAPI application providing:
- **Tree CRUD with Versioning**: Every tree modification creates an immutable snapshot
- **Recording Management**: Attach measurements/recordings to tree nodes
- **Metrics & KPI Engine**: Define thresholds, run evaluations, monitor pass/fail
- **Root Cause Analysis**: Statistical pattern mining when nodes fail
- **Suggestion Workflow**: Propose, review, and apply tree modifications

### 2. React Frontend (`fst-frontend/`)

An interactive web interface featuring:
- **Tree Canvas**: Zoom/pan/click visualization using ReactFlow
- **Node Inspector**: Detailed panel showing metrics, recordings, and analysis
- **Metrics Dashboard**: Charts, KPI cards, failure alerts
- **Suggestions Panel**: Review and approve/reject automated recommendations
- **Version Timeline**: Navigate between tree versions

## Quick Start

### Terminal 1 — Backend
```bash
cd perception-models
pip install fastapi uvicorn sqlalchemy pydantic pyyaml
uvicorn scenario_trees.api.app:app --reload --port 8000
```

### Terminal 2 — Frontend
```bash
cd fst-frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

## How It Works

### The Scenario Tree

The tree follows the PEGASUS/ASAM taxonomy with 6 layers:

| Layer | Name | Examples |
|-------|------|----------|
| L1 | Road Topology | Highway, Urban, Roundabout, Curve |
| L2 | Traffic Infrastructure | Traffic lights, Signs, Markings |
| L3 | Temporary Modifications | Construction, Detours, Closures |
| L4 | Dynamic Objects | Cars, Pedestrians, Bicycles, Cut-ins |
| L5 | Environment | Rain, Fog, Night, Snow |
| L6 | Digital Information | Sensor degradation, Map accuracy |

### The Workflow

```
1. DEFINE TREE        → Structure your scenario taxonomy
2. ATTACH RECORDINGS  → Link real-world data to nodes
3. SET KPIs           → Define pass/fail thresholds per node
4. RUN EVALUATION     → Execute metric scripts on recordings
5. MONITOR            → Dashboard shows pass/warn/fail status
6. ANALYZE FAILURES   → Root cause engine mines patterns
7. REVIEW SUGGESTIONS → Approve or reject proposed changes
8. ITERATE            → Tree evolves, new version created
```

### Root Cause Analysis — Detailed Flow

When a node fails its KPI:

**Step 1: Detection**
The metrics dashboard shows a node in "fail" state (red). For example, node `L4.vehicle` has mAP=0.58 but the threshold is 0.70.

**Step 2: Trigger Analysis**
Click "Run Analysis" in the node detail panel. The engine:
- Identifies all recordings that fail the KPI
- Collects attributes from each recording's metadata
- Compares attribute distributions between failing and passing recordings
- Computes statistical "lift" — how much more likely an attribute is in failures

**Step 3: Pattern Discovery**
The engine discovers, for example:
- 83% of failing recordings have `object_types=bicycle`
- Only 25% of all recordings have bicycles
- **Lift = 3.3x** — bicycles are 3.3x more likely in failures

**Step 4: Suggestion Generation**
Based on high-lift patterns, the engine suggests:
> "Split node by 'object_types': with/without 'bicycle'"
> Confidence: 72% | Impact: High

This means creating two new child nodes:
- `L4.vehicle.with_bicycle` — gets the 25 recordings with bicycles
- `L4.vehicle.without_bicycle` — gets the remaining 75 recordings

**Step 5: Human Review**
The developer reviews the suggestion:
- Sees the evidence (lift, sample recordings, confidence)
- Can approve (applies the split) or reject (dismisses it)
- On approval: tree is modified, recordings reassigned, new version created

**Step 6: Iterate**
After splitting, each sub-node can be independently monitored. The `with_bicycle` node likely still fails, but now the team knows *why* and can focus training data collection or model improvements specifically on bicycle scenarios.

## API Reference

### Tree Endpoints

```bash
# Get current tree (full JSON including all nodes)
curl http://localhost:8000/api/tree

# List all versions
curl http://localhost:8000/api/tree/versions

# Create a version snapshot
curl -X POST http://localhost:8000/api/tree/versions \
  -H "Content-Type: application/json" \
  -d '{"change_description": "Added bicycle split", "created_by": "john"}'

# Update a node
curl -X PUT http://localhost:8000/api/tree/nodes/L4.vehicle \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description for vehicle detection"}'

# Add a child node
curl -X POST http://localhost:8000/api/tree/nodes/L4.vehicle/children \
  -H "Content-Type: application/json" \
  -d '{"id": "L4.vehicle.e_scooter", "name": "E-Scooter", "layer": 4}'

# Split a node
curl -X POST http://localhost:8000/api/tree/nodes/L4.vehicle/split \
  -H "Content-Type: application/json" \
  -d '{
    "split_criteria": "has_bicycle",
    "branch_names": ["with_bicycle", "without_bicycle"],
    "auto_reassign": true
  }'
```

### Recording Endpoints

```bash
# Attach a recording
curl -X POST http://localhost:8000/api/nodes/L4.vehicle/recordings \
  -H "Content-Type: application/json" \
  -d '{"recording_id": "rec_001", "path": "/data/recordings/rec_001.bag"}'

# Bulk import
curl -X POST http://localhost:8000/api/recordings/bulk-import \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "L4.vehicle",
    "recordings": [
      {"id": "rec_001", "path": "/data/rec_001.bag", "duration": 30.5, "location": "munich"},
      {"id": "rec_002", "path": "/data/rec_002.bag", "duration": 45.2, "location": "stuttgart"}
    ]
  }'
```

### Metrics & KPI Endpoints

```bash
# Set KPI threshold
curl -X PUT http://localhost:8000/api/nodes/L4.vehicle/kpi \
  -H "Content-Type: application/json" \
  -d '{"metric_name": "mAP", "threshold": 0.7, "direction": "above", "warning_margin": 0.05}'

# Get metrics summary
curl http://localhost:8000/api/nodes/L4.vehicle/metrics

# Add evaluation script
curl -X POST http://localhost:8000/api/nodes/L4.vehicle/scripts \
  -H "Content-Type: application/json" \
  -d '{
    "name": "compute_detection_metrics",
    "script_content": "import numpy as np\n# compute mAP here..."
  }'
```

### Analysis Endpoints

```bash
# Trigger root cause analysis
curl -X POST http://localhost:8000/api/nodes/L4.vehicle/analyze

# Get pending suggestions
curl http://localhost:8000/api/nodes/L4.vehicle/suggestions?status=pending

# Approve a suggestion
curl -X POST http://localhost:8000/api/suggestions/<suggestion_id>/approve \
  -H "Content-Type: application/json" \
  -d '{"reviewed_by": "developer", "notes": "Pattern is clear"}'
```

## Recording Attributes for Pattern Mining

For root cause analysis to work effectively, recordings should have rich attributes in their metadata:

```json
{
  "id": "rec_001",
  "path": "/data/rec_001.bag",
  "duration": 30.5,
  "location": "munich_inner_city",
  "attributes": {
    "object_types": ["car", "bicycle", "pedestrian"],
    "weather": "rain",
    "time_of_day": "dusk",
    "road_type": "urban",
    "has_bicycle": true,
    "has_construction": false,
    "num_objects": 12,
    "ego_speed_avg": 35.2
  }
}
```

The analysis engine looks at ALL attribute keys/values and finds which ones are statistically overrepresented in failures.

## Versioning Strategy

- **Patch version** (v1.0.X): Node property updates, metadata changes
- **Minor version** (v1.X.0): Structural changes — add/remove/split nodes
- **Major version** (vX.0.0): Reserved for taxonomy restructuring

All versions are immutable — you can always revert to any previous state.

## Integration with Perception Models

The FST system connects to the perception model evaluation pipeline:

1. **Run model evaluation** on a dataset of recordings
2. **Import results** as metrics attached to tree nodes
3. **Monitor KPIs** per scenario category
4. **Analyze failures** to identify which scenarios need improvement
5. **Guide data collection** by identifying underperforming scenario types

This creates a closed feedback loop between model development and scenario management.
