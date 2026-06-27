# FST Interactive Frontend + Model Completeness Audit

**Date:** 2026-06-27
**Status:** IN PROGRESS
**Session:** Autonomous implementation (user sleeping)

---

## Part 1: Model Completeness Audit

### Scope
Verify all 14 perception models have complete:
- Training scripts (optimizer, scheduler, data loading, checkpointing)
- Loss functions (per-paper implementations, not stubs)
- Evaluation scripts (proper metric computation)
- Inference pipelines (end-to-end single-sample)
- TensorFlow implementations (functional, not stubs)

### Known Gaps (from implementation_status.md)
- Radar Occupancy TensorFlow model: ~80% complete
- Missing: losses.py for radar_occupancy, tests, scripts

### Models Being Audited
| Sensor | Category | Model |
|--------|----------|-------|
| Camera | Dynamic Objects | BEVFormer, DETR3D, PETR/StreamPETR |
| Camera | Static Map | HDMapNet, MapTR, StreamMapNet |
| LiDAR | Dynamic Objects | CenterPoint, PointNet++, PointPillars |
| LiDAR | Static Map | Cylinder3D, RangeNet++ |
| Radar | Dynamic Objects | CRAFT, RadarPillarNet |
| Radar | Static Map | Radar Occupancy |

### Tag Convention
All code implemented to fill gaps will be tagged with:
```python
# [IMPLEMENTED BY CLAUDE - was missing]
```

---

## Part 2: FST React Frontend Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  React Frontend (Vite + TypeScript)       │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Tree View│  │ Node Detail  │  │ Root Cause Panel │  │
│  │ (D3/     │  │ Dashboard    │  │ + Suggestions    │  │
│  │  ReactFlow)│ │ (Metrics)   │  │                  │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │ REST API
┌─────────────────────────┴───────────────────────────────┐
│              FastAPI Backend (Python)                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Tree API │  │ Metrics API  │  │ Analysis Engine  │  │
│  │ (CRUD +  │  │ (Evaluation  │  │ (Root Cause +    │  │
│  │ Version) │  │  Results)    │  │  Suggestions)    │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────┐
│              SQLite + File Storage                        │
│  - Tree versions (JSON snapshots)                        │
│  - Recordings/measurements DB                            │
│  - Metric results                                        │
│  - Evaluation scripts per node                           │
└─────────────────────────────────────────────────────────┘
```

### Core Features

#### 1. Interactive Tree Visualization
- Hierarchical tree with expand/collapse
- Color-coded nodes based on KPI status (green/yellow/red)
- Drag-and-drop for reorganization
- Search and filter by layer/category/status
- Zoom and pan for large trees

#### 2. Tree Versioning
- Every modification creates a new version (immutable snapshots)
- Version history with diff view
- Ability to branch and merge tree versions
- Rollback to any previous version
- Git-like semantic versioning (v1.0.0, v1.1.0, etc.)

#### 3. Per-Node Metrics Dashboard
- Each node shows aggregated metrics from attached recordings
- KPI thresholds configurable per node
- Pass/Fail/Warning status indicators
- Time-series trend charts
- Comparison with parent/sibling nodes

#### 4. Recording/Measurement Attachment
- Attach recordings to leaf or intermediate nodes
- Bulk import from filesystem paths
- Metadata display (duration, location, timestamp)
- Preview of recording contents
- Re-assignment between nodes

#### 5. Evaluation Script Management
- Attach Python metric scripts to any node
- Script editor (Monaco-based)
- Run evaluations on demand or scheduled
- Results stored per recording per node
- Script templates for common metrics

#### 6. Semi-Automated Root Cause Analysis
When a node fails its KPI threshold:
1. **Detection**: System detects KPI breach
2. **Pattern Mining**: Analyze failing recordings for common attributes
   - Object types present (bicycle, pedestrian, truck...)
   - Weather conditions
   - Time of day
   - Road type
   - Sensor conditions
3. **Suggestion Generation**: Propose tree modifications
   - Split node into sub-branches (e.g., "with bicycle" / "without bicycle")
   - Suggest new evaluation criteria
   - Identify recording outliers
4. **Approval Workflow**:
   - Suggestions displayed with evidence/confidence
   - Developer can approve/reject/modify
   - On approval: tree is updated, recordings reassigned
   - Version bump with change log

### Tech Stack
- **Frontend**: React 18 + TypeScript + Vite
- **Tree Visualization**: ReactFlow (for node graph) + D3.js (for metrics charts)
- **State Management**: Zustand
- **UI Components**: Shadcn/ui + Tailwind CSS
- **Backend**: FastAPI (Python)
- **Database**: SQLite (existing) + JSON file versioning
- **Charts**: Recharts

### API Endpoints

```
# Tree Management
GET    /api/tree                      - Get current tree
GET    /api/tree/versions             - List all versions
GET    /api/tree/versions/:id         - Get specific version
POST   /api/tree/versions             - Create new version (snapshot)
PUT    /api/tree/nodes/:nodeId        - Update node
POST   /api/tree/nodes/:parentId/children - Add child node
DELETE /api/tree/nodes/:nodeId        - Remove node
POST   /api/tree/nodes/:nodeId/split  - Split node (with suggestions)

# Recordings/Measurements
GET    /api/nodes/:nodeId/recordings  - Get recordings for node
POST   /api/nodes/:nodeId/recordings  - Attach recording
DELETE /api/nodes/:nodeId/recordings/:recId - Detach recording
POST   /api/recordings/bulk-import    - Bulk import recordings

# Metrics & Evaluation
GET    /api/nodes/:nodeId/metrics     - Get metrics summary
POST   /api/nodes/:nodeId/evaluate    - Run evaluation
GET    /api/nodes/:nodeId/kpi         - Get KPI status
PUT    /api/nodes/:nodeId/kpi         - Set KPI thresholds
GET    /api/nodes/:nodeId/scripts     - Get evaluation scripts
POST   /api/nodes/:nodeId/scripts     - Add evaluation script

# Root Cause Analysis
POST   /api/nodes/:nodeId/analyze     - Trigger root cause analysis
GET    /api/nodes/:nodeId/suggestions - Get pending suggestions
POST   /api/suggestions/:id/approve   - Approve suggestion
POST   /api/suggestions/:id/reject    - Reject suggestion
```

### Data Models (Backend)

```python
# Tree Version
class TreeVersion:
    id: str                  # UUID
    version: str             # Semantic version (v1.0.0)
    tree_data: dict          # Full tree JSON snapshot
    created_at: datetime
    created_by: str
    change_description: str
    parent_version_id: Optional[str]

# Node KPI Configuration
class NodeKPI:
    node_id: str
    metric_name: str         # e.g., "mAP", "recall"
    threshold: float         # Minimum acceptable value
    direction: str           # "above" or "below"

# Evaluation Script
class EvaluationScript:
    id: str
    node_id: str
    name: str
    script_content: str      # Python code
    created_at: datetime
    last_run: Optional[datetime]

# Root Cause Suggestion
class NodeSuggestion:
    id: str
    node_id: str
    suggestion_type: str     # "split", "reassign", "adjust_threshold"
    evidence: dict           # Pattern analysis results
    proposed_changes: dict   # What would change
    confidence: float
    status: str              # "pending", "approved", "rejected"
    created_at: datetime
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]
```

---

## Implementation Plan

### Phase 1: Model Audit & Fix (Current)
1. Run parallel audits on camera/lidar/radar models
2. Identify gaps
3. Implement missing components
4. Tag all new code

### Phase 2: FST Backend API
1. Extend existing database schema for versioning, KPIs, scripts
2. Build FastAPI application with all endpoints
3. Implement root cause analysis engine
4. Add suggestion generation logic

### Phase 3: FST React Frontend
1. Scaffold React + Vite + TypeScript project
2. Build tree visualization component
3. Build node detail panel with metrics dashboard
4. Build root cause analysis panel
5. Add version management UI
6. Add recording management UI

### Phase 4: Documentation
1. API documentation (auto-generated from FastAPI)
2. User guide for FST frontend
3. Developer setup instructions
4. Architecture documentation

---

## Progress Tracking

- [x] Repository exploration
- [x] Design document written
- [ ] Model audit: Camera (in progress - agent)
- [ ] Model audit: LiDAR (in progress - agent)
- [ ] Model audit: Radar (in progress - agent)
- [ ] Implement missing model components
- [ ] FST Backend API implementation
- [ ] FST React Frontend implementation
- [ ] Comprehensive documentation

*Last updated: 2026-06-27 (session start)*
