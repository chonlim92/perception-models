# FST - Functional Scenario Tree Frontend

Interactive React application for visualizing, managing, and analyzing the Functional Scenario Tree (FST) used in autonomous driving perception validation.

## Features

- **Interactive Tree Visualization**: Hierarchical tree with expand/collapse, color-coded KPI status, pan/zoom
- **Per-Node Metrics Dashboard**: Aggregated metrics, KPI threshold monitoring, trend charts
- **Tree Versioning**: Immutable snapshots, version history, rollback capability
- **Recording Management**: Attach measurements/recordings to nodes, bulk import
- **Semi-Automated Root Cause Analysis**: Pattern mining in failures, node splitting suggestions
- **Human Approval Workflow**: Review and approve/reject suggested tree modifications

## Architecture

```
Frontend (React + TypeScript + Vite)
  ├── Tree Visualization (ReactFlow)
  ├── Metrics Dashboard (Recharts)
  ├── State Management (Zustand)
  └── API Client (fetch + React Query)
         │
         ▼ REST API (/api/*)
Backend (FastAPI - Python)
  ├── Tree CRUD + Versioning
  ├── Metrics & Evaluation Engine
  ├── Root Cause Analysis Engine
  └── SQLite Database
```

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- pip

### Backend Setup

```bash
cd scenario_trees
pip install fastapi uvicorn sqlalchemy pydantic pyyaml

# Start the API server
uvicorn scenario_trees.api.app:app --reload --port 8000
```

### Frontend Setup

```bash
cd fst-frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:3000` and proxies API requests to `http://localhost:8000`.

## Project Structure

```
fst-frontend/
├── src/
│   ├── api/
│   │   └── client.ts          # API client with all endpoints
│   ├── components/
│   │   ├── App.tsx            # Main app layout
│   │   ├── Header.tsx         # Top bar with version info
│   │   ├── TreeVisualization.tsx  # ReactFlow tree canvas
│   │   ├── ScenarioNode.tsx   # Custom node component
│   │   ├── NodeDetailPanel.tsx    # Right panel for selected node
│   │   ├── MetricsDashboard.tsx   # Charts and KPI cards
│   │   ├── SuggestionsPanel.tsx   # Root cause suggestions
│   │   ├── RecordingsPanel.tsx    # Attached recordings list
│   │   └── VersionPanel.tsx   # Version timeline at bottom
│   ├── store/
│   │   └── useTreeStore.ts    # Zustand state management
│   ├── types/
│   │   └── index.ts           # TypeScript interfaces
│   ├── main.tsx               # App entry point
│   └── index.css              # Tailwind CSS imports
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

## API Endpoints

### Tree Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tree` | Get current tree |
| GET | `/api/tree/versions` | List all versions |
| GET | `/api/tree/versions/:id` | Get specific version |
| POST | `/api/tree/versions` | Create new version |
| PUT | `/api/tree/nodes/:nodeId` | Update node |
| POST | `/api/tree/nodes/:parentId/children` | Add child node |
| DELETE | `/api/tree/nodes/:nodeId` | Remove node |
| POST | `/api/tree/nodes/:nodeId/split` | Split node |

### Recordings
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/nodes/:nodeId/recordings` | Get attached recordings |
| POST | `/api/nodes/:nodeId/recordings` | Attach recording |
| DELETE | `/api/nodes/:nodeId/recordings/:recId` | Detach recording |
| POST | `/api/recordings/bulk-import` | Bulk import |

### Metrics & Evaluation
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/nodes/:nodeId/metrics` | Get metrics summary |
| POST | `/api/nodes/:nodeId/evaluate` | Run evaluation scripts |
| POST | `/api/nodes/:nodeId/results` | Submit results directly |
| GET | `/api/nodes/:nodeId/kpi` | Get KPI config |
| PUT | `/api/nodes/:nodeId/kpi` | Set KPI threshold |

### Root Cause Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/nodes/:nodeId/analyze` | Trigger analysis |
| GET | `/api/nodes/:nodeId/suggestions` | Get suggestions |
| POST | `/api/suggestions/:id/approve` | Approve suggestion |
| POST | `/api/suggestions/:id/reject` | Reject suggestion |

## Root Cause Analysis Flow

When a node fails its KPI threshold:

1. **Detection**: System detects metrics below threshold
2. **Analysis**: Click "Run Analysis" to mine patterns in failing recordings
3. **Pattern Discovery**: Engine identifies attributes (e.g., "has_bicycle") that are statistically overrepresented in failures
4. **Suggestion**: System proposes corrective actions (split node, investigate, adjust threshold)
5. **Review**: Developer reviews evidence and approves/rejects
6. **Application**: Approved splits create new child nodes and reassign recordings

### Example Scenario

A node `L4.vehicle` has 100 recordings with mAP KPI threshold of 0.7. After evaluation:
- 30 recordings fail (mAP < 0.7)
- Analysis finds: 25/30 failures contain bicycles (lift=3.2x)
- System suggests: Split into `L4.vehicle.with_bicycle` and `L4.vehicle.without_bicycle`
- Developer approves → tree is updated, recordings reassigned, new version created

## Configuration

### KPI Thresholds
Set via API:
```json
PUT /api/nodes/L4.vehicle/kpi
{
  "metric_name": "mAP",
  "threshold": 0.7,
  "direction": "above",
  "warning_margin": 0.05
}
```

### Evaluation Scripts
Attach Python scripts to nodes:
```json
POST /api/nodes/L4.vehicle/scripts
{
  "name": "compute_map",
  "script_content": "import numpy as np\n..."
}
```

## Development

```bash
# Type checking
npx tsc --noEmit

# Build for production
npm run build

# Preview production build
npm run preview
```

## Technology Stack

- **React 18** with TypeScript
- **Vite 5** for fast HMR and builds
- **ReactFlow 11** for interactive node graphs
- **Recharts 2** for metric visualizations
- **Zustand 4** for lightweight state management
- **React Query 5** for server state caching
- **Tailwind CSS 3** for styling
- **FastAPI** (Python backend)
- **SQLite** for persistence
