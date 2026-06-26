"""
Scenario Manager - Query, split generation, and export interface.

Provides tooling for managing scenario-based autonomous driving datasets:
- Database storage of recording metadata and scenario tags
- Query engine for complex scenario-based searches
- Balanced split generation (train/val/test)
- Dataset export for model training
- Text-based dashboard for dataset statistics
"""

from .database import ScenarioDatabase
from .query_engine import ScenarioQueryEngine
from .split_generator import SplitGenerator
from .dashboard import ScenarioDashboard
from .export import ScenarioExporter

__all__ = [
    "ScenarioDatabase",
    "ScenarioQueryEngine",
    "SplitGenerator",
    "ScenarioDashboard",
    "ScenarioExporter",
]
