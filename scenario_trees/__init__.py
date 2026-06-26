"""
Functional Scenario Trees for Autonomous Driving Data Management.

This package provides automated annotation, scenario tagging, data mining,
and scenario-based test management for autonomous driving recordings.

Based on the PEGASUS/ASAM 6-layer scenario taxonomy:
- Layer 1: Road Topology
- Layer 2: Traffic Infrastructure
- Layer 3: Temporary Modifications
- Layer 4: Dynamic Objects
- Layer 5: Environment
- Layer 6: Digital Information

Modules:
    taxonomy: Scenario classification hierarchy and tree definitions
    auto_annotation: Automated scenario tagging from sensor data
    data_mining: Corner case discovery and coverage analysis
    scenario_manager: Query, split generation, and export interface
"""

__version__ = "1.0.0"

from .taxonomy.scenario_tree import build_default_tree, get_node_by_id, get_nodes_by_layer
from .taxonomy.scenario_schema import ScenarioTag, ScenarioAnnotation, ScenarioQuery

__all__ = [
    "build_default_tree",
    "get_node_by_id",
    "get_nodes_by_layer",
    "ScenarioTag",
    "ScenarioAnnotation",
    "ScenarioQuery",
]
