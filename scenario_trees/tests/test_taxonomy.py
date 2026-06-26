"""Tests for the taxonomy module."""

import pytest
from scenario_trees.taxonomy.scenario_tree import (
    ScenarioTreeNode,
    build_default_tree,
    get_node_by_id,
    get_nodes_by_layer,
    get_leaf_nodes,
)
from scenario_trees.taxonomy.scenario_schema import (
    ScenarioTag,
    ScenarioAnnotation,
    ScenarioQuery,
    ScenarioTreeModel,
)
from scenario_trees.taxonomy.tree_visualization import (
    render_text,
    render_html,
    render_graphviz,
)


class TestScenarioTree:
    """Test the scenario tree structure."""

    def setup_method(self):
        """Build the default tree for each test."""
        self.tree = build_default_tree()

    def test_tree_root_exists(self):
        """Root node should exist and be the top-level node."""
        assert self.tree is not None
        assert self.tree.name == "Functional Scenario Tree"
        assert self.tree.parent_id is None

    def test_tree_has_six_layers(self):
        """Tree should have children representing 6 layers."""
        assert len(self.tree.children) == 6

    def test_layer_names(self):
        """Each layer should have the correct name."""
        expected_names = [
            "Road Topology",
            "Traffic Infrastructure",
            "Temporary Modifications",
            "Dynamic Objects",
            "Environment",
            "Digital Information",
        ]
        for child, expected in zip(self.tree.children, expected_names):
            assert child.name == expected

    def test_get_node_by_id(self):
        """Should be able to find any node by its ID."""
        # Root
        root = get_node_by_id(self.tree, "root")
        assert root is not None
        assert root.name == "Functional Scenario Tree"

        # Layer 1 nodes
        highway = get_node_by_id(self.tree, "L1.highway")
        assert highway is not None
        assert highway.name == "Highway"
        assert highway.layer == 1

    def test_get_node_nonexistent(self):
        """Should return None for non-existent IDs."""
        result = get_node_by_id(self.tree, "nonexistent_id")
        assert result is None

    def test_get_nodes_by_layer(self):
        """Should return all nodes at a given layer."""
        layer1_nodes = get_nodes_by_layer(self.tree, 1)
        assert len(layer1_nodes) > 0
        for node in layer1_nodes:
            assert node.layer == 1

        layer5_nodes = get_nodes_by_layer(self.tree, 5)
        assert len(layer5_nodes) > 0
        for node in layer5_nodes:
            assert node.layer == 5

    def test_get_leaf_nodes(self):
        """Leaf nodes should have no children."""
        leaves = get_leaf_nodes(self.tree)
        assert len(leaves) > 0
        for leaf in leaves:
            assert len(leaf.children) == 0

    def test_node_has_detection_method(self):
        """Nodes should have detection method descriptions."""
        highway = get_node_by_id(self.tree, "L1.highway")
        assert highway.detection_method is not None
        assert len(highway.detection_method) > 0

    def test_layer4_has_behaviors(self):
        """Layer 4 should include behavioral scenarios."""
        layer4_nodes = get_nodes_by_layer(self.tree, 4)
        node_ids = [n.id for n in layer4_nodes]
        # Check some expected behavior nodes exist
        assert any("cut_in" in nid for nid in node_ids)

    def test_layer5_has_weather(self):
        """Layer 5 should include weather conditions."""
        layer5_nodes = get_nodes_by_layer(self.tree, 5)
        node_ids = [n.id for n in layer5_nodes]
        assert any("rain" in nid for nid in node_ids)
        assert any("fog" in nid for nid in node_ids)
        assert any("snow" in nid for nid in node_ids)


class TestScenarioSchema:
    """Test Pydantic schema models."""

    def test_scenario_tag_creation(self):
        """Should create a valid ScenarioTag."""
        tag = ScenarioTag(
            node_id="L1.highway",
            confidence=0.95,
            source="auto",
        )
        assert tag.node_id == "L1.highway"
        assert tag.confidence == 0.95
        assert tag.source == "auto"

    def test_scenario_tag_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        tag = ScenarioTag(node_id="L1.highway", confidence=0.5, source="auto")
        assert 0 <= tag.confidence <= 1

    def test_scenario_annotation(self):
        """Should create a valid ScenarioAnnotation."""
        tags = [
            ScenarioTag(node_id="L1.highway", confidence=0.9, source="auto"),
            ScenarioTag(node_id="L5.weather.rain", confidence=0.7, source="auto"),
        ]
        annotation = ScenarioAnnotation(
            recording_id="test_001",
            timestamp="2024-01-15T10:00:00",
            tags=tags,
            metadata={"location": "Munich"},
        )
        assert annotation.recording_id == "test_001"
        assert len(annotation.tags) == 2
        assert annotation.metadata["location"] == "Munich"

    def test_scenario_query(self):
        """Should create a valid ScenarioQuery."""
        query = ScenarioQuery(
            required_tags=["L1.highway", "L5.weather.rain"],
            excluded_tags=["L5.lighting.night"],
            min_confidence=0.6,
        )
        assert len(query.required_tags) == 2
        assert len(query.excluded_tags) == 1
        assert query.min_confidence == 0.6

    def test_annotation_serialization(self):
        """Should serialize and deserialize annotations."""
        tags = [
            ScenarioTag(node_id="L1.urban", confidence=0.85, source="manual"),
        ]
        annotation = ScenarioAnnotation(
            recording_id="test_002",
            timestamp="2024-02-20T14:30:00",
            tags=tags,
            metadata={},
        )
        # Serialize to dict
        data = annotation.model_dump()
        assert data["recording_id"] == "test_002"
        assert len(data["tags"]) == 1

        # Deserialize back
        restored = ScenarioAnnotation.model_validate(data)
        assert restored.recording_id == annotation.recording_id
        assert restored.tags[0].node_id == "L1.urban"

    def test_scenario_tree_model_serialization(self):
        """Should serialize the tree to a model and back."""
        tree = build_default_tree()
        model = ScenarioTreeModel.from_tree_node(tree)
        assert model.root.id == "root"
        assert len(model.root.children) == 6

        # Convert to dict
        data = model.model_dump()
        assert "root" in data
        assert "children" in data["root"]

        # Restore
        restored = ScenarioTreeModel.model_validate(data)
        assert restored.root.id == "root"
        assert len(restored.root.children) == 6


class TestTreeVisualization:
    """Test tree visualization outputs."""

    def setup_method(self):
        """Build the default tree."""
        self.tree = build_default_tree()

    def test_render_text(self):
        """Text rendering should produce readable output."""
        text = render_text(self.tree, max_depth=2)
        assert len(text) > 0
        assert "Functional Scenario Tree" in text
        assert "Road Topology" in text

    def test_render_text_with_depth_limit(self):
        """Should respect max_depth parameter."""
        shallow = render_text(self.tree, max_depth=1)
        deep = render_text(self.tree, max_depth=3)
        # Deeper rendering should have more content
        assert len(deep) > len(shallow)

    def test_render_html(self):
        """HTML rendering should produce valid HTML structure."""
        html = render_html(self.tree)
        assert "<ul>" in html
        assert "<li>" in html
        assert "Road Topology" in html

    def test_render_graphviz(self):
        """Graphviz rendering should produce DOT format."""
        dot = render_graphviz(self.tree)
        assert "digraph" in dot
        assert "->" in dot
        assert "root" in dot


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
