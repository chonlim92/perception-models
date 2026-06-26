"""
Pydantic v2 models for scenario annotation, querying, and tree serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ScenarioTag(BaseModel):
    """A single scenario tag linking a recording segment to a tree node."""

    node_id: str = Field(
        ...,
        description="ID of the ScenarioTreeNode this tag refers to (e.g., 'L4.3.1')",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this tag assignment (0.0 to 1.0)",
    )
    source: Literal["auto", "manual", "model"] = Field(
        ...,
        description="How this tag was generated: auto (rule-based), manual (human), or model (ML)",
    )

    @field_validator("node_id")
    @classmethod
    def node_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("node_id must not be empty")
        return v


class ScenarioAnnotation(BaseModel):
    """Full scenario annotation for a recording or recording segment."""

    recording_id: str = Field(
        ...,
        description="Unique identifier of the recording being annotated",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this annotation was created or last modified",
    )
    tags: list[ScenarioTag] = Field(
        default_factory=list,
        description="List of scenario tags assigned to this recording",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., annotator, tool version, segment offsets)",
    )

    @field_validator("recording_id")
    @classmethod
    def recording_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("recording_id must not be empty")
        return v


class ScenarioQuery(BaseModel):
    """Query criteria for searching recordings by scenario tags."""

    required_tags: list[str] = Field(
        default_factory=list,
        description="Node IDs that must be present in matching annotations",
    )
    excluded_tags: list[str] = Field(
        default_factory=list,
        description="Node IDs that must NOT be present in matching annotations",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for tag matching",
    )
    layer_filters: list[int] = Field(
        default_factory=list,
        description="Only consider tags from these layers (empty = all layers)",
    )

    def matches(self, annotation: ScenarioAnnotation) -> bool:
        """
        Check if an annotation matches this query.

        Args:
            annotation: The annotation to evaluate.

        Returns:
            True if the annotation satisfies all query criteria.
        """
        # Filter tags by confidence
        qualifying_tags = [
            tag for tag in annotation.tags if tag.confidence >= self.min_confidence
        ]

        # Filter by layer if layer_filters specified
        if self.layer_filters:
            # Layer is encoded in node_id as "L{layer}...." prefix
            qualifying_tags = [
                tag
                for tag in qualifying_tags
                if _extract_layer(tag.node_id) in self.layer_filters
            ]

        qualifying_node_ids = {tag.node_id for tag in qualifying_tags}

        # All required tags must be present
        if self.required_tags:
            if not all(rid in qualifying_node_ids for rid in self.required_tags):
                return False

        # No excluded tags should be present (check against ALL tags, not just qualifying)
        if self.excluded_tags:
            all_node_ids = {tag.node_id for tag in annotation.tags}
            if any(eid in all_node_ids for eid in self.excluded_tags):
                return False

        return True


def _extract_layer(node_id: str) -> int:
    """
    Extract the layer number from a node ID.

    Node IDs are formatted as 'L{layer}...' (e.g., 'L4.3.1' -> layer 4).
    Returns 0 if the format is unexpected.
    """
    if node_id.startswith("L") and len(node_id) > 1:
        try:
            # Extract digits immediately after 'L'
            digit_chars = ""
            for ch in node_id[1:]:
                if ch.isdigit():
                    digit_chars += ch
                else:
                    break
            if digit_chars:
                return int(digit_chars)
        except ValueError:
            pass
    return 0


class ScenarioTreeNodeModel(BaseModel):
    """Serializable model for a single tree node."""

    id: str
    name: str
    layer: int
    description: str = ""
    detection_method: str = ""
    parent_id: Optional[str] = None
    children: list["ScenarioTreeNodeModel"] = Field(default_factory=list)


class ScenarioTreeModel(BaseModel):
    """
    Serializable Pydantic model for the entire scenario tree.

    Supports conversion to/from dict for JSON serialization.
    """

    root: ScenarioTreeNodeModel
    version: str = Field(default="1.0.0", description="Schema version")
    description: str = Field(
        default="PEGASUS/ASAM Functional Scenario Tree",
        description="Human-readable description of this tree",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tree to a dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioTreeModel":
        """Deserialize a tree from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_tree_node(cls, node: Any) -> "ScenarioTreeModel":
        """
        Create a ScenarioTreeModel from a ScenarioTreeNode (from scenario_tree module).

        Args:
            node: A ScenarioTreeNode instance (the root).

        Returns:
            A fully serializable ScenarioTreeModel.
        """
        return cls(root=_convert_node(node))


def _convert_node(node: Any) -> ScenarioTreeNodeModel:
    """Recursively convert a ScenarioTreeNode dataclass to a Pydantic model."""
    return ScenarioTreeNodeModel(
        id=node.id,
        name=node.name,
        layer=node.layer,
        description=node.description,
        detection_method=node.detection_method,
        parent_id=node.parent_id,
        children=[_convert_node(child) for child in node.children],
    )
