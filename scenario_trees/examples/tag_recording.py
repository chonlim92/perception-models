"""
Example: Auto-tag a single recording with scenario attributes.

This example demonstrates how to use the annotation pipeline to automatically
tag a recording with scenario attributes from the 6-layer taxonomy.
"""

import numpy as np
from pathlib import Path

from scenario_trees.auto_annotation.annotation_pipeline import (
    AnnotationPipeline,
    FrameData,
)
from scenario_trees.auto_annotation.object_detector import DetectedObject, EgoState
from scenario_trees.taxonomy.scenario_tree import build_default_tree
from scenario_trees.taxonomy.tree_visualization import render_text


def create_synthetic_frame() -> FrameData:
    """Create a synthetic frame for demonstration purposes."""
    # Simulate a camera image (640x480 RGB)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add some brightness to simulate daylight
    image = np.clip(image.astype(np.int32) + 100, 0, 255).astype(np.uint8)

    # Simulate LiDAR point cloud (N x 4: x, y, z, intensity)
    num_points = 50000
    points = np.random.randn(num_points, 4).astype(np.float32)
    points[:, 0] *= 50  # x range: -50 to 50m
    points[:, 1] *= 50  # y range: -50 to 50m
    points[:, 2] = np.abs(points[:, 2]) * 3  # z range: 0 to ~9m
    points[:, 3] = np.random.uniform(0, 1, num_points)  # intensity

    # Simulate detected objects
    detections = [
        DetectedObject(
            class_name="car",
            bbox=np.array([200, 300, 280, 380]),
            track_id=1,
            velocity=np.array([10.0, 0.5, 0.0]),
            position_3d=np.array([15.0, 2.0, 0.0]),
        ),
        DetectedObject(
            class_name="car",
            bbox=np.array([400, 310, 460, 370]),
            track_id=2,
            velocity=np.array([12.0, -0.3, 0.0]),
            position_3d=np.array([25.0, -1.5, 0.0]),
        ),
        DetectedObject(
            class_name="pedestrian",
            bbox=np.array([100, 350, 130, 420]),
            track_id=3,
            velocity=np.array([0.0, 1.2, 0.0]),
            position_3d=np.array([8.0, 5.0, 0.0]),
        ),
    ]

    # Ego vehicle state
    ego_state = EgoState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([15.0, 0.0, 0.0]),
        acceleration=np.array([0.5, 0.0, 0.0]),
        heading=0.0,
        timestamp=0.0,
        lane_id=1,
    )

    return FrameData(
        image=image,
        points=points,
        detections=detections,
        ego_state=ego_state,
        timestamp=0.0,
    )


def main():
    """Demonstrate auto-tagging a recording."""
    print("=" * 70)
    print("Functional Scenario Trees - Recording Auto-Tagging Example")
    print("=" * 70)

    # 1. Show the scenario tree
    print("\n--- Scenario Taxonomy Tree (first 3 layers) ---\n")
    tree = build_default_tree()
    print(render_text(tree, max_depth=2))

    # 2. Create the annotation pipeline
    print("\n--- Initializing Annotation Pipeline ---\n")
    pipeline = AnnotationPipeline(
        use_clip=False,  # Set to True if CLIP model is available
        confidence_threshold=0.3,
    )
    print("Pipeline initialized successfully.")

    # 3. Process a synthetic frame
    print("\n--- Processing Synthetic Frame ---\n")
    frame = create_synthetic_frame()
    tags = pipeline.process_frame(frame)

    print(f"Generated {len(tags)} scenario tags:")
    for tag in tags:
        print(f"  - Node: {tag.node_id:30s} | Confidence: {tag.confidence:.3f} | Source: {tag.source}")

    # 4. Show the full annotation
    print("\n--- Full Recording Annotation ---\n")
    from scenario_trees.taxonomy.scenario_schema import ScenarioAnnotation

    annotation = ScenarioAnnotation(
        recording_id="example_recording_001",
        timestamp="2024-03-15T10:30:00",
        tags=tags,
        metadata={
            "location": "Munich, Germany",
            "vehicle": "test_car_01",
            "duration_seconds": 120.0,
        },
    )

    print(f"Recording ID: {annotation.recording_id}")
    print(f"Timestamp: {annotation.timestamp}")
    print(f"Number of tags: {len(annotation.tags)}")
    print(f"Metadata: {annotation.metadata}")

    # 5. Filter by layer
    print("\n--- Tags by Layer ---\n")
    layer_names = {
        1: "Road Topology",
        2: "Traffic Infrastructure",
        3: "Temporary Modifications",
        4: "Dynamic Objects",
        5: "Environment",
        6: "Digital Information",
    }

    for layer_num, layer_name in layer_names.items():
        layer_tags = [t for t in tags if t.node_id.startswith(f"L{layer_num}")]
        if layer_tags:
            print(f"  Layer {layer_num} ({layer_name}):")
            for tag in layer_tags:
                print(f"    - {tag.node_id}: {tag.confidence:.3f}")

    print("\n" + "=" * 70)
    print("Auto-tagging complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
