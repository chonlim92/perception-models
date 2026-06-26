#!/usr/bin/env python3
"""
prepare_data.py - Prepare nuScenes dataset for DETR3D training.

Parses the nuScenes database using nuscenes-devkit and creates info files (pkl)
containing camera calibration, 3D annotations, ego pose, and metadata needed
for efficient data loading during training.

Usage:
    python scripts/prepare_data.py \
        --data-root ./data/nuscenes \
        --version v1.0-trainval \
        --output-dir ./data/nuscenes/infos
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.splits import create_splits_scenes
except ImportError:
    print("ERROR: nuscenes-devkit is required. Install with:")
    print("  pip install nuscenes-devkit")
    sys.exit(1)


# ============================================================================
# Constants
# ============================================================================

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# nuScenes detection class mapping (10 classes for DETR3D)
CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Mapping from nuScenes category names to our class names
CATEGORY_TO_CLASS = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}


# ============================================================================
# Data Processing Functions
# ============================================================================


def get_ego_pose(nusc: NuScenes, sample_data_token: str) -> Dict[str, Any]:
    """Get ego vehicle pose for a given sample data token."""
    sample_data = nusc.get("sample_data", sample_data_token)
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    return {
        "translation": np.array(ego_pose["translation"], dtype=np.float64),
        "rotation": np.array(ego_pose["rotation"], dtype=np.float64),
        "token": ego_pose["token"],
        "timestamp": ego_pose["timestamp"],
    }


def get_camera_calibration(
    nusc: NuScenes, sample_data_token: str
) -> Dict[str, Any]:
    """Get camera intrinsic and extrinsic calibration parameters."""
    sample_data = nusc.get("sample_data", sample_data_token)
    calibrated_sensor = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )

    # Intrinsic camera matrix (3x3)
    intrinsic = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float64)

    # Extrinsic: sensor-to-ego transformation
    translation = np.array(calibrated_sensor["translation"], dtype=np.float64)
    rotation = Quaternion(calibrated_sensor["rotation"])

    # Build 4x4 sensor-to-ego transformation matrix
    sensor_to_ego = np.eye(4, dtype=np.float64)
    sensor_to_ego[:3, :3] = rotation.rotation_matrix
    sensor_to_ego[:3, 3] = translation

    return {
        "intrinsic": intrinsic,
        "extrinsic": sensor_to_ego,
        "translation": translation,
        "rotation": np.array(calibrated_sensor["rotation"], dtype=np.float64),
        "token": calibrated_sensor["token"],
    }


def get_camera_info(
    nusc: NuScenes, sample: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Get calibration and file path information for all 6 cameras."""
    camera_info = {}

    for cam_name in CAMERA_NAMES:
        cam_token = sample["data"][cam_name]
        cam_data = nusc.get("sample_data", cam_token)

        calibration = get_camera_calibration(nusc, cam_token)
        ego_pose = get_ego_pose(nusc, cam_token)

        # Compute sensor-to-global transformation
        ego_to_global = np.eye(4, dtype=np.float64)
        ego_rotation = Quaternion(ego_pose["rotation"])
        ego_to_global[:3, :3] = ego_rotation.rotation_matrix
        ego_to_global[:3, 3] = ego_pose["translation"]

        sensor_to_global = ego_to_global @ calibration["extrinsic"]

        # Compute global-to-sensor (for projecting world points to camera)
        global_to_sensor = np.linalg.inv(sensor_to_global)

        # Compute lidar-to-camera transform (using ego as intermediate)
        # viewMatrix = intrinsic @ global_to_sensor[:3, :]
        viewmatrix = calibration["intrinsic"] @ global_to_sensor[:3, :]

        camera_info[cam_name] = {
            "data_path": cam_data["filename"],
            "timestamp": cam_data["timestamp"],
            "intrinsic": calibration["intrinsic"],
            "sensor_to_ego": calibration["extrinsic"],
            "ego_to_global": ego_to_global,
            "sensor_to_global": sensor_to_global,
            "global_to_sensor": global_to_sensor,
            "viewmatrix": viewmatrix,
            "width": cam_data["width"],
            "height": cam_data["height"],
        }

    return camera_info


def get_annotations(
    nusc: NuScenes,
    sample: Dict[str, Any],
    ego_pose: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get 3D bounding box annotations for a sample.

    Returns list of annotation dicts with box parameters in global frame,
    class labels, velocities, and visibility information.
    """
    annotations = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)

        # Filter by category
        category = ann["category_name"]
        class_name = None
        for cat_prefix, cls_name in CATEGORY_TO_CLASS.items():
            if category.startswith(cat_prefix):
                class_name = cls_name
                break

        if class_name is None:
            continue

        class_id = CLASS_NAMES.index(class_name)

        # 3D bounding box in global coordinates
        # center: [x, y, z], size: [w, l, h] (width, length, height)
        center = np.array(ann["translation"], dtype=np.float64)
        size = np.array(ann["size"], dtype=np.float64)  # [w, l, h]
        rotation = Quaternion(ann["rotation"])

        # Compute velocity (in global frame)
        velocity = nusc.box_velocity(ann_token)
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3, dtype=np.float64)

        # Convert rotation to yaw angle (around z-axis)
        # nuScenes uses quaternion, we extract yaw for sin/cos encoding
        yaw = rotation.yaw_pitch_roll[0]

        # Build the 10-dimensional code vector:
        # [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
        bbox_code = np.array(
            [
                center[0],
                center[1],
                center[2],
                size[0],  # width
                size[1],  # length
                size[2],  # height
                np.sin(yaw),
                np.cos(yaw),
                velocity[0],
                velocity[1],
            ],
            dtype=np.float64,
        )

        # Visibility
        visibility_token = ann["visibility_token"]
        visibility = int(nusc.get("visibility", visibility_token)["level"])

        # Number of lidar/radar points
        num_lidar_pts = ann["num_lidar_pts"]
        num_radar_pts = ann["num_radar_pts"]

        annotations.append(
            {
                "token": ann_token,
                "instance_token": ann["instance_token"],
                "class_name": class_name,
                "class_id": class_id,
                "center": center,
                "size": size,
                "rotation": np.array(ann["rotation"], dtype=np.float64),
                "yaw": yaw,
                "velocity": velocity[:2],  # only x, y
                "bbox_code": bbox_code,
                "visibility": visibility,
                "num_lidar_pts": num_lidar_pts,
                "num_radar_pts": num_radar_pts,
            }
        )

    return annotations


def normalize_annotations_to_ego(
    annotations: List[Dict[str, Any]],
    ego_translation: np.ndarray,
    ego_rotation: np.ndarray,
) -> List[Dict[str, Any]]:
    """Transform annotations from global frame to ego vehicle frame."""
    ego_rot = Quaternion(ego_rotation)
    ego_rot_inv = ego_rot.inverse

    normalized = []
    for ann in annotations:
        # Transform center to ego frame
        center_global = ann["center"]
        center_ego = ego_rot_inv.rotate(center_global - ego_translation)

        # Transform velocity to ego frame
        vel_global = np.array([ann["velocity"][0], ann["velocity"][1], 0.0])
        vel_ego = ego_rot_inv.rotate(vel_global)

        # Transform rotation to ego frame
        box_rot = Quaternion(ann["rotation"])
        box_rot_ego = ego_rot_inv * box_rot
        yaw_ego = box_rot_ego.yaw_pitch_roll[0]

        # Rebuild bbox code in ego frame
        bbox_code_ego = np.array(
            [
                center_ego[0],
                center_ego[1],
                center_ego[2],
                ann["size"][0],
                ann["size"][1],
                ann["size"][2],
                np.sin(yaw_ego),
                np.cos(yaw_ego),
                vel_ego[0],
                vel_ego[1],
            ],
            dtype=np.float64,
        )

        ann_ego = ann.copy()
        ann_ego["center_ego"] = center_ego
        ann_ego["yaw_ego"] = yaw_ego
        ann_ego["velocity_ego"] = vel_ego[:2]
        ann_ego["bbox_code_ego"] = bbox_code_ego
        normalized.append(ann_ego)

    return normalized


def process_sample(
    nusc: NuScenes,
    sample: Dict[str, Any],
    data_root: str,
) -> Dict[str, Any]:
    """Process a single nuScenes sample into the info dict format.

    Returns a dict containing all information needed for training:
    - Sample metadata (token, timestamp, scene)
    - Camera info (paths, calibration for all 6 cameras)
    - Ego pose
    - 3D annotations in both global and ego frames
    """
    # Get lidar sample data for reference timestamp and ego pose
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    # Ego pose at lidar timestamp (reference frame)
    ego_pose = get_ego_pose(nusc, lidar_token)

    # Camera information
    camera_info = get_camera_info(nusc, sample)

    # 3D annotations
    annotations = get_annotations(nusc, sample, ego_pose)

    # Normalize annotations to ego frame
    annotations_ego = normalize_annotations_to_ego(
        annotations, ego_pose["translation"], ego_pose["rotation"]
    )

    # Scene information
    scene = nusc.get("scene", sample["scene_token"])

    info = {
        "token": sample["token"],
        "timestamp": sample["timestamp"],
        "scene_token": sample["scene_token"],
        "scene_name": scene["name"],
        "lidar_path": lidar_data["filename"],
        "ego_pose": {
            "translation": ego_pose["translation"],
            "rotation": ego_pose["rotation"],
        },
        "cameras": camera_info,
        "annotations": annotations_ego,
        "num_annotations": len(annotations_ego),
        "prev_token": sample["prev"],
        "next_token": sample["next"],
    }

    return info


def compute_dataset_statistics(
    infos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute dataset statistics for normalization and analysis.

    Returns statistics including:
    - Class distribution (counts per class)
    - Box size statistics (mean, std for each class)
    - Velocity statistics
    - Spatial distribution of objects
    """
    class_counts = defaultdict(int)
    class_sizes = defaultdict(list)  # {class_name: [[w, l, h], ...]}
    class_velocities = defaultdict(list)
    all_centers = []
    total_annotations = 0

    for info in infos:
        for ann in info["annotations"]:
            class_name = ann["class_name"]
            class_counts[class_name] += 1
            class_sizes[class_name].append(ann["size"])
            class_velocities[class_name].append(ann["velocity_ego"])
            all_centers.append(ann["center_ego"])
            total_annotations += 1

    # Compute per-class size statistics
    size_stats = {}
    for class_name in CLASS_NAMES:
        if class_sizes[class_name]:
            sizes = np.array(class_sizes[class_name])
            size_stats[class_name] = {
                "mean": sizes.mean(axis=0).tolist(),
                "std": sizes.std(axis=0).tolist(),
                "min": sizes.min(axis=0).tolist(),
                "max": sizes.max(axis=0).tolist(),
                "count": len(sizes),
            }
        else:
            size_stats[class_name] = {
                "mean": [0, 0, 0],
                "std": [0, 0, 0],
                "min": [0, 0, 0],
                "max": [0, 0, 0],
                "count": 0,
            }

    # Compute velocity statistics
    velocity_stats = {}
    for class_name in CLASS_NAMES:
        if class_velocities[class_name]:
            vels = np.array(class_velocities[class_name])
            speeds = np.linalg.norm(vels, axis=1)
            velocity_stats[class_name] = {
                "mean_speed": float(speeds.mean()),
                "max_speed": float(speeds.max()),
                "mean_vx": float(vels[:, 0].mean()),
                "mean_vy": float(vels[:, 1].mean()),
            }
        else:
            velocity_stats[class_name] = {
                "mean_speed": 0.0,
                "max_speed": 0.0,
                "mean_vx": 0.0,
                "mean_vy": 0.0,
            }

    # Spatial statistics
    if all_centers:
        all_centers = np.array(all_centers)
        spatial_stats = {
            "center_mean": all_centers.mean(axis=0).tolist(),
            "center_std": all_centers.std(axis=0).tolist(),
            "center_min": all_centers.min(axis=0).tolist(),
            "center_max": all_centers.max(axis=0).tolist(),
        }
    else:
        spatial_stats = {
            "center_mean": [0, 0, 0],
            "center_std": [0, 0, 0],
            "center_min": [0, 0, 0],
            "center_max": [0, 0, 0],
        }

    return {
        "total_samples": len(infos),
        "total_annotations": total_annotations,
        "class_counts": dict(class_counts),
        "class_distribution": {
            k: v / max(total_annotations, 1)
            for k, v in class_counts.items()
        },
        "size_stats": size_stats,
        "velocity_stats": velocity_stats,
        "spatial_stats": spatial_stats,
    }


# ============================================================================
# Split Generation
# ============================================================================


def get_split_scenes(version: str) -> Dict[str, List[str]]:
    """Get train/val scene splits from nuScenes devkit."""
    splits = create_splits_scenes()

    if version == "v1.0-mini":
        return {
            "train": splits["mini_train"],
            "val": splits["mini_val"],
        }
    elif version == "v1.0-trainval":
        return {
            "train": splits["train"],
            "val": splits["val"],
        }
    elif version == "v1.0-test":
        return {
            "test": splits["test"],
        }
    else:
        raise ValueError(f"Unknown version: {version}")


def split_infos_by_scene(
    infos: List[Dict[str, Any]],
    split_scenes: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Split info list according to scene-based train/val splits."""
    split_infos = {split: [] for split in split_scenes}

    for info in infos:
        scene_name = info["scene_name"]
        for split_name, scenes in split_scenes.items():
            if scene_name in scenes:
                split_infos[split_name].append(info)
                break

    return split_infos


# ============================================================================
# Main Processing Pipeline
# ============================================================================


def process_nuscenes(
    data_root: str,
    version: str,
    output_dir: str,
    max_samples: Optional[int] = None,
) -> None:
    """Main processing pipeline for nuScenes dataset.

    Args:
        data_root: Path to nuScenes data root directory.
        version: Dataset version (e.g., 'v1.0-trainval', 'v1.0-mini').
        output_dir: Directory to save processed info pickle files.
        max_samples: If set, process only this many samples (for debugging).
    """
    print(f"Loading nuScenes {version} from {data_root}...")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    print(f"Dataset loaded: {len(nusc.sample)} samples, {len(nusc.scene)} scenes")

    # Process all samples
    print("Processing samples...")
    all_infos = []
    num_samples = len(nusc.sample)

    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    for idx in range(num_samples):
        sample = nusc.sample[idx]

        if idx % 100 == 0:
            print(f"  Processing sample {idx + 1}/{num_samples}...")

        info = process_sample(nusc, sample, data_root)
        all_infos.append(info)

    print(f"Processed {len(all_infos)} samples total")

    # Get train/val splits
    print("Generating train/val splits...")
    split_scenes = get_split_scenes(version)
    split_infos = split_infos_by_scene(all_infos, split_scenes)

    for split_name, infos in split_infos.items():
        print(f"  {split_name}: {len(infos)} samples")

    # Compute statistics
    print("Computing dataset statistics...")
    stats = {}
    for split_name, infos in split_infos.items():
        if infos:
            stats[split_name] = compute_dataset_statistics(infos)

    # Print summary statistics
    for split_name, split_stats in stats.items():
        print(f"\n  [{split_name}] Statistics:")
        print(f"    Samples: {split_stats['total_samples']}")
        print(f"    Annotations: {split_stats['total_annotations']}")
        print(f"    Class distribution:")
        for cls_name in CLASS_NAMES:
            count = split_stats["class_counts"].get(cls_name, 0)
            pct = split_stats["class_distribution"].get(cls_name, 0) * 100
            print(f"      {cls_name:25s}: {count:6d} ({pct:.1f}%)")

    # Save processed data
    print(f"\nSaving processed data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Save per-split info files
    for split_name, infos in split_infos.items():
        if not infos:
            continue

        output_path = os.path.join(
            output_dir, f"detr3d_infos_{split_name}.pkl"
        )
        with open(output_path, "wb") as f:
            pickle.dump(infos, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved: {output_path} ({len(infos)} samples)")

    # Save all infos combined
    all_output_path = os.path.join(output_dir, "detr3d_infos_all.pkl")
    with open(all_output_path, "wb") as f:
        pickle.dump(all_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {all_output_path} ({len(all_infos)} samples)")

    # Save statistics
    stats_path = os.path.join(output_dir, "dataset_statistics.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {stats_path}")

    # Also save statistics as readable text
    stats_txt_path = os.path.join(output_dir, "dataset_statistics.txt")
    with open(stats_txt_path, "w") as f:
        f.write("DETR3D Dataset Statistics\n")
        f.write("=" * 60 + "\n\n")
        for split_name, split_stats in stats.items():
            f.write(f"Split: {split_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total samples: {split_stats['total_samples']}\n")
            f.write(f"  Total annotations: {split_stats['total_annotations']}\n")
            f.write(f"\n  Class counts:\n")
            for cls_name in CLASS_NAMES:
                count = split_stats["class_counts"].get(cls_name, 0)
                f.write(f"    {cls_name:25s}: {count}\n")
            f.write(f"\n  Box size statistics (mean [w, l, h]):\n")
            for cls_name in CLASS_NAMES:
                mean = split_stats["size_stats"][cls_name]["mean"]
                f.write(
                    f"    {cls_name:25s}: "
                    f"[{mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}]\n"
                )
            f.write("\n")
    print(f"  Saved: {stats_txt_path}")

    print("\nData preparation complete!")


# ============================================================================
# Entry Point
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare nuScenes dataset for DETR3D training"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to nuScenes data root directory",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        choices=["v1.0-trainval", "v1.0-mini", "v1.0-test"],
        help="nuScenes dataset version (default: v1.0-trainval)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed files (default: <data-root>/infos)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only N samples (for debugging)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        print(f"ERROR: Data root directory does not exist: {data_root}")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(data_root, "infos")
    output_dir = os.path.abspath(output_dir)

    process_nuscenes(
        data_root=data_root,
        version=args.version,
        output_dir=output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
