"""
Prepare nuScenes data for PETR/StreamPETR training.

This script parses the nuScenes database and generates training info files (pickle)
containing all necessary metadata for the data pipeline:
  - Image paths for all 6 cameras
  - Camera intrinsics and extrinsics (camera-to-world)
  - 3D annotations (bounding boxes, categories, velocities)
  - Ego-motion matrices for temporal modeling
  - Temporal frame linkage (previous/next sample tokens)

Usage:
    python prepare_data.py \
        --data_root /data/nuscenes \
        --version v1.0-trainval \
        --output_dir ./data/infos
"""

import os
import sys
import argparse
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict


CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

NUSCENES_CLASSES = {
    "car": 0,
    "truck": 1,
    "construction_vehicle": 2,
    "bus": 3,
    "trailer": 4,
    "barrier": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "pedestrian": 8,
    "traffic_cone": 9,
}

DETECTION_NAMES = list(NUSCENES_CLASSES.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare nuScenes data for PETR training")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of nuScenes dataset")
    parser.add_argument("--version", type=str, default="v1.0-trainval",
                        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
                        help="nuScenes dataset version")
    parser.add_argument("--output_dir", type=str, default="./data/infos",
                        help="Output directory for info files")
    parser.add_argument("--max_sweeps", type=int, default=10,
                        help="Maximum number of sweeps (for temporal)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    return parser.parse_args()


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Args:
        quaternion: (4,) array [w, x, y, z]

    Returns:
        rotation: (3, 3) rotation matrix
    """
    w, x, y, z = quaternion
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0

    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z

    rotation = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float32)

    return rotation


def get_sensor_transform(
    nusc,
    sample_data_token: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get sensor-to-ego and ego-to-global transforms for a sample_data entry.

    Returns:
        sensor_to_ego: (4, 4) transformation matrix
        ego_to_global: (4, 4) transformation matrix
    """
    sample_data = nusc.get("sample_data", sample_data_token)

    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    sensor_to_ego = np.eye(4, dtype=np.float32)
    sensor_to_ego[:3, :3] = quaternion_to_rotation_matrix(
        np.array(calibrated_sensor["rotation"])
    )
    sensor_to_ego[:3, 3] = np.array(calibrated_sensor["translation"])

    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    ego_to_global = np.eye(4, dtype=np.float32)
    ego_to_global[:3, :3] = quaternion_to_rotation_matrix(
        np.array(ego_pose["rotation"])
    )
    ego_to_global[:3, 3] = np.array(ego_pose["translation"])

    return sensor_to_ego, ego_to_global


def get_camera_intrinsics(nusc, sample_data_token: str) -> np.ndarray:
    """
    Get 3x3 camera intrinsic matrix.

    Returns:
        intrinsics: (3, 3) camera intrinsic matrix
    """
    sample_data = nusc.get("sample_data", sample_data_token)
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    intrinsics = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
    return intrinsics


def get_annotation_info(
    nusc,
    sample_token: str,
    ego_to_global: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get 3D annotations for a sample in the ego frame.

    Args:
        nusc: NuScenes instance
        sample_token: sample token
        ego_to_global: (4, 4) ego-to-global transform of the reference sensor

    Returns:
        gt_labels: (N,) class indices
        gt_bboxes: (N, 10) [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
    """
    sample = nusc.get("sample", sample_token)
    annotations = sample["anns"]

    global_to_ego = np.linalg.inv(ego_to_global)

    gt_labels = []
    gt_bboxes = []

    for ann_token in annotations:
        ann = nusc.get("sample_annotation", ann_token)

        category = ann["category_name"]
        mapped_class = _map_category_to_class(category)
        if mapped_class is None:
            continue

        center_global = np.array(ann["translation"], dtype=np.float32)
        size = np.array(ann["size"], dtype=np.float32)
        rotation_quat = np.array(ann["rotation"], dtype=np.float32)

        center_homo = np.array([*center_global, 1.0], dtype=np.float32)
        center_ego = (global_to_ego @ center_homo)[:3]

        rot_matrix_global = quaternion_to_rotation_matrix(rotation_quat)
        rot_matrix_ego = global_to_ego[:3, :3] @ rot_matrix_global
        yaw = np.arctan2(rot_matrix_ego[1, 0], rot_matrix_ego[0, 0])

        velocity = nusc.box_velocity(ann_token)
        if np.any(np.isnan(velocity)):
            velocity = np.array([0.0, 0.0, 0.0])
        velocity_ego = (global_to_ego[:3, :3] @ velocity)[:2]

        bbox = np.array([
            center_ego[0], center_ego[1], center_ego[2],
            size[0], size[1], size[2],
            np.sin(yaw), np.cos(yaw),
            velocity_ego[0], velocity_ego[1],
        ], dtype=np.float32)

        gt_labels.append(NUSCENES_CLASSES[mapped_class])
        gt_bboxes.append(bbox)

    if len(gt_labels) == 0:
        return np.array([], dtype=np.int32), np.zeros((0, 10), dtype=np.float32)

    return np.array(gt_labels, dtype=np.int32), np.array(gt_bboxes, dtype=np.float32)


def _map_category_to_class(category_name: str) -> Optional[str]:
    """Map nuScenes category name to one of the 10 detection classes."""
    category_map = {
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
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
    }

    for prefix, cls_name in category_map.items():
        if category_name.startswith(prefix):
            return cls_name
    return None


def compute_ego_motion(
    nusc,
    current_sample_token: str,
    prev_sample_token: str,
) -> np.ndarray:
    """
    Compute the ego-motion transformation from previous to current frame.

    Returns:
        ego_motion: (4, 4) transformation matrix that maps points from
                    the previous ego frame to the current ego frame
    """
    current_sample = nusc.get("sample", current_sample_token)
    prev_sample = nusc.get("sample", prev_sample_token)

    current_sd_token = current_sample["data"]["CAM_FRONT"]
    prev_sd_token = prev_sample["data"]["CAM_FRONT"]

    current_sd = nusc.get("sample_data", current_sd_token)
    prev_sd = nusc.get("sample_data", prev_sd_token)

    current_ego_pose = nusc.get("ego_pose", current_sd["ego_pose_token"])
    prev_ego_pose = nusc.get("ego_pose", prev_sd["ego_pose_token"])

    current_ego_to_global = np.eye(4, dtype=np.float32)
    current_ego_to_global[:3, :3] = quaternion_to_rotation_matrix(
        np.array(current_ego_pose["rotation"])
    )
    current_ego_to_global[:3, 3] = np.array(current_ego_pose["translation"])

    prev_ego_to_global = np.eye(4, dtype=np.float32)
    prev_ego_to_global[:3, :3] = quaternion_to_rotation_matrix(
        np.array(prev_ego_pose["rotation"])
    )
    prev_ego_to_global[:3, 3] = np.array(prev_ego_pose["translation"])

    global_to_current_ego = np.linalg.inv(current_ego_to_global)
    ego_motion = global_to_current_ego @ prev_ego_to_global

    return ego_motion


def process_sample(
    nusc,
    sample_token: str,
    data_root: str,
) -> Dict:
    """
    Process a single sample and extract all relevant info.

    Returns:
        info dict with image paths, camera params, annotations, and temporal links.
    """
    sample = nusc.get("sample", sample_token)

    img_paths = []
    intrinsics_list = []
    extrinsics_list = []

    ref_sd_token = sample["data"]["CAM_FRONT"]
    _, ref_ego_to_global = get_sensor_transform(nusc, ref_sd_token)

    for cam_name in CAMERA_NAMES:
        sd_token = sample["data"][cam_name]
        sample_data = nusc.get("sample_data", sd_token)

        img_path = sample_data["filename"]
        img_paths.append(img_path)

        intrinsics = get_camera_intrinsics(nusc, sd_token)
        intrinsics_list.append(intrinsics)

        sensor_to_ego, ego_to_global = get_sensor_transform(nusc, sd_token)
        camera_to_world = ego_to_global @ sensor_to_ego
        extrinsics_list.append(camera_to_world)

    gt_labels, gt_bboxes = get_annotation_info(nusc, sample_token, ref_ego_to_global)

    info = {
        "token": sample_token,
        "timestamp": sample["timestamp"],
        "img_paths": img_paths,
        "intrinsics": np.array(intrinsics_list, dtype=np.float32),
        "extrinsics": np.array(extrinsics_list, dtype=np.float32),
        "gt_labels": gt_labels,
        "gt_bboxes": gt_bboxes,
        "prev_token": sample["prev"] if sample["prev"] else None,
        "next_token": sample["next"] if sample["next"] else None,
    }

    if sample["prev"]:
        ego_motion = compute_ego_motion(nusc, sample_token, sample["prev"])
        info["ego_motion"] = ego_motion
    else:
        info["ego_motion"] = np.eye(4, dtype=np.float32)

    return info


def create_splits(
    nusc,
    version: str,
) -> Tuple[List[str], List[str]]:
    """
    Get train and val sample tokens based on the dataset version.

    Returns:
        train_tokens: list of sample tokens for training
        val_tokens: list of sample tokens for validation
    """
    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()

    if "mini" in version:
        train_scenes = splits["mini_train"]
        val_scenes = splits["mini_val"]
    elif "test" in version:
        train_scenes = []
        val_scenes = splits["test"]
    else:
        train_scenes = splits["train"]
        val_scenes = splits["val"]

    scene_name_to_token = {scene["name"]: scene["token"] for scene in nusc.scene}

    def get_sample_tokens_for_scenes(scene_names):
        tokens = []
        for scene_name in scene_names:
            if scene_name not in scene_name_to_token:
                continue
            scene_token = scene_name_to_token[scene_name]
            scene = nusc.get("scene", scene_token)
            sample_token = scene["first_sample_token"]
            while sample_token:
                tokens.append(sample_token)
                sample = nusc.get("sample", sample_token)
                sample_token = sample["next"] if sample["next"] else None
        return tokens

    train_tokens = get_sample_tokens_for_scenes(train_scenes)
    val_tokens = get_sample_tokens_for_scenes(val_scenes)

    return train_tokens, val_tokens


def compute_dataset_statistics(infos: List[Dict]) -> Dict:
    """
    Compute dataset statistics for normalization and analysis.

    Returns:
        stats dict with class counts, bbox statistics, etc.
    """
    class_counts = defaultdict(int)
    all_centers = []
    all_sizes = []
    all_velocities = []

    for info in infos:
        for label in info["gt_labels"]:
            class_name = DETECTION_NAMES[label]
            class_counts[class_name] += 1

        if len(info["gt_bboxes"]) > 0:
            all_centers.append(info["gt_bboxes"][:, :3])
            all_sizes.append(info["gt_bboxes"][:, 3:6])
            all_velocities.append(info["gt_bboxes"][:, 8:10])

    stats = {
        "num_samples": len(infos),
        "class_counts": dict(class_counts),
        "total_annotations": sum(class_counts.values()),
    }

    if all_centers:
        centers = np.concatenate(all_centers, axis=0)
        sizes = np.concatenate(all_sizes, axis=0)
        velocities = np.concatenate(all_velocities, axis=0)

        stats["center_mean"] = centers.mean(axis=0).tolist()
        stats["center_std"] = centers.std(axis=0).tolist()
        stats["center_min"] = centers.min(axis=0).tolist()
        stats["center_max"] = centers.max(axis=0).tolist()
        stats["size_mean"] = sizes.mean(axis=0).tolist()
        stats["size_std"] = sizes.std(axis=0).tolist()
        stats["velocity_mean"] = velocities.mean(axis=0).tolist()
        stats["velocity_std"] = velocities.std(axis=0).tolist()

    return stats


def build_temporal_index(infos: List[Dict]) -> List[Dict]:
    """
    Build temporal index linking each sample to its previous frame index.
    This allows efficient lookup during training for StreamPETR.
    """
    token_to_idx = {info["token"]: idx for idx, info in enumerate(infos)}

    for idx, info in enumerate(infos):
        prev_token = info.get("prev_token")
        if prev_token and prev_token in token_to_idx:
            info["prev_idx"] = token_to_idx[prev_token]
        else:
            info["prev_idx"] = -1

        next_token = info.get("next_token")
        if next_token and next_token in token_to_idx:
            info["next_idx"] = token_to_idx[next_token]
        else:
            info["next_idx"] = -1

    return infos


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("nuScenes Data Preparation for PETR/StreamPETR")
    print("=" * 60)
    print(f"  Data root:   {args.data_root}")
    print(f"  Version:     {args.version}")
    print(f"  Output dir:  {args.output_dir}")
    print("=" * 60)

    print("\nLoading nuScenes database...")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    print(f"\nDataset loaded:")
    print(f"  Scenes: {len(nusc.scene)}")
    print(f"  Samples: {len(nusc.sample)}")
    print(f"  Sample data: {len(nusc.sample_data)}")
    print(f"  Annotations: {len(nusc.sample_annotation)}")

    print("\nCreating train/val splits...")
    train_tokens, val_tokens = create_splits(nusc, args.version)
    print(f"  Train samples: {len(train_tokens)}")
    print(f"  Val samples: {len(val_tokens)}")

    print("\nProcessing training samples...")
    train_infos = []
    for i, token in enumerate(train_tokens):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(train_tokens)}")
        info = process_sample(nusc, token, args.data_root)
        train_infos.append(info)

    print("\nProcessing validation samples...")
    val_infos = []
    for i, token in enumerate(val_tokens):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(val_tokens)}")
        info = process_sample(nusc, token, args.data_root)
        val_infos.append(info)

    print("\nBuilding temporal indices...")
    train_infos = build_temporal_index(train_infos)
    val_infos = build_temporal_index(val_infos)

    temporal_linked_train = sum(1 for info in train_infos if info["prev_idx"] >= 0)
    temporal_linked_val = sum(1 for info in val_infos if info["prev_idx"] >= 0)
    print(f"  Train: {temporal_linked_train}/{len(train_infos)} have previous frame")
    print(f"  Val: {temporal_linked_val}/{len(val_infos)} have previous frame")

    print("\nComputing dataset statistics...")
    train_stats = compute_dataset_statistics(train_infos)
    val_stats = compute_dataset_statistics(val_infos)

    print(f"\nTraining set statistics:")
    print(f"  Total annotations: {train_stats['total_annotations']}")
    print(f"  Per-class counts:")
    for cls_name in DETECTION_NAMES:
        count = train_stats["class_counts"].get(cls_name, 0)
        print(f"    {cls_name:25s}: {count}")

    if "center_mean" in train_stats:
        print(f"  Center mean: {train_stats['center_mean']}")
        print(f"  Center range: {train_stats['center_min']} to {train_stats['center_max']}")
        print(f"  Size mean: {train_stats['size_mean']}")

    print("\nSaving info files...")

    train_info_path = os.path.join(args.output_dir, f"petr_infos_train_{args.version.replace('.', '_')}.pkl")
    with open(train_info_path, "wb") as f:
        pickle.dump(train_infos, f)
    print(f"  Train infos: {train_info_path} ({len(train_infos)} samples)")

    val_info_path = os.path.join(args.output_dir, f"petr_infos_val_{args.version.replace('.', '_')}.pkl")
    with open(val_info_path, "wb") as f:
        pickle.dump(val_infos, f)
    print(f"  Val infos: {val_info_path} ({len(val_infos)} samples)")

    stats_path = os.path.join(args.output_dir, f"dataset_stats_{args.version.replace('.', '_')}.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump({"train": train_stats, "val": val_stats}, f)
    print(f"  Statistics: {stats_path}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  {train_info_path}")
    print(f"  {val_info_path}")
    print(f"  {stats_path}")
    print(f"\nUse these paths in your training config YAML.")


if __name__ == "__main__":
    main()
