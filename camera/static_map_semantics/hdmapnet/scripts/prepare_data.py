"""
Prepare HDMapNet BEV ground truth from nuScenes dataset.

Generates per-sample .npz files containing:
  - Camera images (resized)
  - Camera intrinsics/extrinsics
  - Semantic BEV masks (lane_divider, road_divider, ped_crossing)
  - Instance BEV masks
  - Direction BEV masks (tangent vectors along line elements)

Usage:
    python prepare_data.py \
        --dataroot /path/to/nuscenes \
        --version v1.0-mini \
        --split train \
        --output_dir /path/to/output \
        --num_workers 8
"""

import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes

# BEV grid configuration
BEV_HEIGHT = 200  # pixels
BEV_WIDTH = 200   # pixels
X_RANGE = (-30.0, 30.0)  # meters, left-right
Y_RANGE = (-15.0, 15.0)  # meters, front-back

# Map layer configuration
LINE_LAYERS = ['lane_divider', 'road_divider']
POLYGON_LAYERS = ['ped_crossing']
NUM_CLASSES = 3  # lane_divider, road_divider, ped_crossing

# Camera names in canonical order
CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_FRONT_LEFT',
]

# Line thickness for rasterization (pixels)
LINE_THICKNESS = 2


def world_to_bev(points: np.ndarray) -> np.ndarray:
    """
    Convert world-frame 2D points (already in ego frame) to BEV pixel coordinates.

    Args:
        points: (N, 2) array of (x, y) in ego vehicle frame (meters).

    Returns:
        (N, 2) array of (col, row) pixel coordinates in BEV image.
    """
    x = points[:, 0]
    y = points[:, 1]

    # Map x from [X_RANGE[0], X_RANGE[1]] -> [0, BEV_WIDTH]
    col = (x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * BEV_WIDTH
    # Map y from [Y_RANGE[0], Y_RANGE[1]] -> [BEV_HEIGHT, 0] (flip y so forward is up)
    row = (1.0 - (y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0])) * BEV_HEIGHT

    return np.stack([col, row], axis=-1)


def transform_points_world_to_ego(
    points: np.ndarray,
    ego_translation: np.ndarray,
    ego_rotation: Quaternion,
) -> np.ndarray:
    """
    Transform 3D or 2D points from world frame to ego vehicle frame.

    Args:
        points: (N, 2) or (N, 3) world coordinates.
        ego_translation: (3,) ego pose translation.
        ego_rotation: Quaternion for ego pose rotation.

    Returns:
        (N, 2) or (N, 3) points in ego frame.
    """
    if points.shape[1] == 2:
        # Pad with zeros for z
        points_3d = np.concatenate(
            [points, np.zeros((points.shape[0], 1))], axis=1
        )
    else:
        points_3d = points

    # World to ego: p_ego = R^{-1} * (p_world - t)
    rot_inv = ego_rotation.rotation_matrix.T
    translated = points_3d - ego_translation[np.newaxis, :]
    transformed = translated @ rot_inv.T

    if points.shape[1] == 2:
        return transformed[:, :2]
    return transformed


def get_map_layers_in_ego(
    nusc_map: NuScenesMap,
    ego_translation: np.ndarray,
    ego_rotation: Quaternion,
    patch_radius: float = 50.0,
) -> Dict[str, List[np.ndarray]]:
    """
    Retrieve map layer geometries within a patch around the ego and transform to ego frame.

    Args:
        nusc_map: NuScenesMap instance.
        ego_translation: (3,) ego pose translation (world frame).
        ego_rotation: Quaternion for ego rotation.
        patch_radius: radius (meters) of the square patch to query.

    Returns:
        Dictionary mapping layer name to list of (N, 2) arrays in ego frame.
    """
    # Define a square patch in world coordinates centered on ego
    x, y = ego_translation[0], ego_translation[1]
    patch_box = (x, y, 2 * patch_radius, 2 * patch_radius)
    patch_angle = np.degrees(
        2 * np.arctan2(ego_rotation.z, ego_rotation.w)
    )
    # We query without rotating the patch; we'll transform points later
    patch_box_world = (x, y, 2 * patch_radius, 2 * patch_radius)

    layers = {}

    # Lane dividers (lines)
    layer_name = 'lane_divider'
    records = nusc_map.get_records_in_patch(
        patch_box_world, [layer_name], mode='intersect'
    )
    lines = []
    for token in records.get(layer_name, []):
        record = nusc_map.get(layer_name, token)
        line_token = record['line_token']
        line_record = nusc_map.get('line', line_token)
        nodes = [nusc_map.get('node', n) for n in line_record['node_tokens']]
        pts = np.array([[n['x'], n['y']] for n in nodes], dtype=np.float64)
        if len(pts) >= 2:
            pts_ego = transform_points_world_to_ego(pts, ego_translation, ego_rotation)
            lines.append(pts_ego)
    layers['lane_divider'] = lines

    # Road dividers (lines)
    layer_name = 'road_divider'
    records = nusc_map.get_records_in_patch(
        patch_box_world, [layer_name], mode='intersect'
    )
    lines = []
    for token in records.get(layer_name, []):
        record = nusc_map.get(layer_name, token)
        line_token = record['line_token']
        line_record = nusc_map.get('line', line_token)
        nodes = [nusc_map.get('node', n) for n in line_record['node_tokens']]
        pts = np.array([[n['x'], n['y']] for n in nodes], dtype=np.float64)
        if len(pts) >= 2:
            pts_ego = transform_points_world_to_ego(pts, ego_translation, ego_rotation)
            lines.append(pts_ego)
    layers['road_divider'] = lines

    # Pedestrian crossings (polygons)
    layer_name = 'ped_crossing'
    records = nusc_map.get_records_in_patch(
        patch_box_world, [layer_name], mode='intersect'
    )
    polygons = []
    for token in records.get(layer_name, []):
        record = nusc_map.get(layer_name, token)
        polygon_token = record['polygon_token']
        polygon_record = nusc_map.get('polygon', polygon_token)
        nodes = [
            nusc_map.get('node', n) for n in polygon_record['exterior_node_tokens']
        ]
        pts = np.array([[n['x'], n['y']] for n in nodes], dtype=np.float64)
        if len(pts) >= 3:
            pts_ego = transform_points_world_to_ego(pts, ego_translation, ego_rotation)
            polygons.append(pts_ego)
    layers['ped_crossing'] = polygons

    return layers


def rasterize_map(
    layers: Dict[str, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize map layers to BEV masks.

    Args:
        layers: dict mapping layer name -> list of polylines/polygons in ego frame.

    Returns:
        semantic_masks: (200, 200, 3) float32, binary masks per class.
        instance_masks: (200, 200) int32, unique instance IDs.
        direction_masks: (200, 200, 2) float32, unit tangent direction vectors.
    """
    semantic_masks = np.zeros((BEV_HEIGHT, BEV_WIDTH, NUM_CLASSES), dtype=np.float32)
    instance_masks = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.int32)
    direction_masks = np.zeros((BEV_HEIGHT, BEV_WIDTH, 2), dtype=np.float32)

    instance_id = 1  # Start from 1; 0 means background

    # Channel 0: lane_divider (lines)
    for pts_ego in layers.get('lane_divider', []):
        pts_bev = world_to_bev(pts_ego)
        pts_px = np.round(pts_bev).astype(np.int32)

        # Semantic mask
        cv2.polylines(
            semantic_masks[:, :, 0:1].view(np.uint8) if False else semantic_masks[:, :, 0],
            [pts_px],
            isClosed=False,
            color=1.0,
            thickness=LINE_THICKNESS,
        )

        # Instance mask
        inst_canvas = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
        cv2.polylines(inst_canvas, [pts_px], isClosed=False, color=1, thickness=LINE_THICKNESS)
        instance_masks[inst_canvas > 0] = instance_id

        # Direction mask
        _rasterize_direction(direction_masks, pts_ego, pts_bev, inst_canvas)

        instance_id += 1

    # Channel 1: road_divider (lines)
    for pts_ego in layers.get('road_divider', []):
        pts_bev = world_to_bev(pts_ego)
        pts_px = np.round(pts_bev).astype(np.int32)

        cv2.polylines(
            semantic_masks[:, :, 1],
            [pts_px],
            isClosed=False,
            color=1.0,
            thickness=LINE_THICKNESS,
        )

        inst_canvas = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
        cv2.polylines(inst_canvas, [pts_px], isClosed=False, color=1, thickness=LINE_THICKNESS)
        instance_masks[inst_canvas > 0] = instance_id

        # Direction mask for road dividers as well
        _rasterize_direction(direction_masks, pts_ego, pts_bev, inst_canvas)

        instance_id += 1

    # Channel 2: ped_crossing (polygons)
    for pts_ego in layers.get('ped_crossing', []):
        pts_bev = world_to_bev(pts_ego)
        pts_px = np.round(pts_bev).astype(np.int32)

        cv2.fillPoly(
            semantic_masks[:, :, 2],
            [pts_px],
            color=1.0,
        )

        inst_canvas = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
        cv2.fillPoly(inst_canvas, [pts_px], color=1)
        instance_masks[inst_canvas > 0] = instance_id

        # No direction for polygon elements
        instance_id += 1

    return semantic_masks, instance_masks, direction_masks


def _rasterize_direction(
    direction_masks: np.ndarray,
    pts_ego: np.ndarray,
    pts_bev: np.ndarray,
    mask: np.ndarray,
) -> None:
    """
    Rasterize tangent direction vectors along a polyline onto the direction mask.

    For each segment of the polyline, compute the unit tangent in ego frame,
    then fill all pixels belonging to that segment with that direction.

    Args:
        direction_masks: (H, W, 2) float32 array to write into.
        pts_ego: (N, 2) polyline points in ego frame (meters).
        pts_bev: (N, 2) polyline points in BEV pixel coordinates.
        mask: (H, W) uint8 binary mask of the rasterized polyline.
    """
    if len(pts_ego) < 2:
        return

    for i in range(len(pts_ego) - 1):
        # Compute tangent direction in ego frame
        dx = pts_ego[i + 1, 0] - pts_ego[i, 0]
        dy = pts_ego[i + 1, 1] - pts_ego[i, 1]
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        dir_x = dx / length
        dir_y = dy / length

        # Create a mask for just this segment
        seg_canvas = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
        p1 = np.round(pts_bev[i]).astype(np.int32)
        p2 = np.round(pts_bev[i + 1]).astype(np.int32)
        cv2.line(seg_canvas, tuple(p1), tuple(p2), color=1, thickness=LINE_THICKNESS)

        # Only write where segment overlaps the polyline mask
        segment_pixels = (seg_canvas > 0) & (mask > 0)
        direction_masks[segment_pixels, 0] = dir_x
        direction_masks[segment_pixels, 1] = dir_y


def get_camera_data(
    nusc: NuScenes,
    sample: dict,
    img_height: int,
    img_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load camera images, intrinsics, and extrinsics for all 6 cameras.

    Args:
        nusc: NuScenes instance.
        sample: Sample record dict.
        img_height: Target image height after resize.
        img_width: Target image width after resize.

    Returns:
        images: (6, H, W, 3) uint8
        intrinsics: (6, 3, 3) float32 (adjusted for resize)
        extrinsics: (6, 4, 4) float32 (camera-to-ego transform)
    """
    images = np.zeros((6, img_height, img_width, 3), dtype=np.uint8)
    intrinsics = np.zeros((6, 3, 3), dtype=np.float32)
    extrinsics = np.zeros((6, 4, 4), dtype=np.float32)

    for idx, cam_name in enumerate(CAMERA_NAMES):
        cam_token = sample['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)

        # Load and resize image
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img_resized = img.resize((img_width, img_height), Image.BILINEAR)
        images[idx] = np.array(img_resized, dtype=np.uint8)

        # Get calibration
        calib_token = cam_data['calibrated_sensor_token']
        calib = nusc.get('calibrated_sensor', calib_token)

        # Intrinsics (3x3)
        K = np.array(calib['camera_intrinsic'], dtype=np.float32)
        # Adjust intrinsics for resize
        scale_x = img_width / orig_w
        scale_y = img_height / orig_h
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        intrinsics[idx] = K

        # Extrinsics: sensor (camera) to ego vehicle transform
        translation = np.array(calib['translation'], dtype=np.float32)
        rotation = Quaternion(calib['rotation'])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation.rotation_matrix.astype(np.float32)
        T[:3, 3] = translation
        extrinsics[idx] = T

    return images, intrinsics, extrinsics


def get_ego_pose(nusc: NuScenes, sample: dict) -> Tuple[np.ndarray, Quaternion]:
    """
    Get ego pose (translation, rotation) for a sample.

    Uses the LIDAR_TOP timestamp as the canonical ego pose for the sample.

    Args:
        nusc: NuScenes instance.
        sample: Sample record dict.

    Returns:
        translation: (3,) float64 array
        rotation: Quaternion
    """
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    translation = np.array(ego_pose['translation'], dtype=np.float64)
    rotation = Quaternion(ego_pose['rotation'])
    return translation, rotation


def get_scene_to_map_name(nusc: NuScenes) -> Dict[str, str]:
    """
    Build mapping from scene name to map location name.

    Args:
        nusc: NuScenes instance.

    Returns:
        Dict mapping scene_name -> map location (e.g., 'singapore-onenorth').
    """
    scene_to_map = {}
    for scene in nusc.scene:
        log = nusc.get('log', scene['log_token'])
        scene_to_map[scene['name']] = log['location']
    return scene_to_map


def process_sample(
    sample_token: str,
    dataroot: str,
    version: str,
    output_dir: str,
    img_height: int,
    img_width: int,
    scene_name: str,
    map_location: str,
) -> str:
    """
    Process a single sample: generate BEV ground truth and save as .npz.

    This function is designed to be called in a worker process.

    Args:
        sample_token: nuScenes sample token.
        dataroot: Path to nuScenes data root.
        version: nuScenes version string.
        output_dir: Output directory for .npz files.
        img_height: Target image height.
        img_width: Target image width.
        scene_name: Name of the scene this sample belongs to.
        map_location: Map location name for this scene.

    Returns:
        Output file path on success, or error message string.
    """
    try:
        # Initialize nuScenes and map API in the worker process
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_location)

        sample = nusc.get('sample', sample_token)

        # Get ego pose
        ego_translation, ego_rotation = get_ego_pose(nusc, sample)

        # Get camera data
        images, intrinsics_arr, extrinsics_arr = get_camera_data(
            nusc, sample, img_height, img_width
        )

        # Get map layers in ego frame
        layers = get_map_layers_in_ego(nusc_map, ego_translation, ego_rotation)

        # Rasterize to BEV masks
        semantic_masks, instance_masks, direction_masks = rasterize_map(layers)

        # Save output
        output_filename = f"{sample_token}.npz"
        output_path = os.path.join(output_dir, output_filename)

        np.savez_compressed(
            output_path,
            images=images,
            extrinsics=extrinsics_arr,
            intrinsics=intrinsics_arr,
            semantic_masks=semantic_masks,
            instance_masks=instance_masks,
            direction_masks=direction_masks,
        )

        return output_path

    except Exception as e:
        return f"ERROR processing {sample_token}: {str(e)}"


def process_sample_wrapper(args: tuple) -> str:
    """Wrapper to unpack arguments for multiprocessing Pool.map."""
    return process_sample(*args)


def get_split_scenes(version: str, split: str) -> List[str]:
    """
    Get list of scene names for a given split.

    Args:
        version: nuScenes version.
        split: 'train' or 'val'.

    Returns:
        List of scene name strings.
    """
    splits = create_splits_scenes()

    if version == 'v1.0-mini':
        split_key = f"mini_{split}"
    else:
        split_key = split

    if split_key not in splits:
        available = list(splits.keys())
        raise ValueError(
            f"Split '{split_key}' not found. Available splits: {available}"
        )

    return splits[split_key]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HDMapNet BEV ground truth from nuScenes."
    )
    parser.add_argument(
        '--dataroot',
        type=str,
        required=True,
        help='Path to nuScenes data root directory.',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        choices=['v1.0-mini', 'v1.0-trainval'],
        help='nuScenes dataset version.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Dataset split to process.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for .npz files.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of parallel worker processes.',
    )
    parser.add_argument(
        '--img_height',
        type=int,
        default=128,
        help='Target image height after resize.',
    )
    parser.add_argument(
        '--img_width',
        type=int,
        default=352,
        help='Target image width after resize.',
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Initializing nuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Get split scenes
    split_scenes = get_split_scenes(args.version, args.split)
    print(f"Split '{args.split}' contains {len(split_scenes)} scenes.")

    # Build scene name to map location mapping
    scene_to_map = get_scene_to_map_name(nusc)

    # Collect all sample tokens for the split
    sample_args_list = []
    for scene in nusc.scene:
        if scene['name'] not in split_scenes:
            continue

        map_location = scene_to_map[scene['name']]

        # Iterate through all samples in the scene
        sample_token = scene['first_sample_token']
        while sample_token:
            sample_args_list.append((
                sample_token,
                args.dataroot,
                args.version,
                args.output_dir,
                args.img_height,
                args.img_width,
                scene['name'],
                map_location,
            ))
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']

    total_samples = len(sample_args_list)
    print(f"Total samples to process: {total_samples}")

    if total_samples == 0:
        print("No samples found for the given split. Exiting.")
        return

    # Process with multiprocessing
    print(f"Processing with {args.num_workers} workers...")

    if args.num_workers <= 1:
        # Single-process mode for debugging
        results = []
        for i, sample_args in enumerate(sample_args_list):
            result = process_sample_wrapper(sample_args)
            results.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == total_samples:
                print(f"  Progress: {i + 1}/{total_samples}")
    else:
        with Pool(processes=args.num_workers) as pool:
            results = []
            for i, result in enumerate(
                pool.imap_unordered(process_sample_wrapper, sample_args_list)
            ):
                results.append(result)
                if (i + 1) % 50 == 0 or (i + 1) == total_samples:
                    print(f"  Progress: {i + 1}/{total_samples}")

    # Report results
    errors = [r for r in results if r.startswith("ERROR")]
    successes = len(results) - len(errors)
    print(f"\nDone! Processed {successes}/{total_samples} samples successfully.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors[:20]:  # Print first 20 errors
            print(f"  {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")

    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
