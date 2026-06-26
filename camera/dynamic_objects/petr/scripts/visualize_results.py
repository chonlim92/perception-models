"""
Visualization script for PETR/StreamPETR inference results.

Features:
  - Load inference results from pickle
  - Project 3D bounding boxes onto multi-view images
  - Draw boxes with class-specific colors and confidence scores
  - Bird's Eye View (BEV) visualization
  - Save individual images or create video sequences
  - Temporal visualization showing StreamPETR query propagation
"""

import os
import argparse
import pickle
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path


NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

CLASS_COLORS = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PETR inference results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to inference results pickle file")
    parser.add_argument("--data_info", type=str, required=True,
                        help="Path to data info pickle (for image paths and camera params)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of nuScenes data")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="Output directory for visualization images")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Maximum number of samples to visualize")
    parser.add_argument("--score_threshold", type=float, default=0.3,
                        help="Minimum score to visualize")
    parser.add_argument("--show_bev", action="store_true",
                        help="Generate BEV visualizations")
    parser.add_argument("--create_video", action="store_true",
                        help="Create video from visualizations")
    parser.add_argument("--video_fps", type=int, default=2,
                        help="FPS for output video")
    parser.add_argument("--temporal", action="store_true",
                        help="Visualize temporal query propagation")
    parser.add_argument("--bev_range", type=float, default=50.0,
                        help="BEV visualization range in meters")
    return parser.parse_args()


def get_3d_box_corners(bbox: np.ndarray) -> np.ndarray:
    """
    Compute 8 corners of a 3D bounding box.

    Args:
        bbox: (10,) [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]

    Returns:
        corners: (8, 3) corner coordinates in world frame
    """
    cx, cy, cz = bbox[0], bbox[1], bbox[2]
    w, l, h = bbox[3], bbox[4], bbox[5]
    sin_yaw, cos_yaw = bbox[6], bbox[7]

    dx = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    dy = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])

    rot_dx = cos_yaw * dx - sin_yaw * dy
    rot_dy = sin_yaw * dx + cos_yaw * dy

    corners = np.stack([
        cx + rot_dx,
        cy + rot_dy,
        cz + dz,
    ], axis=-1)

    return corners


def project_3d_to_image(
    corners_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Project 3D box corners onto an image using camera parameters.

    Args:
        corners_3d: (8, 3) world-frame corners
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) camera-to-world transformation (we invert to get world-to-camera)
        image_size: (H, W) image dimensions

    Returns:
        corners_2d: (8, 2) projected pixel coordinates, or None if behind camera
    """
    world_to_camera = np.linalg.inv(extrinsics)

    corners_homo = np.concatenate(
        [corners_3d, np.ones((8, 1))], axis=-1
    )
    corners_camera = (world_to_camera @ corners_homo.T).T[:, :3]

    if np.all(corners_camera[:, 2] <= 0):
        return None

    valid_mask = corners_camera[:, 2] > 0.1
    if not np.any(valid_mask):
        return None

    corners_2d_homo = (intrinsics @ corners_camera.T).T
    z = corners_2d_homo[:, 2:3]
    z = np.where(z > 0.1, z, 0.1)
    corners_2d = corners_2d_homo[:, :2] / z

    H, W = image_size
    in_image = (
        (corners_2d[:, 0] >= -W * 0.5) & (corners_2d[:, 0] < W * 1.5) &
        (corners_2d[:, 1] >= -H * 0.5) & (corners_2d[:, 1] < H * 1.5)
    )
    if not np.any(in_image & valid_mask):
        return None

    return corners_2d


def draw_3d_box_on_image(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    """
    Draw projected 3D bounding box edges on image.

    Args:
        image: (H, W, 3) BGR image
        corners_2d: (8, 2) projected pixel coordinates
        color: BGR color tuple
        thickness: line thickness
    """
    corners = corners_2d.astype(np.int32)

    bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    vertical_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]

    for i, j in bottom_edges:
        cv2.line(image, tuple(corners[i]), tuple(corners[j]), color, thickness)

    for i, j in top_edges:
        cv2.line(image, tuple(corners[i]), tuple(corners[j]), color, thickness)

    for i, j in vertical_edges:
        cv2.line(image, tuple(corners[i]), tuple(corners[j]), color, thickness)

    front_edges = [(0, 1), (4, 5)]
    for i, j in front_edges:
        cv2.line(image, tuple(corners[i]), tuple(corners[j]), color, thickness + 1)


def draw_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.5,
):
    """Draw text label with background on image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, 1)

    x, y = position
    cv2.rectangle(image, (x, y - text_h - 4), (x + text_w + 4, y + 4), color, -1)
    cv2.putText(image, text, (x + 2, y), font, font_scale, (255, 255, 255), 1)


def visualize_multiview(
    image_paths: List[str],
    data_root: str,
    detections: List[Dict],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """
    Create multi-view visualization with projected 3D boxes.

    Args:
        image_paths: list of 6 camera image paths
        data_root: data root directory
        detections: list of detection dicts
        intrinsics: (6, 3, 3) camera intrinsics
        extrinsics: (6, 4, 4) camera extrinsics
        score_threshold: minimum score to draw

    Returns:
        vis_image: combined multi-view visualization image
    """
    images = []
    for img_path in image_paths:
        full_path = os.path.join(data_root, img_path)
        img = cv2.imread(full_path)
        if img is None:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)
        images.append(img)

    for det in detections:
        if det["score"] < score_threshold:
            continue

        class_name = det["class"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        bbox = np.array([
            det["center"][0], det["center"][1], det["center"][2],
            det["size"][0], det["size"][1], det["size"][2],
            np.sin(det["yaw"]), np.cos(det["yaw"]),
            det["velocity"][0], det["velocity"][1],
        ])

        corners_3d = get_3d_box_corners(bbox)

        for cam_idx in range(6):
            img_h, img_w = images[cam_idx].shape[:2]
            corners_2d = project_3d_to_image(
                corners_3d,
                intrinsics[cam_idx],
                extrinsics[cam_idx],
                (img_h, img_w),
            )

            if corners_2d is not None:
                draw_3d_box_on_image(images[cam_idx], corners_2d, color)
                label_text = f"{class_name} {det['score']:.2f}"
                label_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].min()) - 5)
                label_pos = (max(0, label_pos[0]), max(15, label_pos[1]))
                draw_label(images[cam_idx], label_text, label_pos, color)

    target_h, target_w = 300, 533
    resized = [cv2.resize(img, (target_w, target_h)) for img in images]

    for i, cam_name in enumerate(CAMERA_NAMES):
        cv2.putText(resized[i], cam_name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    top_row = np.concatenate([resized[5], resized[0], resized[1]], axis=1)
    bottom_row = np.concatenate([resized[4], resized[3], resized[2]], axis=1)
    vis_image = np.concatenate([top_row, bottom_row], axis=0)

    return vis_image


def create_bev_visualization(
    detections: List[Dict],
    bev_range: float = 50.0,
    image_size: int = 800,
    score_threshold: float = 0.3,
    ego_marker: bool = True,
) -> np.ndarray:
    """
    Create Bird's Eye View visualization of detections.

    Args:
        detections: list of detection dicts
        bev_range: visualization range in meters (symmetric around ego)
        image_size: output image size in pixels (square)
        score_threshold: minimum score to visualize
        ego_marker: whether to draw ego vehicle marker

    Returns:
        bev_image: (image_size, image_size, 3) BEV visualization
    """
    bev_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    bev_image[:] = (40, 40, 40)

    scale = image_size / (2 * bev_range)
    center = image_size // 2

    for dist in [10, 20, 30, 40, 50]:
        radius = int(dist * scale)
        cv2.circle(bev_image, (center, center), radius, (80, 80, 80), 1)
        cv2.putText(bev_image, f"{dist}m", (center + radius + 5, center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    cv2.line(bev_image, (center, 0), (center, image_size), (80, 80, 80), 1)
    cv2.line(bev_image, (0, center), (image_size, center), (80, 80, 80), 1)

    if ego_marker:
        ego_pts = np.array([
            [center, center - 10],
            [center - 6, center + 6],
            [center + 6, center + 6],
        ], dtype=np.int32)
        cv2.fillPoly(bev_image, [ego_pts], (0, 200, 0))

    for det in detections:
        if det["score"] < score_threshold:
            continue

        class_name = det["class"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        cx, cy = det["center"][0], det["center"][1]
        w, l = det["size"][0], det["size"][1]
        yaw = det["yaw"]

        px = int(center + cy * scale)
        py = int(center - cx * scale)

        if 0 <= px < image_size and 0 <= py < image_size:
            half_w = w * scale / 2
            half_l = l * scale / 2

            corners_local = np.array([
                [-half_l, -half_w],
                [half_l, -half_w],
                [half_l, half_w],
                [-half_l, half_w],
            ])

            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            corners_rotated = (rot @ corners_local.T).T

            corners_pixel = corners_rotated + np.array([py, px])
            corners_pixel = corners_pixel.astype(np.int32)

            cv2.polylines(bev_image, [corners_pixel], True, color, 2)

            front_mid = ((corners_pixel[0] + corners_pixel[1]) / 2).astype(np.int32)
            box_center = np.array([py, px], dtype=np.int32)
            cv2.line(bev_image, tuple(box_center), tuple(front_mid), color, 2)

            if det["velocity"][0] != 0 or det["velocity"][1] != 0:
                vx, vy = det["velocity"]
                vel_end_x = int(py - vx * scale * 0.5)
                vel_end_y = int(px + vy * scale * 0.5)
                cv2.arrowedLine(bev_image, (py, px), (vel_end_x, vel_end_y),
                                (0, 255, 255), 1, tipLength=0.3)

    legend_y = 20
    for cls_name, color in CLASS_COLORS.items():
        cv2.rectangle(bev_image, (10, legend_y - 10), (25, legend_y + 2), color, -1)
        cv2.putText(bev_image, cls_name, (30, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        legend_y += 18

    return bev_image


def visualize_temporal_propagation(
    results_sequence: List[Dict],
    data_infos: List[Dict],
    data_root: str,
    output_dir: str,
    max_frames: int = 20,
    score_threshold: float = 0.3,
):
    """
    Visualize temporal query propagation across frames for StreamPETR.

    Shows how object queries track objects across time by color-coding consistent
    detections with the same query ID across consecutive frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    query_colors = np.random.randint(50, 255, size=(900, 3)).tolist()

    frames_to_vis = min(max_frames, len(results_sequence))

    for frame_idx in range(frames_to_vis):
        result = results_sequence[frame_idx]
        info = data_infos[result["sample_idx"]]

        front_path = os.path.join(data_root, info["img_paths"][0])
        img = cv2.imread(front_path)
        if img is None:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)

        for det_idx, det in enumerate(result["detections"]):
            if det["score"] < score_threshold:
                continue

            color = tuple(query_colors[det_idx % len(query_colors)])

            bbox = np.array([
                det["center"][0], det["center"][1], det["center"][2],
                det["size"][0], det["size"][1], det["size"][2],
                np.sin(det["yaw"]), np.cos(det["yaw"]),
                det["velocity"][0], det["velocity"][1],
            ])

            corners_3d = get_3d_box_corners(bbox)
            corners_2d = project_3d_to_image(
                corners_3d,
                info["intrinsics"][0],
                info["extrinsics"][0],
                img.shape[:2],
            )

            if corners_2d is not None:
                draw_3d_box_on_image(img, corners_2d, color, thickness=2)
                label = f"Q{det_idx} {det['class']} {det['score']:.2f}"
                pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].min()) - 5)
                pos = (max(0, pos[0]), max(15, pos[1]))
                draw_label(img, label, pos, color)

        frame_label = f"Frame {frame_idx} | {len(result['detections'])} detections"
        cv2.putText(img, frame_label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        output_path = os.path.join(output_dir, f"temporal_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, img)


def create_video(
    image_dir: str,
    output_path: str,
    fps: int = 2,
    pattern: str = "*.jpg",
):
    """Create video from a directory of images."""
    from pathlib import Path

    image_files = sorted(Path(image_dir).glob(pattern))
    if not image_files:
        print(f"No images found in {image_dir} matching {pattern}")
        return

    first_img = cv2.imread(str(image_files[0]))
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)

    writer.release()
    print(f"Video saved: {output_path} ({len(image_files)} frames, {fps} FPS)")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    with open(args.results, "rb") as f:
        results = pickle.load(f)

    print("Loading data info...")
    with open(args.data_info, "rb") as f:
        data_infos = pickle.load(f)

    print(f"  Results: {len(results)} samples")
    print(f"  Data infos: {len(data_infos)} samples")

    num_to_visualize = min(args.max_samples, len(results))
    print(f"\nVisualizing {num_to_visualize} samples...")

    multiview_dir = os.path.join(args.output_dir, "multiview")
    os.makedirs(multiview_dir, exist_ok=True)

    bev_dir = os.path.join(args.output_dir, "bev")
    if args.show_bev:
        os.makedirs(bev_dir, exist_ok=True)

    for idx in range(num_to_visualize):
        result = results[idx]
        sample_idx = result["sample_idx"]

        if sample_idx >= len(data_infos):
            continue

        info = data_infos[sample_idx]
        detections = result["detections"]

        if (idx + 1) % 10 == 0:
            print(f"  Visualizing {idx + 1}/{num_to_visualize}")

        vis_multiview = visualize_multiview(
            image_paths=info["img_paths"],
            data_root=args.data_root,
            detections=detections,
            intrinsics=info["intrinsics"],
            extrinsics=info["extrinsics"],
            score_threshold=args.score_threshold,
        )
        mv_path = os.path.join(multiview_dir, f"sample_{sample_idx:06d}.jpg")
        cv2.imwrite(mv_path, vis_multiview)

        if args.show_bev:
            vis_bev = create_bev_visualization(
                detections=detections,
                bev_range=args.bev_range,
                score_threshold=args.score_threshold,
            )
            bev_path = os.path.join(bev_dir, f"bev_{sample_idx:06d}.jpg")
            cv2.imwrite(bev_path, vis_bev)

    if args.temporal:
        print("\nGenerating temporal visualization...")
        temporal_dir = os.path.join(args.output_dir, "temporal")
        visualize_temporal_propagation(
            results_sequence=results,
            data_infos=data_infos,
            data_root=args.data_root,
            output_dir=temporal_dir,
            max_frames=min(args.max_samples, len(results)),
            score_threshold=args.score_threshold,
        )

    if args.create_video:
        print("\nCreating videos...")
        mv_video_path = os.path.join(args.output_dir, "multiview_video.mp4")
        create_video(multiview_dir, mv_video_path, fps=args.video_fps)

        if args.show_bev:
            bev_video_path = os.path.join(args.output_dir, "bev_video.mp4")
            create_video(bev_dir, bev_video_path, fps=args.video_fps)

        if args.temporal:
            temporal_video_path = os.path.join(args.output_dir, "temporal_video.mp4")
            create_video(
                os.path.join(args.output_dir, "temporal"),
                temporal_video_path,
                fps=args.video_fps,
            )

    print(f"\nVisualization complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
