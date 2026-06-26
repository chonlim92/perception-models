"""
DETR3D Inference Demo
Load a trained model and run inference on a single nuScenes sample with visualization.

Usage:
    python inference.py --checkpoint ./checkpoints/ --sample_path /path/to/sample_info.json
    python inference.py --checkpoint ./checkpoints/ --data_root /path/to/nuscenes --sample_idx 0
"""

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf

from model import build_detr3d


NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
NUM_CLASSES = len(NUSCENES_CLASSES)

CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

CLASS_COLORS = [
    (255, 0, 0),       # car - red
    (0, 255, 0),       # truck - green
    (0, 0, 255),       # construction_vehicle - blue
    (255, 255, 0),     # bus - yellow
    (255, 0, 255),     # trailer - magenta
    (0, 255, 255),     # barrier - cyan
    (128, 0, 0),       # motorcycle - dark red
    (0, 128, 0),       # bicycle - dark green
    (0, 0, 128),       # pedestrian - dark blue
    (128, 128, 0),     # traffic_cone - olive
]


def parse_args():
    parser = argparse.ArgumentParser(description='DETR3D Inference Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory or weights file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--sample_path', type=str, default=None,
                        help='Path to sample info JSON file')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to use from validation set')
    parser.add_argument('--num_queries', type=int, default=900)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--img_h', type=int, default=900)
    parser.add_argument('--img_w', type=int, default=1600)
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument('--nms_threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    return parser.parse_args()


def load_sample_from_json(sample_path, img_h, img_w):
    """Load sample info from a JSON file."""
    with open(sample_path, 'r') as f:
        sample_info = json.load(f)

    images = []
    intrinsics = []
    extrinsics = []

    for cam_name in CAMERA_NAMES:
        cam_info = sample_info['cameras'][cam_name]
        img_path = cam_info['image_path']

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_w, img_h))
        images.append(img.astype(np.float32))

        intrinsics.append(np.array(cam_info['intrinsic'], dtype=np.float32).reshape(3, 3))
        extrinsics.append(np.array(cam_info['extrinsic'], dtype=np.float32).reshape(4, 4))

    return {
        'images': np.stack(images, axis=0),
        'intrinsics': np.stack(intrinsics, axis=0),
        'extrinsics': np.stack(extrinsics, axis=0),
    }


def load_sample_from_nuscenes(data_root, sample_idx, img_h, img_w):
    """Load a sample from nuScenes dataset files."""
    info_path = os.path.join(data_root, 'nuscenes_infos_val.json')
    if not os.path.exists(info_path):
        info_path = os.path.join(data_root, 'nuscenes_infos_val.pkl')
        import pickle
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(info_path, 'r') as f:
            data = json.load(f)

    if isinstance(data, dict):
        samples = data.get('infos', data.get('data_list', []))
    else:
        samples = data

    sample_info = samples[sample_idx]
    images = []
    intrinsics = []
    extrinsics = []
    image_paths = []

    cams = sample_info.get('cams', {})
    if not cams:
        cam_infos = sample_info.get('images', {})
        for cam_name in CAMERA_NAMES:
            cam_info = cam_infos.get(cam_name, {})
            img_path = os.path.join(data_root, cam_info.get('img_path', ''))
            image_paths.append(img_path)

            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_w, img_h))
            else:
                img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            images.append(img.astype(np.float32))

            cam_intrinsic = np.array(
                cam_info.get('cam_intrinsic', np.eye(3)), dtype=np.float32
            ).reshape(3, 3)
            intrinsics.append(cam_intrinsic)

            lidar2cam = np.array(
                cam_info.get('lidar2cam', np.eye(4)), dtype=np.float32
            ).reshape(4, 4)
            extrinsics.append(lidar2cam)
    else:
        for cam_name in CAMERA_NAMES:
            cam_info = cams.get(cam_name, {})
            img_path = os.path.join(data_root, cam_info.get('data_path', ''))
            image_paths.append(img_path)

            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_w, img_h))
            else:
                img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            images.append(img.astype(np.float32))

            cam_intrinsic = np.array(
                cam_info.get('cam_intrinsic', np.eye(3)), dtype=np.float32
            ).reshape(3, 3)
            intrinsics.append(cam_intrinsic)

            sensor2lidar_rotation = np.array(
                cam_info.get('sensor2lidar_rotation', np.eye(3)), dtype=np.float32
            ).reshape(3, 3)
            sensor2lidar_translation = np.array(
                cam_info.get('sensor2lidar_translation', np.zeros(3)), dtype=np.float32
            )
            sensor2lidar = np.eye(4, dtype=np.float32)
            sensor2lidar[:3, :3] = sensor2lidar_rotation
            sensor2lidar[:3, 3] = sensor2lidar_translation
            lidar2sensor = np.linalg.inv(sensor2lidar)
            extrinsics.append(lidar2sensor)

    return {
        'images': np.stack(images, axis=0),
        'intrinsics': np.stack(intrinsics, axis=0),
        'extrinsics': np.stack(extrinsics, axis=0),
        'image_paths': image_paths,
    }


def nms_3d(boxes, scores, labels, nms_threshold=0.5):
    """
    3D NMS based on BEV (Bird's Eye View) center distance.

    Args:
        boxes: (N, 10) predicted 3D boxes
        scores: (N,) confidence scores
        labels: (N,) class predictions
        nms_threshold: BEV distance threshold for suppression
    Returns:
        keep_indices: indices of kept detections
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    sort_idx = np.argsort(-scores)
    boxes = boxes[sort_idx]
    scores = scores[sort_idx]
    labels = labels[sort_idx]

    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(sort_idx[i])

        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            if labels[i] != labels[j]:
                continue

            dist = np.sqrt(np.sum((boxes[i, :2] - boxes[j, :2]) ** 2))
            size_i = np.sqrt(boxes[i, 3] ** 2 + boxes[i, 4] ** 2)
            size_j = np.sqrt(boxes[j, 3] ** 2 + boxes[j, 4] ** 2)
            avg_size = (size_i + size_j) / 2.0

            if dist < nms_threshold * avg_size:
                suppressed[j] = True

    return np.array(keep, dtype=np.int64)


def project_3d_box_to_image(box_3d, intrinsic, extrinsic):
    """
    Project 3D bounding box corners to 2D image coordinates.

    Args:
        box_3d: (10,) [cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy]
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) world-to-camera transformation matrix
    Returns:
        corners_2d: (8, 2) projected 2D corners or None if behind camera
    """
    cx, cy, cz = box_3d[0], box_3d[1], box_3d[2]
    w, l, h = box_3d[3], box_3d[4], box_3d[5]
    sin_yaw, cos_yaw = box_3d[6], box_3d[7]

    corners_local = np.array([
        [-w/2, -l/2, -h/2],
        [ w/2, -l/2, -h/2],
        [ w/2,  l/2, -h/2],
        [-w/2,  l/2, -h/2],
        [-w/2, -l/2,  h/2],
        [ w/2, -l/2,  h/2],
        [ w/2,  l/2,  h/2],
        [-w/2,  l/2,  h/2],
    ], dtype=np.float32)

    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1],
    ], dtype=np.float32)

    corners_world = corners_local @ rotation_matrix.T
    corners_world[:, 0] += cx
    corners_world[:, 1] += cy
    corners_world[:, 2] += cz

    corners_homo = np.concatenate([
        corners_world, np.ones((8, 1), dtype=np.float32)
    ], axis=1)

    corners_cam = (extrinsic @ corners_homo.T).T
    corners_cam_3d = corners_cam[:, :3]

    if np.any(corners_cam_3d[:, 2] <= 0):
        return None

    corners_proj = (intrinsic @ corners_cam_3d.T).T
    corners_2d = corners_proj[:, :2] / corners_proj[:, 2:3]

    return corners_2d


def draw_3d_box_on_image(image, corners_2d, color, thickness=2):
    """Draw projected 3D bounding box on image."""
    corners = corners_2d.astype(np.int32)

    for i in range(4):
        cv2.line(image,
                 tuple(corners[i]), tuple(corners[(i + 1) % 4]),
                 color, thickness)

    for i in range(4):
        cv2.line(image,
                 tuple(corners[i + 4]), tuple(corners[(i + 1) % 4 + 4]),
                 color, thickness)

    for i in range(4):
        cv2.line(image,
                 tuple(corners[i]), tuple(corners[i + 4]),
                 color, thickness)

    return image


def visualize_predictions(images, predictions, intrinsics, extrinsics,
                          score_threshold=0.3, output_dir='./inference_output'):
    """
    Visualize 3D box predictions projected onto camera images.

    Args:
        images: (6, H, W, 3) camera images
        predictions: dict with 'boxes', 'scores', 'labels'
        intrinsics: (6, 3, 3) camera intrinsics
        extrinsics: (6, 4, 4) camera extrinsics
        score_threshold: minimum score to display
        output_dir: directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    fig, axes = plt.subplots(2, 3, figsize=(48, 18))
    axes = axes.flatten()

    for cam_idx in range(6):
        ax = axes[cam_idx]
        img = images[cam_idx].copy().astype(np.uint8)
        intrinsic = intrinsics[cam_idx]
        extrinsic = extrinsics[cam_idx]

        num_drawn = 0
        for box_idx in range(len(boxes)):
            corners_2d = project_3d_box_to_image(boxes[box_idx], intrinsic, extrinsic)
            if corners_2d is None:
                continue

            h, w = img.shape[:2]
            if np.all(corners_2d[:, 0] < 0) or np.all(corners_2d[:, 0] > w):
                continue
            if np.all(corners_2d[:, 1] < 0) or np.all(corners_2d[:, 1] > h):
                continue

            color = CLASS_COLORS[labels[box_idx] % len(CLASS_COLORS)]
            img = draw_3d_box_on_image(img, corners_2d, color, thickness=2)

            label_text = f'{NUSCENES_CLASSES[labels[box_idx]]}: {scores[box_idx]:.2f}'
            text_pos = (int(np.min(corners_2d[:, 0])), int(np.min(corners_2d[:, 1])) - 5)
            text_pos = (max(0, text_pos[0]), max(15, text_pos[1]))
            cv2.putText(img, label_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            num_drawn += 1

        ax.imshow(img)
        ax.set_title(f'{CAMERA_NAMES[cam_idx]} ({num_drawn} detections)', fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'detr3d_predictions_multiview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Multi-view visualization saved to: {output_path}')

    for cam_idx in range(6):
        fig_single, ax_single = plt.subplots(1, 1, figsize=(16, 9))
        img = images[cam_idx].copy().astype(np.uint8)
        intrinsic = intrinsics[cam_idx]
        extrinsic = extrinsics[cam_idx]

        for box_idx in range(len(boxes)):
            corners_2d = project_3d_box_to_image(boxes[box_idx], intrinsic, extrinsic)
            if corners_2d is None:
                continue

            h, w = img.shape[:2]
            if np.all(corners_2d[:, 0] < 0) or np.all(corners_2d[:, 0] > w):
                continue
            if np.all(corners_2d[:, 1] < 0) or np.all(corners_2d[:, 1] > h):
                continue

            color = CLASS_COLORS[labels[box_idx] % len(CLASS_COLORS)]
            img = draw_3d_box_on_image(img, corners_2d, color, thickness=2)

            label_text = f'{NUSCENES_CLASSES[labels[box_idx]]}: {scores[box_idx]:.2f}'
            text_pos = (int(np.min(corners_2d[:, 0])), int(np.min(corners_2d[:, 1])) - 5)
            text_pos = (max(0, text_pos[0]), max(15, text_pos[1]))
            cv2.putText(img, label_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ax_single.imshow(img)
        ax_single.set_title(f'{CAMERA_NAMES[cam_idx]}', fontsize=16)
        ax_single.axis('off')

        cam_output_path = os.path.join(output_dir, f'detr3d_{CAMERA_NAMES[cam_idx].lower()}.png')
        plt.savefig(cam_output_path, dpi=150, bbox_inches='tight')
        plt.close()

    bev_fig, bev_ax = plt.subplots(1, 1, figsize=(10, 10))
    bev_ax.set_xlim(-50, 50)
    bev_ax.set_ylim(-50, 50)
    bev_ax.set_aspect('equal')
    bev_ax.set_xlabel('X (m)')
    bev_ax.set_ylabel('Y (m)')
    bev_ax.set_title('Bird\'s Eye View Detections')
    bev_ax.grid(True, alpha=0.3)

    bev_ax.plot(0, 0, 'k^', markersize=15, label='Ego Vehicle')

    for box_idx in range(len(boxes)):
        cx, cy = boxes[box_idx, 0], boxes[box_idx, 1]
        w, l = boxes[box_idx, 3], boxes[box_idx, 4]
        sin_yaw, cos_yaw = boxes[box_idx, 6], boxes[box_idx, 7]
        yaw = np.arctan2(sin_yaw, cos_yaw)

        color_rgb = np.array(CLASS_COLORS[labels[box_idx] % len(CLASS_COLORS)]) / 255.0

        corners = np.array([
            [-w/2, -l/2],
            [ w/2, -l/2],
            [ w/2,  l/2],
            [-w/2,  l/2],
        ])

        rot = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)],
        ])
        corners = corners @ rot.T
        corners[:, 0] += cx
        corners[:, 1] += cy

        polygon = plt.Polygon(corners, fill=False, edgecolor=color_rgb,
                              linewidth=2, alpha=0.8)
        bev_ax.add_patch(polygon)

        if boxes[box_idx].shape[0] >= 10:
            vx, vy = boxes[box_idx, 8], boxes[box_idx, 9]
            vel_mag = np.sqrt(vx**2 + vy**2)
            if vel_mag > 0.5:
                bev_ax.arrow(cx, cy, vx, vy, head_width=0.3,
                             head_length=0.2, fc=color_rgb, ec=color_rgb, alpha=0.6)

    legend_handles = []
    for cls_idx, cls_name in enumerate(NUSCENES_CLASSES):
        if np.any(labels == cls_idx):
            color_rgb = np.array(CLASS_COLORS[cls_idx]) / 255.0
            legend_handles.append(
                patches.Patch(facecolor=color_rgb, edgecolor=color_rgb, label=cls_name)
            )
    if legend_handles:
        bev_ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    bev_output_path = os.path.join(output_dir, 'detr3d_bev.png')
    plt.savefig(bev_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'BEV visualization saved to: {bev_output_path}')


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Building DETR3D model...')
    model = build_detr3d(
        num_classes=NUM_CLASSES,
        num_queries=args.num_queries,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_decoder_layers=args.num_decoder_layers,
    )

    dummy_input = {
        'images': tf.zeros([1, 6, args.img_h, args.img_w, 3]),
        'intrinsics': tf.zeros([1, 6, 3, 3]),
        'extrinsics': tf.zeros([1, 6, 4, 4]),
    }
    _ = model(dummy_input, training=False)

    print('Loading checkpoint...')
    if args.checkpoint.endswith('.h5') or args.checkpoint.endswith('.weights.h5'):
        model.load_weights(args.checkpoint)
        print(f'Loaded weights: {args.checkpoint}')
    else:
        checkpoint = tf.train.Checkpoint(model=model)
        latest = tf.train.latest_checkpoint(args.checkpoint)
        if latest:
            checkpoint.restore(latest).expect_partial()
            print(f'Restored: {latest}')
        else:
            print(f'ERROR: No checkpoint found at {args.checkpoint}')
            return

    print('Loading sample...')
    if args.sample_path:
        sample = load_sample_from_json(args.sample_path, args.img_h, args.img_w)
    elif args.data_root:
        sample = load_sample_from_nuscenes(
            args.data_root, args.sample_idx, args.img_h, args.img_w
        )
    else:
        print('ERROR: Must provide either --sample_path or --data_root')
        return

    print('Running inference...')
    model_inputs = {
        'images': tf.expand_dims(tf.constant(sample['images'], dtype=tf.float32), 0),
        'intrinsics': tf.expand_dims(tf.constant(sample['intrinsics'], dtype=tf.float32), 0),
        'extrinsics': tf.expand_dims(tf.constant(sample['extrinsics'], dtype=tf.float32), 0),
    }

    outputs = model(model_inputs, training=False)

    cls_logits = outputs['cls_logits'][0].numpy()
    reg_preds = outputs['reg_preds'][0].numpy()

    cls_scores = 1.0 / (1.0 + np.exp(-cls_logits))
    max_scores = np.max(cls_scores, axis=-1)
    max_labels = np.argmax(cls_scores, axis=-1)

    valid_mask = max_scores > args.score_threshold
    pred_boxes = reg_preds[valid_mask]
    pred_scores = max_scores[valid_mask]
    pred_labels = max_labels[valid_mask]

    print(f'Detections before NMS: {len(pred_boxes)}')

    keep_indices = nms_3d(pred_boxes, pred_scores, pred_labels, args.nms_threshold)
    pred_boxes = pred_boxes[keep_indices]
    pred_scores = pred_scores[keep_indices]
    pred_labels = pred_labels[keep_indices]

    print(f'Detections after NMS: {len(pred_boxes)}')

    print('\nDetections:')
    print(f'{"Class":<25} {"Score":<8} {"X":<8} {"Y":<8} {"Z":<8} {"W":<6} {"L":<6} {"H":<6}')
    print('-' * 80)
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        print(
            f'{NUSCENES_CLASSES[pred_labels[i]]:<25} '
            f'{pred_scores[i]:<8.3f} '
            f'{box[0]:<8.2f} {box[1]:<8.2f} {box[2]:<8.2f} '
            f'{box[3]:<6.2f} {box[4]:<6.2f} {box[5]:<6.2f}'
        )

    predictions = {
        'boxes': pred_boxes,
        'scores': pred_scores,
        'labels': pred_labels,
    }

    print('\nGenerating visualizations...')
    visualize_predictions(
        sample['images'], predictions,
        sample['intrinsics'], sample['extrinsics'],
        score_threshold=args.score_threshold,
        output_dir=args.output_dir,
    )

    results_json = {
        'num_detections': int(len(pred_boxes)),
        'detections': [
            {
                'class': NUSCENES_CLASSES[int(pred_labels[i])],
                'score': float(pred_scores[i]),
                'box_3d': pred_boxes[i].tolist(),
            }
            for i in range(len(pred_boxes))
        ]
    }
    json_path = os.path.join(args.output_dir, 'detections.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f'Detection results saved to: {json_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
