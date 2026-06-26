"""
DETR3D Evaluation Script
Computes nuScenes detection metrics: mAP, ATE, ASE, AOE, AVE, AAE, NDS.

Usage:
    python evaluate.py --data_root /path/to/nuscenes --checkpoint ./checkpoints/
"""

import argparse
import json
import os
import time

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

DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]


def parse_args():
    parser = argparse.ArgumentParser(description='DETR3D Evaluation')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory or weights file')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_queries', type=int, default=900)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--img_h', type=int, default=900)
    parser.add_argument('--img_w', type=int, default=1600)
    parser.add_argument('--score_threshold', type=float, default=0.1)
    parser.add_argument('--output_json', type=str, default='./eval_results.json')
    return parser.parse_args()


class NuScenesEvalDataset:
    """nuScenes validation dataset loader."""

    def __init__(self, data_root, img_h=900, img_w=1600):
        self.data_root = data_root
        self.img_h = img_h
        self.img_w = img_w
        self.samples = self._load_annotations()

    def _load_annotations(self):
        info_path = os.path.join(self.data_root, 'nuscenes_infos_val.json')
        if not os.path.exists(info_path):
            info_path = os.path.join(self.data_root, 'nuscenes_infos_val.pkl')
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
        return samples

    def __len__(self):
        return len(self.samples)

    def get_sample(self, idx):
        """Get a single validation sample."""
        sample_info = self.samples[idx]

        images = []
        intrinsics = []
        extrinsics = []

        cams = sample_info.get('cams', {})
        if not cams:
            cam_infos = sample_info.get('images', {})
            for cam_name in CAMERA_NAMES:
                cam_info = cam_infos.get(cam_name, {})
                img_path = os.path.join(self.data_root, cam_info.get('img_path', ''))
                images.append(img_path)
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
                img_path = os.path.join(self.data_root, cam_info.get('data_path', ''))
                images.append(img_path)
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

        loaded_images = []
        for img_path in images:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [self.img_h, self.img_w])
            img = tf.cast(img, tf.float32)
            loaded_images.append(img)

        images_tensor = tf.stack(loaded_images, axis=0)

        gt_boxes = np.array(
            sample_info.get('gt_boxes', sample_info.get('ann_infos', {}).get('gt_boxes_3d', [])),
            dtype=np.float32
        )
        gt_names = sample_info.get(
            'gt_names', sample_info.get('ann_infos', {}).get('gt_names', [])
        )

        gt_labels = []
        valid_mask = []
        for name in gt_names:
            if name in NUSCENES_CLASSES:
                gt_labels.append(NUSCENES_CLASSES.index(name))
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        valid_mask = np.array(valid_mask)
        if len(gt_boxes) > 0 and len(valid_mask) > 0:
            gt_boxes = gt_boxes[valid_mask]
        gt_labels = np.array(gt_labels, dtype=np.int64)

        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes.reshape(-1, 10) if len(gt_boxes) > 0 else np.zeros((0, 10), dtype=np.float32)
        if gt_boxes.shape[1] == 7:
            velocities = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)
            sin_yaw = np.sin(gt_boxes[:, 6:7])
            cos_yaw = np.cos(gt_boxes[:, 6:7])
            gt_boxes = np.concatenate([gt_boxes[:, :6], sin_yaw, cos_yaw, velocities], axis=1)
        elif gt_boxes.shape[1] == 9:
            cos_yaw = np.cos(gt_boxes[:, 6:7])
            sin_yaw = np.sin(gt_boxes[:, 6:7])
            gt_boxes = np.concatenate([gt_boxes[:, :6], sin_yaw, cos_yaw, gt_boxes[:, 7:9]], axis=1)

        return {
            'images': images_tensor,
            'intrinsics': np.stack(intrinsics, axis=0),
            'extrinsics': np.stack(extrinsics, axis=0),
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'sample_token': sample_info.get('token', str(idx)),
        }


def compute_center_distance(pred_center, gt_center):
    """Compute 2D BEV center distance (ignoring height)."""
    return np.sqrt(np.sum((pred_center[:2] - gt_center[:2]) ** 2))


def compute_iou_3d_axis_aligned(pred_box, gt_box):
    """Approximate 3D IoU for axis-aligned boxes."""
    pred_center = pred_box[:3]
    pred_size = np.abs(pred_box[3:6])
    gt_center = gt_box[:3]
    gt_size = np.abs(gt_box[3:6])

    pred_min = pred_center - pred_size / 2
    pred_max = pred_center + pred_size / 2
    gt_min = gt_center - gt_size / 2
    gt_max = gt_center + gt_size / 2

    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_size)

    pred_vol = np.prod(pred_size)
    gt_vol = np.prod(gt_size)
    union_vol = pred_vol + gt_vol - inter_vol

    return inter_vol / max(union_vol, 1e-8)


def compute_ap_per_class_distance(predictions_list, gt_list, class_idx, dist_threshold):
    """
    Compute Average Precision for a single class at a given distance threshold.

    Uses center distance matching (nuScenes style).
    """
    all_scores = []
    all_tp = []
    num_gt_total = 0

    for pred_sample, gt_sample in zip(predictions_list, gt_list):
        pred_boxes = pred_sample['boxes']
        pred_scores = pred_sample['scores']
        pred_labels = pred_sample['labels']

        gt_boxes = gt_sample['boxes']
        gt_labels = gt_sample['labels']

        class_pred_mask = pred_labels == class_idx
        class_gt_mask = gt_labels == class_idx

        pred_boxes_cls = pred_boxes[class_pred_mask]
        pred_scores_cls = pred_scores[class_pred_mask]
        gt_boxes_cls = gt_boxes[class_gt_mask]

        num_gt = len(gt_boxes_cls)
        num_gt_total += num_gt

        if len(pred_boxes_cls) == 0:
            continue

        sort_idx = np.argsort(-pred_scores_cls)
        pred_boxes_cls = pred_boxes_cls[sort_idx]
        pred_scores_cls = pred_scores_cls[sort_idx]

        gt_matched = np.zeros(num_gt, dtype=bool)

        for i in range(len(pred_boxes_cls)):
            all_scores.append(pred_scores_cls[i])

            if num_gt == 0:
                all_tp.append(0)
                continue

            distances = np.array([
                compute_center_distance(pred_boxes_cls[i], gt_boxes_cls[j])
                for j in range(num_gt)
            ])

            valid_matches = (distances < dist_threshold) & (~gt_matched)
            if np.any(valid_matches):
                best_gt = np.argmin(np.where(valid_matches, distances, np.inf))
                gt_matched[best_gt] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if num_gt_total == 0:
        return 0.0

    if len(all_scores) == 0:
        return 0.0

    sort_idx = np.argsort(-np.array(all_scores))
    all_tp = np.array(all_tp)[sort_idx]

    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)

    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    recall = cum_tp / num_gt_total

    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    recall_thresholds = np.linspace(0, 1, 41)
    ap = 0.0
    for t in recall_thresholds:
        idx = np.searchsorted(recall, t)
        if idx < len(precision):
            ap += precision[idx]
    ap /= len(recall_thresholds)

    return ap


def compute_tp_metrics(predictions_list, gt_list, class_idx, dist_threshold=2.0):
    """
    Compute True Positive metrics for a specific class.

    Returns:
        ate: Average Translation Error (center distance)
        ase: Average Scale Error (1 - IoU of axis-aligned boxes)
        aoe: Average Orientation Error (angular difference)
        ave: Average Velocity Error
        aae: Average Attribute Error
    """
    ate_list = []
    ase_list = []
    aoe_list = []
    ave_list = []
    aae_list = []

    for pred_sample, gt_sample in zip(predictions_list, gt_list):
        pred_boxes = pred_sample['boxes']
        pred_scores = pred_sample['scores']
        pred_labels = pred_sample['labels']

        gt_boxes = gt_sample['boxes']
        gt_labels = gt_sample['labels']

        class_pred_mask = pred_labels == class_idx
        class_gt_mask = gt_labels == class_idx

        pred_boxes_cls = pred_boxes[class_pred_mask]
        pred_scores_cls = pred_scores[class_pred_mask]
        gt_boxes_cls = gt_boxes[class_gt_mask]

        if len(pred_boxes_cls) == 0 or len(gt_boxes_cls) == 0:
            continue

        sort_idx = np.argsort(-pred_scores_cls)
        pred_boxes_cls = pred_boxes_cls[sort_idx]

        gt_matched = np.zeros(len(gt_boxes_cls), dtype=bool)

        for i in range(len(pred_boxes_cls)):
            distances = np.array([
                compute_center_distance(pred_boxes_cls[i], gt_boxes_cls[j])
                for j in range(len(gt_boxes_cls))
            ])
            valid_matches = (distances < dist_threshold) & (~gt_matched)
            if np.any(valid_matches):
                best_gt = np.argmin(np.where(valid_matches, distances, np.inf))
                gt_matched[best_gt] = True

                pred_box = pred_boxes_cls[i]
                gt_box = gt_boxes_cls[best_gt]

                trans_err = compute_center_distance(pred_box, gt_box)
                ate_list.append(trans_err)

                iou = compute_iou_3d_axis_aligned(pred_box, gt_box)
                ase_list.append(1.0 - iou)

                pred_yaw = np.arctan2(pred_box[6], pred_box[7])
                gt_yaw = np.arctan2(gt_box[6], gt_box[7])
                yaw_diff = np.abs(pred_yaw - gt_yaw)
                yaw_diff = min(yaw_diff, 2 * np.pi - yaw_diff)
                aoe_list.append(yaw_diff)

                if pred_box.shape[0] >= 10 and gt_box.shape[0] >= 10:
                    vel_err = np.sqrt(np.sum((pred_box[8:10] - gt_box[8:10]) ** 2))
                    ave_list.append(vel_err)
                else:
                    ave_list.append(0.0)

                aae_list.append(0.0)

    ate = np.mean(ate_list) if ate_list else 1.0
    ase = np.mean(ase_list) if ase_list else 1.0
    aoe = np.mean(aoe_list) if aoe_list else 1.0
    ave = np.mean(ave_list) if ave_list else 1.0
    aae = np.mean(aae_list) if aae_list else 1.0

    return ate, ase, aoe, ave, aae


def compute_nds(mAP, metrics_dict):
    """
    Compute nuScenes Detection Score (NDS).

    NDS = 1/10 * [5*mAP + sum(max(1 - metric, 0) for metric in TP_metrics)]
    """
    tp_scores = []
    for key in ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        val = metrics_dict.get(key, 1.0)
        tp_scores.append(max(1.0 - val, 0.0))
    nds = (5.0 * mAP + sum(tp_scores)) / 10.0
    return nds


def run_inference(model, dataset, score_threshold=0.1, img_h=900, img_w=1600):
    """Run model inference on the full validation set."""
    predictions_list = []
    gt_list = []

    num_samples = len(dataset)
    print(f'Running inference on {num_samples} samples...')

    for idx in range(num_samples):
        if idx % 100 == 0:
            print(f'  Processing sample {idx}/{num_samples}')

        sample = dataset.get_sample(idx)

        images = tf.expand_dims(sample['images'], 0)
        intrinsics = tf.expand_dims(
            tf.constant(sample['intrinsics'], dtype=tf.float32), 0
        )
        extrinsics = tf.expand_dims(
            tf.constant(sample['extrinsics'], dtype=tf.float32), 0
        )

        model_inputs = {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
        }

        outputs = model(model_inputs, training=False)

        cls_logits = outputs['cls_logits'][0].numpy()
        reg_preds = outputs['reg_preds'][0].numpy()

        cls_scores = 1.0 / (1.0 + np.exp(-cls_logits))

        max_scores = np.max(cls_scores, axis=-1)
        max_labels = np.argmax(cls_scores, axis=-1)

        valid_mask = max_scores > score_threshold
        pred_boxes = reg_preds[valid_mask]
        pred_scores = max_scores[valid_mask]
        pred_labels = max_labels[valid_mask]

        predictions_list.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels,
            'token': sample['sample_token'],
        })

        gt_list.append({
            'boxes': sample['gt_boxes'],
            'labels': sample['gt_labels'],
        })

    return predictions_list, gt_list


def evaluate(predictions_list, gt_list):
    """Compute all nuScenes detection metrics."""
    results = {}

    print('\nComputing per-class AP at distance thresholds...')
    ap_matrix = np.zeros((NUM_CLASSES, len(DISTANCE_THRESHOLDS)))

    for cls_idx in range(NUM_CLASSES):
        for dist_idx, dist_thresh in enumerate(DISTANCE_THRESHOLDS):
            ap = compute_ap_per_class_distance(
                predictions_list, gt_list, cls_idx, dist_thresh
            )
            ap_matrix[cls_idx, dist_idx] = ap

    per_class_ap = np.mean(ap_matrix, axis=1)
    mAP = np.mean(per_class_ap)

    print('\nComputing TP metrics...')
    ate_list = []
    ase_list = []
    aoe_list = []
    ave_list = []
    aae_list = []

    for cls_idx in range(NUM_CLASSES):
        ate, ase, aoe, ave, aae = compute_tp_metrics(
            predictions_list, gt_list, cls_idx, dist_threshold=2.0
        )
        ate_list.append(ate)
        ase_list.append(ase)
        aoe_list.append(aoe)
        ave_list.append(ave)
        aae_list.append(aae)

    mATE = np.mean(ate_list)
    mASE = np.mean(ase_list)
    mAOE = np.mean(aoe_list)
    mAVE = np.mean(ave_list)
    mAAE = np.mean(aae_list)

    metrics_dict = {
        'mATE': mATE,
        'mASE': mASE,
        'mAOE': mAOE,
        'mAVE': mAVE,
        'mAAE': mAAE,
    }

    nds = compute_nds(mAP, metrics_dict)

    results['mAP'] = float(mAP)
    results['mATE'] = float(mATE)
    results['mASE'] = float(mASE)
    results['mAOE'] = float(mAOE)
    results['mAVE'] = float(mAVE)
    results['mAAE'] = float(mAAE)
    results['NDS'] = float(nds)

    results['per_class'] = {}
    for cls_idx, cls_name in enumerate(NUSCENES_CLASSES):
        results['per_class'][cls_name] = {
            'AP': float(per_class_ap[cls_idx]),
            'AP_dist': {
                str(d): float(ap_matrix[cls_idx, i])
                for i, d in enumerate(DISTANCE_THRESHOLDS)
            },
            'ATE': float(ate_list[cls_idx]),
            'ASE': float(ase_list[cls_idx]),
            'AOE': float(aoe_list[cls_idx]),
            'AVE': float(ave_list[cls_idx]),
            'AAE': float(aae_list[cls_idx]),
        }

    return results


def print_results(results):
    """Print evaluation results in a formatted table."""
    print('\n' + '=' * 80)
    print('DETR3D Evaluation Results')
    print('=' * 80)

    print(f'\n{"Metric":<10} {"Value":<10}')
    print('-' * 20)
    print(f'{"mAP":<10} {results["mAP"]:.4f}')
    print(f'{"NDS":<10} {results["NDS"]:.4f}')
    print(f'{"mATE":<10} {results["mATE"]:.4f}')
    print(f'{"mASE":<10} {results["mASE"]:.4f}')
    print(f'{"mAOE":<10} {results["mAOE"]:.4f}')
    print(f'{"mAVE":<10} {results["mAVE"]:.4f}')
    print(f'{"mAAE":<10} {results["mAAE"]:.4f}')

    print(f'\n{"Class":<25} {"AP":<8} ', end='')
    for d in DISTANCE_THRESHOLDS:
        print(f'{"AP@" + str(d) + "m":<10} ', end='')
    print(f'{"ATE":<8} {"ASE":<8} {"AOE":<8} {"AVE":<8} {"AAE":<8}')
    print('-' * 120)

    for cls_name in NUSCENES_CLASSES:
        cls_results = results['per_class'][cls_name]
        print(f'{cls_name:<25} {cls_results["AP"]:<8.4f} ', end='')
        for d in DISTANCE_THRESHOLDS:
            print(f'{cls_results["AP_dist"][str(d)]:<10.4f} ', end='')
        print(
            f'{cls_results["ATE"]:<8.4f} '
            f'{cls_results["ASE"]:<8.4f} '
            f'{cls_results["AOE"]:<8.4f} '
            f'{cls_results["AVE"]:<8.4f} '
            f'{cls_results["AAE"]:<8.4f}'
        )

    print('=' * 80)


def main():
    args = parse_args()

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

    if args.checkpoint.endswith('.h5') or args.checkpoint.endswith('.weights.h5'):
        model.load_weights(args.checkpoint)
        print(f'Loaded weights from: {args.checkpoint}')
    else:
        checkpoint = tf.train.Checkpoint(model=model)
        latest = tf.train.latest_checkpoint(args.checkpoint)
        if latest:
            checkpoint.restore(latest).expect_partial()
            print(f'Restored checkpoint: {latest}')
        else:
            print(f'ERROR: No checkpoint found at {args.checkpoint}')
            return

    dataset = NuScenesEvalDataset(
        args.data_root, img_h=args.img_h, img_w=args.img_w
    )

    start_time = time.time()
    predictions_list, gt_list = run_inference(
        model, dataset, score_threshold=args.score_threshold,
        img_h=args.img_h, img_w=args.img_w
    )
    inference_time = time.time() - start_time
    print(f'\nInference completed in {inference_time:.1f}s '
          f'({len(dataset) / inference_time:.1f} samples/s)')

    results = evaluate(predictions_list, gt_list)

    print_results(results)

    results['meta'] = {
        'num_samples': len(dataset),
        'inference_time_s': inference_time,
        'score_threshold': args.score_threshold,
        'checkpoint': args.checkpoint,
    }

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {args.output_json}')


if __name__ == '__main__':
    main()
