"""
DETR3D Training Script
TensorFlow 2 Implementation with nuScenes dataset support.

Usage:
    python train.py --data_root /path/to/nuscenes --epochs 24 --batch_size 1
"""

import argparse
import json
import math
import os
import time

import numpy as np
import tensorflow as tf

from model import DETR3D, build_detr3d


# nuScenes category mapping
NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
NUM_CLASSES = len(NUSCENES_CLASSES)

CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]


def parse_args():
    parser = argparse.ArgumentParser(description='DETR3D Training')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--clip_norm', type=float, default=35.0)
    parser.add_argument('--num_queries', type=int, default=900)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--img_h', type=int, default=900)
    parser.add_argument('--img_w', type=int, default=1600)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=2,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=50,
                        help='Log every N steps')
    return parser.parse_args()


class NuScenesDataset:
    """nuScenes dataset loader for DETR3D training."""

    def __init__(self, data_root, split='train', img_h=900, img_w=1600,
                 augment=True):
        self.data_root = data_root
        self.split = split
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.samples = self._load_annotations()

    def _load_annotations(self):
        """Load nuScenes annotations from JSON info files."""
        info_path = os.path.join(
            self.data_root, f'nuscenes_infos_{self.split}.json'
        )
        if not os.path.exists(info_path):
            info_path = os.path.join(
                self.data_root, f'nuscenes_infos_{self.split}.pkl'
            )
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

    def _load_image(self, img_path):
        """Load and preprocess a single image."""
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, [self.img_h, self.img_w])
        img = tf.cast(img, tf.float32)
        return img

    def _parse_sample(self, sample_info):
        """Parse a single nuScenes sample into model inputs."""
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

                cam_intrinsic = np.array(cam_info.get('cam_intrinsic',
                                         np.eye(3)), dtype=np.float32).reshape(3, 3)
                intrinsics.append(cam_intrinsic)

                lidar2cam = np.array(cam_info.get('lidar2cam',
                                     np.eye(4)), dtype=np.float32).reshape(4, 4)
                extrinsics.append(lidar2cam)
        else:
            for cam_name in CAMERA_NAMES:
                cam_info = cams.get(cam_name, {})
                img_path = os.path.join(self.data_root, cam_info.get('data_path', ''))
                images.append(img_path)

                cam_intrinsic = np.array(
                    cam_info.get('cam_intrinsic', np.eye(3)),
                    dtype=np.float32
                ).reshape(3, 3)
                intrinsics.append(cam_intrinsic)

                sensor2lidar_rotation = np.array(
                    cam_info.get('sensor2lidar_rotation', np.eye(3)),
                    dtype=np.float32
                ).reshape(3, 3)
                sensor2lidar_translation = np.array(
                    cam_info.get('sensor2lidar_translation', np.zeros(3)),
                    dtype=np.float32
                )

                sensor2lidar = np.eye(4, dtype=np.float32)
                sensor2lidar[:3, :3] = sensor2lidar_rotation
                sensor2lidar[:3, 3] = sensor2lidar_translation

                lidar2sensor = np.linalg.inv(sensor2lidar)
                extrinsics.append(lidar2sensor)

        gt_boxes = np.array(
            sample_info.get('gt_boxes', sample_info.get('ann_infos', {}).get('gt_boxes_3d', [])),
            dtype=np.float32
        )
        gt_names = sample_info.get('gt_names', sample_info.get('ann_infos', {}).get('gt_names', []))

        gt_labels = []
        valid_box_mask = []
        for name in gt_names:
            if name in NUSCENES_CLASSES:
                gt_labels.append(NUSCENES_CLASSES.index(name))
                valid_box_mask.append(True)
            else:
                valid_box_mask.append(False)

        valid_box_mask = np.array(valid_box_mask)
        if len(gt_boxes) > 0 and len(valid_box_mask) > 0:
            gt_boxes = gt_boxes[valid_box_mask]

        gt_labels = np.array(gt_labels, dtype=np.int64)

        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes.reshape(-1, 10) if len(gt_boxes) > 0 else np.zeros((0, 10), dtype=np.float32)
        if gt_boxes.shape[1] == 7:
            velocities = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)
            sin_yaw = np.sin(gt_boxes[:, 6:7])
            cos_yaw = np.cos(gt_boxes[:, 6:7])
            gt_boxes = np.concatenate([
                gt_boxes[:, :6], sin_yaw, cos_yaw, velocities
            ], axis=1)
        elif gt_boxes.shape[1] == 9:
            cos_yaw = np.cos(gt_boxes[:, 6:7])
            sin_yaw = np.sin(gt_boxes[:, 6:7])
            gt_boxes = np.concatenate([
                gt_boxes[:, :6], sin_yaw, cos_yaw, gt_boxes[:, 7:9]
            ], axis=1)

        return {
            'image_paths': images,
            'intrinsics': np.stack(intrinsics, axis=0),
            'extrinsics': np.stack(extrinsics, axis=0),
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._parse_sample(self.samples[idx])


def create_tf_dataset(nuscenes_dataset, batch_size, img_h, img_w, augment=True):
    """Create tf.data.Dataset pipeline from NuScenesDataset."""

    def generator():
        indices = np.arange(len(nuscenes_dataset))
        np.random.shuffle(indices)
        for idx in indices:
            sample = nuscenes_dataset[int(idx)]
            yield sample

    output_signature = {
        'image_paths': tf.TensorSpec(shape=(6,), dtype=tf.string),
        'intrinsics': tf.TensorSpec(shape=(6, 3, 3), dtype=tf.float32),
        'extrinsics': tf.TensorSpec(shape=(6, 4, 4), dtype=tf.float32),
        'gt_boxes': tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
        'gt_labels': tf.TensorSpec(shape=(None,), dtype=tf.int64),
    }

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    def load_and_preprocess(sample):
        """Load images and apply augmentations."""
        def load_single_image(path):
            img_raw = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img_raw, channels=3)
            img = tf.image.resize(img, [img_h, img_w])
            img = tf.cast(img, tf.float32)
            return img

        images = tf.map_fn(load_single_image, sample['image_paths'], fn_output_signature=tf.float32)

        if augment:
            do_flip = tf.random.uniform([]) > 0.5
            if do_flip:
                images = tf.reverse(images, axis=[2])
                intrinsics = sample['intrinsics']
                cx_update = tf.constant(img_w, dtype=tf.float32) - intrinsics[:, 0, 2]
                new_intrinsics = tf.identity(intrinsics)
                indices = tf.constant([[i, 0, 2] for i in range(6)])
                new_intrinsics = tf.tensor_scatter_nd_update(
                    new_intrinsics, indices, cx_update
                )
                sample['intrinsics'] = new_intrinsics

                gt_boxes = sample['gt_boxes']
                if tf.shape(gt_boxes)[0] > 0:
                    y_flipped = -gt_boxes[:, 1:2]
                    sin_flipped = -gt_boxes[:, 6:7]
                    vy_flipped = -gt_boxes[:, 9:10]
                    gt_boxes = tf.concat([
                        gt_boxes[:, 0:1], y_flipped, gt_boxes[:, 2:6],
                        sin_flipped, gt_boxes[:, 7:9], vy_flipped
                    ], axis=-1)
                    sample['gt_boxes'] = gt_boxes

            scale_factor = tf.random.uniform([], 0.9, 1.1)
            new_h = tf.cast(tf.cast(img_h, tf.float32) * scale_factor, tf.int32)
            new_w = tf.cast(tf.cast(img_w, tf.float32) * scale_factor, tf.int32)
            images = tf.image.resize(images, [new_h, new_w])
            images = tf.image.resize_with_crop_or_pad(images, img_h, img_w)

            intrinsics = sample['intrinsics']
            scale_matrix = tf.constant([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
            fx_scale = tf.concat([
                tf.expand_dims(scale_factor, 0),
                tf.constant([1.0, 1.0])
            ], axis=0)
            fy_scale = tf.concat([
                tf.constant([1.0]),
                tf.expand_dims(scale_factor, 0),
                tf.constant([1.0])
            ], axis=0)
            scale_diag = tf.concat([
                tf.expand_dims(scale_factor, 0),
                tf.expand_dims(scale_factor, 0),
                tf.constant([1.0])
            ], axis=0)
            scale_mat = tf.linalg.diag(scale_diag)
            sample['intrinsics'] = tf.einsum('ij,bjk->bik', scale_mat, intrinsics)

        sample['images'] = images
        del sample['image_paths']
        return sample

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={
            'images': [6, img_h, img_w, 3],
            'intrinsics': [6, 3, 3],
            'extrinsics': [6, 4, 4],
            'gt_boxes': [None, 10],
            'gt_labels': [None],
        },
        padding_values={
            'images': 0.0,
            'intrinsics': 0.0,
            'extrinsics': 0.0,
            'gt_boxes': 0.0,
            'gt_labels': tf.constant(-1, dtype=tf.int64),
        }
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay learning rate schedule with linear warmup."""

    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.base_lr * (step / tf.maximum(warmup_steps, 1.0))

        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(math.pi * progress))

        lr = tf.where(step < warmup_steps, warmup_lr, cosine_lr)
        return lr

    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
        }


def train_step(model, optimizer, batch, clip_norm):
    """Single training step with gradient computation and clipping."""
    model_inputs = {
        'images': batch['images'],
        'intrinsics': batch['intrinsics'],
        'extrinsics': batch['extrinsics'],
    }

    gt_labels_list = []
    gt_boxes_list = []
    batch_size = batch['gt_labels'].shape[0]

    for b in range(batch_size):
        labels_b = batch['gt_labels'][b]
        boxes_b = batch['gt_boxes'][b]
        valid = labels_b >= 0
        gt_labels_list.append(tf.boolean_mask(labels_b, valid))
        gt_boxes_list.append(tf.boolean_mask(boxes_b, valid))

    with tf.GradientTape() as tape:
        predictions = model(model_inputs, training=True)
        total_loss, loss_dict = model.compute_loss(
            predictions, gt_labels_list, gt_boxes_list
        )

    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients, grad_norm = tf.clip_by_global_norm(gradients, clip_norm)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss_dict['grad_norm'] = grad_norm
    return loss_dict


@tf.function(reduce_retracing=True)
def distributed_train_step(strategy, model, optimizer, batch, clip_norm):
    """Distributed training step for multi-GPU."""
    per_replica_losses = strategy.run(
        train_step, args=(model, optimizer, batch, clip_norm)
    )
    reduced_losses = {}
    for key, value in per_replica_losses.items():
        reduced_losses[key] = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, value, axis=None
        )
    return reduced_losses


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    nuscenes_train = NuScenesDataset(
        args.data_root, split='train',
        img_h=args.img_h, img_w=args.img_w, augment=True
    )

    num_samples = len(nuscenes_train)
    steps_per_epoch = num_samples // (args.batch_size * strategy.num_replicas_in_sync)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    print(f'Dataset size: {num_samples} samples')
    print(f'Steps per epoch: {steps_per_epoch}')
    print(f'Total steps: {total_steps}')
    print(f'Warmup steps: {warmup_steps}')

    train_dataset = create_tf_dataset(
        nuscenes_train, args.batch_size, args.img_h, args.img_w, augment=True
    )

    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    with strategy.scope():
        model = build_detr3d(
            num_classes=NUM_CLASSES,
            num_queries=args.num_queries,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_decoder_layers=args.num_decoder_layers,
        )

        lr_schedule = CosineDecayWithWarmup(
            base_lr=args.lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            clipnorm=args.clip_norm,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )

        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=optimizer
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, args.checkpoint_dir, max_to_keep=5
        )

        if args.resume:
            checkpoint.restore(args.resume)
            print(f'Resumed from checkpoint: {args.resume}')
        elif checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f'Restored from: {checkpoint_manager.latest_checkpoint}')

    summary_writer = tf.summary.create_file_writer(args.log_dir)

    global_step = 0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        epoch_start = time.time()
        epoch_losses = {'total_loss': 0.0, 'cls_loss': 0.0, 'reg_loss': 0.0}
        num_batches = 0

        for batch in dist_dataset:
            loss_dict = distributed_train_step(
                strategy, model, optimizer, batch, args.clip_norm
            )

            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += float(loss_dict[key])
            num_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                current_lr = float(lr_schedule(global_step))
                print(
                    f'  Step {global_step} | '
                    f'Loss: {float(loss_dict["total_loss"]):.4f} | '
                    f'Cls: {float(loss_dict["cls_loss"]):.4f} | '
                    f'Reg: {float(loss_dict["reg_loss"]):.4f} | '
                    f'LR: {current_lr:.6f} | '
                    f'Grad norm: {float(loss_dict.get("grad_norm", 0)):.2f}'
                )

                with summary_writer.as_default():
                    tf.summary.scalar('loss/total', loss_dict['total_loss'], step=global_step)
                    tf.summary.scalar('loss/cls', loss_dict['cls_loss'], step=global_step)
                    tf.summary.scalar('loss/reg', loss_dict['reg_loss'], step=global_step)
                    tf.summary.scalar('loss/aux', loss_dict.get('aux_loss', 0), step=global_step)
                    tf.summary.scalar('train/lr', current_lr, step=global_step)
                    tf.summary.scalar('train/grad_norm',
                                      loss_dict.get('grad_norm', 0), step=global_step)

        epoch_time = time.time() - epoch_start
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        print(
            f'  Epoch {epoch + 1} done in {epoch_time:.1f}s | '
            f'Avg Loss: {avg_losses["total_loss"]:.4f} | '
            f'Avg Cls: {avg_losses["cls_loss"]:.4f} | '
            f'Avg Reg: {avg_losses["reg_loss"]:.4f}'
        )

        if (epoch + 1) % args.save_every == 0:
            save_path = checkpoint_manager.save()
            print(f'  Checkpoint saved: {save_path}')

            model.save_weights(
                os.path.join(args.checkpoint_dir, f'detr3d_epoch_{epoch + 1}.weights.h5')
            )

    final_save = checkpoint_manager.save()
    print(f'\nTraining complete. Final checkpoint: {final_save}')

    model.save_weights(
        os.path.join(args.checkpoint_dir, 'detr3d_final.weights.h5')
    )
    print('Final weights saved.')


if __name__ == '__main__':
    main()
