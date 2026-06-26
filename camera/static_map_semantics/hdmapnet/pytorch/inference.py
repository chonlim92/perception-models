"""
Inference script for HDMapNet.

Loads a trained model checkpoint, runs inference on samples,
applies post-processing (vectorization), and visualizes results.
"""

import os
import argparse
import yaml
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from .model import HDMapNet
from .postprocess import vectorize_predictions


# Visualization colors for each class (BGR for OpenCV, RGB for matplotlib)
CLASS_COLORS_RGB = {
    0: (255, 0, 0),      # divider - red
    1: (0, 255, 0),      # boundary - green
    2: (0, 0, 255),      # crossing - blue
}

CLASS_NAMES = ["divider", "boundary", "crossing"]


def load_model(checkpoint_path, config, device):
    """Load a trained HDMapNet model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        config: Model configuration dict.
        device: Torch device.

    Returns:
        Loaded model in eval mode.
    """
    model_config = {
        "backbone": config.get("backbone", "efficientnet-b0"),
        "pretrained_backbone": False,  # Don't need pretrained when loading checkpoint
        "backbone_out_channels": config.get("backbone_out_channels", 64),
        "view_transform": config.get("view_transform", "lss"),
        "xbound": config.get("xbound", [-30.0, 30.0, 0.3]),
        "ybound": config.get("ybound", [-15.0, 15.0, 0.3]),
        "zbound": config.get("zbound", [-10.0, 10.0, 20.0]),
        "dbound": config.get("dbound", [4.0, 45.0, 1.0]),
        "image_size": config.get("image_size", [128, 352]),
        "num_classes": config.get("num_classes", 3),
        "embedding_dim": config.get("embedding_dim", 16),
        "bev_encoder_base_channels": config.get("bev_encoder_base_channels", 64),
        "head_mid_channels": config.get("head_mid_channels", 64),
    }

    model = HDMapNet(model_config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def preprocess_images(image_paths, image_size=(128, 352)):
    """Load and preprocess camera images for inference.

    Args:
        image_paths: List of 6 image file paths (one per camera).
        image_size: Target (H, W).

    Returns:
        images: (1, 6, 3, H, W) tensor, normalized.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensors = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
        tensor = TF.to_tensor(img)
        tensor = normalize(tensor)
        img_tensors.append(tensor)

    images = torch.stack(img_tensors, dim=0).unsqueeze(0)  # (1, 6, 3, H, W)
    return images


def preprocess_calibration(intrinsics_list, extrinsics_list, orig_size, target_size):
    """Preprocess camera calibration matrices.

    Args:
        intrinsics_list: List of 6 intrinsic matrices (3x3 numpy arrays).
        extrinsics_list: List of 6 extrinsic matrices (4x4 numpy arrays).
        orig_size: Original image size (H, W).
        target_size: Target image size (H, W).

    Returns:
        intrinsics: (1, 6, 3, 3) tensor.
        extrinsics: (1, 6, 4, 4) tensor.
    """
    sx = target_size[1] / orig_size[1]
    sy = target_size[0] / orig_size[0]

    adjusted_intrinsics = []
    for K in intrinsics_list:
        K_adj = K.copy()
        K_adj[0, 0] *= sx
        K_adj[0, 2] *= sx
        K_adj[1, 1] *= sy
        K_adj[1, 2] *= sy
        adjusted_intrinsics.append(K_adj)

    intrinsics = torch.from_numpy(np.stack(adjusted_intrinsics, axis=0)).float().unsqueeze(0)
    extrinsics = torch.from_numpy(np.stack(extrinsics_list, axis=0)).float().unsqueeze(0)

    return intrinsics, extrinsics


@torch.no_grad()
def run_inference(model, images, intrinsics, extrinsics, device):
    """Run model inference on a batch.

    Args:
        model: HDMapNet model in eval mode.
        images: (B, N, 3, H, W) tensor.
        intrinsics: (B, N, 3, 3) tensor.
        extrinsics: (B, N, 4, 4) tensor.
        device: Torch device.

    Returns:
        predictions: Dict with 'semantic', 'instance', 'direction' tensors.
    """
    images = images.to(device)
    intrinsics = intrinsics.to(device)
    extrinsics = extrinsics.to(device)

    predictions = model(images, intrinsics, extrinsics)
    return predictions


def postprocess_predictions(predictions, config):
    """Apply post-processing to convert dense predictions to polylines.

    Args:
        predictions: Model output dict for a single sample.
        config: Configuration dict.

    Returns:
        vectorized: Dict mapping class_id to list of polylines in metric coords.
        semantic_masks: Dict mapping class_id to binary mask numpy array.
    """
    xbound = config.get("xbound", [-30.0, 30.0, 0.3])
    ybound = config.get("ybound", [-15.0, 15.0, 0.3])

    sem_logits = predictions["semantic"][0].cpu()    # (C, H, W)
    inst_emb = predictions["instance"][0].cpu()      # (E, H, W)
    direction = predictions["direction"][0].cpu()    # (2, H, W)

    sem_prob = torch.sigmoid(sem_logits).numpy()
    inst_emb_np = inst_emb.numpy()
    dir_np = direction.numpy()

    # Binary masks per class
    semantic_masks = {}
    for c in range(sem_prob.shape[0]):
        semantic_masks[c] = (sem_prob[c] > 0.5).astype(np.uint8)

    # Vectorize
    vectorized = vectorize_predictions(
        sem_prob, inst_emb_np, dir_np,
        semantic_threshold=0.5,
        dbscan_eps=1.5,
        dbscan_min_samples=5,
        nms_threshold=5.0,
        sample_spacing=2,
        xbound=xbound,
        ybound=ybound,
    )

    return vectorized, semantic_masks


def visualize_bev(semantic_masks, vectorized, config, output_path=None):
    """Visualize BEV predictions.

    Draws semantic masks as colored overlays and polylines on a BEV canvas.

    Args:
        semantic_masks: Dict mapping class_id to (H, W) binary numpy array.
        vectorized: Dict mapping class_id to list of polylines.
        config: Configuration dict.
        output_path: Optional path to save the visualization. If None, displays.

    Returns:
        BEV visualization as numpy array (H, W, 3) uint8.
    """
    xbound = config.get("xbound", [-30.0, 30.0, 0.3])
    ybound = config.get("ybound", [-15.0, 15.0, 0.3])

    bev_h = int((ybound[1] - ybound[0]) / ybound[2])
    bev_w = int((xbound[1] - xbound[0]) / xbound[2])

    # Create canvas
    canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    # Draw semantic masks
    for cls_id, mask in semantic_masks.items():
        color = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
        for ch in range(3):
            canvas[:, :, ch] = np.where(mask > 0, color[ch], canvas[:, :, ch])

    # Draw polylines
    for cls_id, polylines in vectorized.items():
        color = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
        for poly in polylines:
            # Convert metric coords back to pixels for drawing
            pixel_poly = np.zeros_like(poly)
            pixel_poly[:, 0] = (poly[:, 0] - ybound[0]) / ybound[2]  # row
            pixel_poly[:, 1] = (poly[:, 1] - xbound[0]) / xbound[2]  # col

            # Draw line segments
            for i in range(len(pixel_poly) - 1):
                r0, c0 = int(pixel_poly[i, 0]), int(pixel_poly[i, 1])
                r1, c1 = int(pixel_poly[i + 1, 0]), int(pixel_poly[i + 1, 1])
                _draw_line(canvas, r0, c0, r1, c1, color, thickness=2)

            # Draw direction arrow at midpoint
            if len(pixel_poly) >= 2:
                mid_idx = len(pixel_poly) // 2
                r_mid = int(pixel_poly[mid_idx, 0])
                c_mid = int(pixel_poly[mid_idx, 1])
                if mid_idx + 1 < len(pixel_poly):
                    dr = pixel_poly[mid_idx + 1, 0] - pixel_poly[mid_idx, 0]
                    dc = pixel_poly[mid_idx + 1, 1] - pixel_poly[mid_idx, 1]
                    length = np.sqrt(dr * dr + dc * dc)
                    if length > 0:
                        dr, dc = dr / length * 5, dc / length * 5
                        r_end = int(r_mid + dr)
                        c_end = int(c_mid + dc)
                        _draw_line(canvas, r_mid, c_mid, r_end, c_end, (255, 255, 0), thickness=1)

    # Draw ego vehicle position (center of BEV)
    ego_r = bev_h // 2
    ego_c = bev_w // 2
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = ego_r + dr, ego_c + dc
            if 0 <= r < bev_h and 0 <= c < bev_w:
                canvas[r, c] = [255, 255, 255]

    if output_path:
        img = Image.fromarray(canvas)
        img.save(output_path)
        print(f"Saved BEV visualization to: {output_path}")

    return canvas


def _draw_line(canvas, r0, c0, r1, c1, color, thickness=1):
    """Draw a line on the canvas using Bresenham's algorithm.

    Args:
        canvas: (H, W, 3) numpy array.
        r0, c0: Start point (row, col).
        r1, c1: End point (row, col).
        color: (R, G, B) tuple.
        thickness: Line thickness in pixels.
    """
    H, W = canvas.shape[:2]
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    half_t = thickness // 2

    while True:
        for dt_r in range(-half_t, half_t + 1):
            for dt_c in range(-half_t, half_t + 1):
                pr, pc = r0 + dt_r, c0 + dt_c
                if 0 <= pr < H and 0 <= pc < W:
                    canvas[pr, pc] = color

        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


def visualize_cameras(image_paths, predictions, config, output_path=None):
    """Create a grid visualization of all camera images with projected results.

    Args:
        image_paths: List of 6 camera image paths.
        predictions: Model output dict (unused for now, for future projection).
        config: Configuration dict.
        output_path: Optional output path.

    Returns:
        Grid image as numpy array.
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((352, 128))
        images.append(np.array(img))

    # Arrange in 2x3 grid
    # Top row: FRONT_LEFT, FRONT, FRONT_RIGHT
    # Bottom row: BACK_LEFT, BACK, BACK_RIGHT
    # Camera order in our convention: FRONT, FRONT_RIGHT, BACK_RIGHT, BACK, BACK_LEFT, FRONT_LEFT
    grid_order = [5, 0, 1, 4, 3, 2]  # FRONT_LEFT, FRONT, FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT

    H, W = 128, 352
    grid = np.zeros((2 * H, 3 * W, 3), dtype=np.uint8)

    for i, cam_idx in enumerate(grid_order):
        row = i // 3
        col = i % 3
        grid[row * H: (row + 1) * H, col * W: (col + 1) * W] = images[cam_idx]

    if output_path:
        img = Image.fromarray(grid)
        img.save(output_path)
        print(f"Saved camera grid to: {output_path}")

    return grid


def run_single_sample(model, sample_data, config, device, output_dir=None):
    """Run inference on a single sample and optionally save visualizations.

    Args:
        model: HDMapNet model.
        sample_data: Dict with 'images', 'intrinsics', 'extrinsics' tensors and
                     optionally 'image_paths' list.
        config: Configuration dict.
        device: Torch device.
        output_dir: Optional directory to save outputs.

    Returns:
        Dict with 'predictions', 'vectorized', 'semantic_masks'.
    """
    images = sample_data["images"].to(device)          # (1, N, 3, H, W)
    intrinsics = sample_data["intrinsics"].to(device)  # (1, N, 3, 3)
    extrinsics = sample_data["extrinsics"].to(device)  # (1, N, 4, 4)

    # Run inference
    predictions = run_inference(model, images, intrinsics, extrinsics, device)

    # Post-process
    vectorized, semantic_masks = postprocess_predictions(predictions, config)

    # Print summary
    print("\nInference Results:")
    print("-" * 40)
    for cls_id in range(config.get("num_classes", 3)):
        num_polys = len(vectorized.get(cls_id, []))
        mask_coverage = semantic_masks.get(cls_id, np.zeros(1)).sum()
        print(f"  {CLASS_NAMES[cls_id]}: {num_polys} polylines, {mask_coverage} mask pixels")

    # Visualize
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        bev_path = os.path.join(output_dir, "bev_prediction.png")
        visualize_bev(semantic_masks, vectorized, config, output_path=bev_path)

        if "image_paths" in sample_data:
            cam_path = os.path.join(output_dir, "camera_grid.png")
            visualize_cameras(sample_data["image_paths"], predictions, config, output_path=cam_path)

    return {
        "predictions": predictions,
        "vectorized": vectorized,
        "semantic_masks": semantic_masks,
    }


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="HDMapNet Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--images", type=str, nargs=6, required=True,
                        help="6 camera image paths in order: FRONT, FRONT_RIGHT, BACK_RIGHT, BACK, BACK_LEFT, FRONT_LEFT")
    parser.add_argument("--intrinsics", type=str, default=None,
                        help="Path to JSON file with camera intrinsics (list of 6 3x3 matrices)")
    parser.add_argument("--extrinsics", type=str, default=None,
                        help="Path to JSON file with camera extrinsics (list of 6 4x4 matrices)")
    parser.add_argument("--output_dir", type=str, default="./output/inference",
                        help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    from .train import get_default_config
    full_config = get_default_config()
    full_config.update(config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, full_config, device)
    print("Model loaded successfully.")

    # Preprocess images
    image_size = tuple(full_config["image_size"])
    images = preprocess_images(args.images, image_size=image_size)

    # Load or create dummy calibration
    import json
    if args.intrinsics and os.path.exists(args.intrinsics):
        with open(args.intrinsics, "r") as f:
            intrinsics_raw = json.load(f)
        intrinsics_list = [np.array(k, dtype=np.float32).reshape(3, 3) for k in intrinsics_raw]
    else:
        # Create default intrinsics (approximate nuScenes front camera)
        print("Warning: No intrinsics provided, using default values")
        fx, fy = 1266.4, 1266.4
        cx, cy = 816.3, 491.5
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics_list = [K.copy() for _ in range(6)]

    if args.extrinsics and os.path.exists(args.extrinsics):
        with open(args.extrinsics, "r") as f:
            extrinsics_raw = json.load(f)
        extrinsics_list = [np.array(e, dtype=np.float32).reshape(4, 4) for e in extrinsics_raw]
    else:
        # Create default extrinsics (identity for all cameras)
        print("Warning: No extrinsics provided, using identity transforms")
        extrinsics_list = [np.eye(4, dtype=np.float32) for _ in range(6)]

    orig_size = (900, 1600)  # Typical nuScenes image size
    intrinsics, extrinsics = preprocess_calibration(
        intrinsics_list, extrinsics_list, orig_size, image_size
    )

    # Build sample data
    sample_data = {
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "image_paths": args.images,
    }

    # Run inference
    results = run_single_sample(model, sample_data, full_config, device, output_dir=args.output_dir)

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
