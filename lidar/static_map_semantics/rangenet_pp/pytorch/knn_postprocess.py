"""KNN-based post-processing for RangeNet++ label refinement.

After predicting labels on the range image, this module transfers labels
back to the original 3D point cloud and refines predictions using K-nearest
neighbor voting. This corrects discretization errors from the spherical
projection and improves boundary accuracy.

Reference: Section III-C of "RangeNet++: Fast and Accurate LiDAR Semantic
Segmentation" (Milioto et al., IROS 2019)
"""

import numpy as np
import torch
from typing import Optional, Tuple


def knn_postprocess_numpy(
    predicted_labels_image: np.ndarray,
    points: np.ndarray,
    pixel_to_point: np.ndarray,
    point_to_pixel: np.ndarray,
    k: int = 5,
    search_radius: float = 1.0,
    num_classes: int = 20,
) -> np.ndarray:
    """KNN-based label refinement using scipy KDTree (CPU, numpy).

    For each point in the original point cloud:
    1. Look up its predicted label from the range image.
    2. Find K nearest neighbors in 3D space.
    3. Perform majority voting among neighbors' predicted labels.
    4. Assign the majority label.

    Args:
        predicted_labels_image: (H, W) int array of predicted class labels in range image.
        points: (N, 4) float array of original point cloud [x, y, z, intensity].
        pixel_to_point: (H, W) int array mapping pixels to point indices (-1 for empty).
        point_to_pixel: (N, 2) int array mapping points to (row, col) (-1 if not projected).
        k: Number of nearest neighbors for voting.
        search_radius: Maximum search distance in meters (for efficiency).
        num_classes: Number of semantic classes.

    Returns:
        refined_labels: (N,) int array of per-point semantic labels.
    """
    from scipy.spatial import cKDTree

    N = points.shape[0]
    xyz = points[:, :3]

    # Step 1: Get initial per-point labels from range image
    initial_labels = np.zeros(N, dtype=np.int32)
    for i in range(N):
        r, c = point_to_pixel[i]
        if r >= 0 and c >= 0:
            initial_labels[i] = predicted_labels_image[r, c]
        # else: label stays 0 (unlabeled)

    # Step 2: Build KD-tree on all valid points (those with labels)
    valid_mask = initial_labels > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return initial_labels

    valid_xyz = xyz[valid_indices]
    tree = cKDTree(valid_xyz)

    # Step 3: For each point, find KNN and vote
    refined_labels = initial_labels.copy()

    # Query all points against valid-point tree
    distances, neighbor_indices = tree.query(xyz, k=k, distance_upper_bound=search_radius)

    for i in range(N):
        # Collect neighbor labels
        vote_counts = np.zeros(num_classes, dtype=np.int32)
        for j in range(k):
            if neighbor_indices[i, j] < len(valid_indices):
                dist = distances[i, j]
                if dist < search_radius:
                    neighbor_point_idx = valid_indices[neighbor_indices[i, j]]
                    label = initial_labels[neighbor_point_idx]
                    if label > 0:
                        vote_counts[label] += 1

        # Majority vote (only override if we have valid votes)
        total_votes = vote_counts[1:].sum()
        if total_votes > 0:
            refined_labels[i] = np.argmax(vote_counts[1:]) + 1

    return refined_labels


def knn_postprocess_numpy_fast(
    predicted_labels_image: np.ndarray,
    points: np.ndarray,
    pixel_to_point: np.ndarray,
    point_to_pixel: np.ndarray,
    k: int = 5,
    search_radius: float = 1.0,
    num_classes: int = 20,
) -> np.ndarray:
    """Optimized KNN post-processing using vectorized operations.

    Same functionality as knn_postprocess_numpy but faster for large point clouds
    by avoiding per-point Python loops.

    Args:
        predicted_labels_image: (H, W) int array.
        points: (N, 4) float array [x, y, z, intensity].
        pixel_to_point: (H, W) int array.
        point_to_pixel: (N, 2) int array.
        k: Number of nearest neighbors.
        search_radius: Max search distance (meters).
        num_classes: Number of classes.

    Returns:
        refined_labels: (N,) int array.
    """
    from scipy.spatial import cKDTree

    N = points.shape[0]
    xyz = points[:, :3]

    # Get initial labels from range image (vectorized)
    initial_labels = np.zeros(N, dtype=np.int32)
    projected_mask = (point_to_pixel[:, 0] >= 0) & (point_to_pixel[:, 1] >= 0)
    projected_indices = np.where(projected_mask)[0]
    rows = point_to_pixel[projected_indices, 0]
    cols = point_to_pixel[projected_indices, 1]
    initial_labels[projected_indices] = predicted_labels_image[rows, cols]

    # Build KD-tree on labeled points
    valid_mask = initial_labels > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return initial_labels

    valid_xyz = xyz[valid_indices]
    tree = cKDTree(valid_xyz)

    # Batch query: find K nearest valid neighbors for all points
    distances, knn_indices = tree.query(xyz, k=k, distance_upper_bound=search_radius)

    # knn_indices can be == len(valid_indices) when no neighbor within radius
    max_valid = len(valid_indices)

    # Gather labels for all neighbors (vectorized)
    # Clip indices for safe indexing, then mask invalid ones
    safe_indices = np.clip(knn_indices, 0, max_valid - 1)
    neighbor_point_indices = valid_indices[safe_indices]  # (N, k)
    neighbor_labels = initial_labels[neighbor_point_indices]  # (N, k)

    # Mask invalid neighbors (beyond search radius or tree size)
    invalid_neighbors = (knn_indices >= max_valid) | (distances >= search_radius)
    neighbor_labels[invalid_neighbors] = 0

    # Majority voting (vectorized using bincount per row)
    refined_labels = initial_labels.copy()

    # For efficiency, process in chunks
    chunk_size = 50000
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_labels = neighbor_labels[start:end]  # (chunk, k)

        for i in range(end - start):
            labels_row = chunk_labels[i]
            valid_votes = labels_row[labels_row > 0]
            if len(valid_votes) > 0:
                counts = np.bincount(valid_votes, minlength=num_classes)
                refined_labels[start + i] = np.argmax(counts[1:]) + 1

    return refined_labels


def knn_postprocess_torch(
    predicted_labels_image: torch.Tensor,
    points: torch.Tensor,
    point_to_pixel: torch.Tensor,
    k: int = 5,
    search_radius: float = 1.0,
    num_classes: int = 20,
    batch_size: int = 4096,
) -> torch.Tensor:
    """KNN post-processing using PyTorch for GPU acceleration.

    Uses brute-force KNN on GPU (efficient for moderate point clouds <200k points).
    For larger clouds, processes in batches.

    Args:
        predicted_labels_image: (H, W) long tensor on device.
        points: (N, 4) float tensor [x, y, z, intensity] on device.
        point_to_pixel: (N, 2) long tensor on device.
        k: Number of neighbors.
        search_radius: Max distance for valid neighbors.
        num_classes: Number of classes.
        batch_size: Points processed per batch for memory efficiency.

    Returns:
        refined_labels: (N,) long tensor of refined per-point labels.
    """
    device = points.device
    N = points.shape[0]
    xyz = points[:, :3]  # (N, 3)

    # Get initial labels from range image
    initial_labels = torch.zeros(N, dtype=torch.long, device=device)
    valid_proj = (point_to_pixel[:, 0] >= 0) & (point_to_pixel[:, 1] >= 0)
    if valid_proj.any():
        rows = point_to_pixel[valid_proj, 0]
        cols = point_to_pixel[valid_proj, 1]
        initial_labels[valid_proj] = predicted_labels_image[rows, cols]

    # Find points with valid labels
    labeled_mask = initial_labels > 0
    labeled_indices = torch.where(labeled_mask)[0]

    if labeled_indices.numel() == 0:
        return initial_labels

    labeled_xyz = xyz[labeled_indices]  # (M, 3)
    labeled_labels = initial_labels[labeled_indices]  # (M,)
    M = labeled_xyz.shape[0]

    refined_labels = initial_labels.clone()

    # Process in batches to avoid OOM
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        query_xyz = xyz[start:end]  # (B_chunk, 3)
        B_chunk = query_xyz.shape[0]

        # Compute pairwise distances: (B_chunk, M)
        # Use chunked computation if M is very large
        if M > 50000:
            # Sub-batch the reference points
            all_knn_dists = torch.full((B_chunk, k), float("inf"), device=device)
            all_knn_indices = torch.zeros((B_chunk, k), dtype=torch.long, device=device)

            ref_batch_size = 50000
            for ref_start in range(0, M, ref_batch_size):
                ref_end = min(ref_start + ref_batch_size, M)
                ref_xyz = labeled_xyz[ref_start:ref_end]  # (R, 3)

                # (B_chunk, R)
                dists = torch.cdist(query_xyz, ref_xyz, p=2.0)
                top_k_dists, top_k_idx = torch.topk(dists, k=min(k, dists.shape[1]), dim=1, largest=False)
                top_k_idx = top_k_idx + ref_start  # offset to global labeled index

                # Merge with current best
                combined_dists = torch.cat([all_knn_dists, top_k_dists], dim=1)
                combined_indices = torch.cat([all_knn_indices, top_k_idx], dim=1)
                best_k = torch.topk(combined_dists, k=k, dim=1, largest=False)
                all_knn_dists = best_k.values
                all_knn_indices = torch.gather(combined_indices, 1, best_k.indices)

            knn_dists = all_knn_dists
            knn_indices = all_knn_indices
        else:
            # Full pairwise distance matrix
            dists = torch.cdist(query_xyz, labeled_xyz, p=2.0)  # (B_chunk, M)
            actual_k = min(k, M)
            knn_dists, knn_indices = torch.topk(dists, k=actual_k, dim=1, largest=False)

            # Pad if M < k
            if actual_k < k:
                pad_dists = torch.full((B_chunk, k - actual_k), float("inf"), device=device)
                pad_idx = torch.zeros((B_chunk, k - actual_k), dtype=torch.long, device=device)
                knn_dists = torch.cat([knn_dists, pad_dists], dim=1)
                knn_indices = torch.cat([knn_indices, pad_idx], dim=1)

        # Gather neighbor labels
        neighbor_labels = labeled_labels[knn_indices]  # (B_chunk, k)

        # Mask out neighbors beyond search radius
        invalid = knn_dists > search_radius
        neighbor_labels[invalid] = 0

        # Majority voting per point
        for i in range(B_chunk):
            votes = neighbor_labels[i]
            valid_votes = votes[votes > 0]
            if valid_votes.numel() > 0:
                # One-hot accumulation for voting
                vote_counts = torch.zeros(num_classes, device=device, dtype=torch.long)
                for v in valid_votes:
                    vote_counts[v] += 1
                # Argmax over classes 1..num_classes-1
                refined_labels[start + i] = torch.argmax(vote_counts[1:]) + 1

    return refined_labels


def knn_postprocess_torch_vectorized(
    predicted_labels_image: torch.Tensor,
    points: torch.Tensor,
    point_to_pixel: torch.Tensor,
    k: int = 5,
    search_radius: float = 1.0,
    num_classes: int = 20,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Fully vectorized KNN post-processing (no Python loops in voting).

    Uses scatter_add for vote counting, significantly faster on GPU.

    Args:
        predicted_labels_image: (H, W) long tensor.
        points: (N, 4) float tensor.
        point_to_pixel: (N, 2) long tensor.
        k: Number of neighbors.
        search_radius: Max neighbor distance.
        num_classes: Number of classes.
        batch_size: Batch size for distance computation.

    Returns:
        refined_labels: (N,) long tensor.
    """
    device = points.device
    N = points.shape[0]
    xyz = points[:, :3]

    # Initial label assignment
    initial_labels = torch.zeros(N, dtype=torch.long, device=device)
    valid_proj = (point_to_pixel[:, 0] >= 0) & (point_to_pixel[:, 1] >= 0)
    if valid_proj.any():
        rows = point_to_pixel[valid_proj, 0]
        cols = point_to_pixel[valid_proj, 1]
        initial_labels[valid_proj] = predicted_labels_image[rows, cols]

    labeled_mask = initial_labels > 0
    labeled_indices = torch.where(labeled_mask)[0]
    if labeled_indices.numel() == 0:
        return initial_labels

    labeled_xyz = xyz[labeled_indices]
    labeled_labels = initial_labels[labeled_indices]
    M = labeled_xyz.shape[0]

    refined_labels = initial_labels.clone()
    actual_k = min(k, M)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        query_xyz = xyz[start:end]
        B_chunk = end - start

        # Pairwise distances
        dists = torch.cdist(query_xyz, labeled_xyz, p=2.0)  # (B_chunk, M)
        knn_dists, knn_idx = torch.topk(dists, k=actual_k, dim=1, largest=False)

        # Gather neighbor labels
        neighbor_labels = labeled_labels[knn_idx]  # (B_chunk, actual_k)

        # Mask invalid (beyond radius)
        invalid = knn_dists > search_radius
        neighbor_labels[invalid] = 0

        # Vectorized majority voting using scatter
        # Create vote matrix: (B_chunk, num_classes)
        vote_counts = torch.zeros(B_chunk, num_classes, device=device, dtype=torch.long)
        # Expand neighbor labels for scatter_add
        flat_labels = neighbor_labels.view(B_chunk, -1)  # (B_chunk, actual_k)
        ones = torch.ones_like(flat_labels)
        vote_counts.scatter_add_(1, flat_labels, ones)

        # Zero out class 0 votes (unlabeled)
        vote_counts[:, 0] = 0

        # Find majority class (argmax over classes 1+)
        has_votes = vote_counts[:, 1:].sum(dim=1) > 0
        winners = torch.argmax(vote_counts[:, 1:], dim=1) + 1  # (B_chunk,)

        # Only update points that got valid votes
        update_mask = has_votes
        chunk_indices = torch.arange(start, end, device=device)
        refined_labels[chunk_indices[update_mask]] = winners[update_mask]

    return refined_labels
