"""
Point cloud sampling and grouping operations for PointNet++.

Implements Farthest Point Sampling, Ball Query, KNN Query, and
pairwise distance computation in pure PyTorch.
"""

import torch


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances between two point sets.

    Uses the expansion: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b

    Args:
        src: Source points, shape (B, N, C)
        dst: Destination points, shape (B, M, C)

    Returns:
        Pairwise squared distances, shape (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    # (B, N, M)
    dist = -2.0 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)  # (B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).permute(0, 2, 1)  # (B, 1, M)

    # Clamp to avoid tiny negative values from floating point
    dist = torch.clamp(dist, min=0.0)
    return dist


def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS).

    Iteratively selects the point that is farthest from the current set
    of selected points, producing a well-spread subset.

    Args:
        xyz: Input point positions, shape (B, N, 3)
        npoint: Number of points to sample

    Returns:
        Indices of sampled points, shape (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, dtype=torch.float32, device=device)

    # Random initialization: pick a random starting point per batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest

        # Get coordinates of the farthest point
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)

        # Compute distances from all points to the new centroid
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)

        # Update minimum distances
        distance = torch.min(distance, dist)

        # Select the point with the maximum minimum distance
        farthest = torch.argmax(distance, dim=-1)  # (B,)

    return centroids


def ball_query(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    """
    Ball Query: find all points within a radius for each query point.

    For each point in new_xyz, finds all points in xyz within the given
    radius and returns up to nsample indices. If fewer than nsample points
    are found, the first found index is repeated to fill.

    Args:
        radius: Search radius
        nsample: Maximum number of neighbors to return
        xyz: All points, shape (B, N, 3)
        new_xyz: Query (center) points, shape (B, S, 3)

    Returns:
        Group indices, shape (B, S, nsample). Each entry is an index into
        the N points of xyz.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    # Compute pairwise squared distances: (B, S, N)
    sqrdists = square_distance(new_xyz, xyz)

    radius_sq = radius * radius

    # Initialize group indices with 0 (will be overwritten)
    group_idx = torch.zeros(B, S, nsample, dtype=torch.long, device=device)

    # Create a mask for points within the radius
    # sqrdists shape: (B, S, N)
    mask = sqrdists <= radius_sq  # (B, S, N)

    # Sort distances to get nearest neighbors first within the ball
    # Replace out-of-radius distances with a large value for sorting
    sqrdists_masked = sqrdists.clone()
    sqrdists_masked[~mask] = 1e10

    # Get sorted indices by distance
    sorted_indices = torch.argsort(sqrdists_masked, dim=-1)  # (B, S, N)

    # Take the first nsample
    sorted_indices = sorted_indices[:, :, :nsample]  # (B, S, nsample)

    # Check which of these are actually within the radius
    # Gather the mask values for the sorted indices
    sorted_mask = torch.gather(mask, 2, sorted_indices)  # (B, S, nsample)

    # For points where no neighbor is within radius, use the first sorted index
    # Get the first valid index for each query point
    first_idx = sorted_indices[:, :, 0:1].expand_as(sorted_indices)  # (B, S, nsample)

    # Where the mask is False (no valid neighbor), replace with first index
    group_idx = torch.where(sorted_mask, sorted_indices, first_idx)

    return group_idx


def knn_query(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    K-Nearest Neighbors query using pairwise distance matrix.

    For each point in new_xyz, finds the k closest points in xyz.

    Args:
        k: Number of nearest neighbors
        xyz: Reference points, shape (B, N, 3)
        new_xyz: Query points, shape (B, S, 3)

    Returns:
        KNN indices, shape (B, S, k). Each entry is an index into the
        N points of xyz.
    """
    # Compute pairwise squared distances: (B, S, N)
    sqrdists = square_distance(new_xyz, xyz)

    # Get top-k smallest distances
    _, indices = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=True)

    return indices


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by indices.

    Args:
        points: Input points, shape (B, N, C)
        idx: Indices to gather, shape (B, S) or (B, S, nsample)

    Returns:
        Gathered points with shape matching idx dimensions + C
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )

    new_points = points[batch_indices, idx, :]
    return new_points
