"""
Cluster analysis for grouping recordings into scenario categories.

Supports multiple clustering methods (KMeans, HDBSCAN, Spectral) on
embedding vectors or tag-derived feature vectors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import MultiLabelBinarizer

from ..taxonomy.scenario_schema import ScenarioAnnotation, ScenarioTag


def _extract_layer(node_id: str) -> int:
    """Extract layer number from a node_id like 'L4.3.1' -> 4."""
    if node_id.startswith("L") and len(node_id) > 1:
        digit_chars = ""
        for ch in node_id[1:]:
            if ch.isdigit():
                digit_chars += ch
            else:
                break
        if digit_chars:
            return int(digit_chars)
    return 0


class ScenarioClusterAnalyzer:
    """
    Cluster recordings into scenario groups using embedding or tag vectors.

    Supports multiple clustering algorithms and provides tools for
    interpreting and labeling the resulting clusters.

    Parameters
    ----------
    random_state : int or None
        Random state for reproducible clustering results.
    """

    def __init__(self, random_state: Optional[int] = 42) -> None:
        self.random_state = random_state

    def cluster_by_embedding(
        self,
        embeddings: np.ndarray,
        method: str = "hdbscan",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Cluster embeddings using the specified method.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix of shape (n_samples, n_features).
        method : str
            Clustering method: "kmeans", "hdbscan", or "spectral".
        **kwargs
            Additional keyword arguments passed to the clustering algorithm.
            - For kmeans: n_clusters (default 10)
            - For hdbscan: min_cluster_size (default 5), min_samples (default 3)
            - For spectral: n_clusters (default 10)

        Returns
        -------
        np.ndarray
            Cluster labels of shape (n_samples,). Label -1 indicates noise
            (for HDBSCAN).

        Raises
        ------
        ValueError
            If an unsupported method is specified.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        n_samples = embeddings.shape[0]
        if n_samples == 0:
            return np.array([], dtype=np.int32)

        method = method.lower()

        if method == "kmeans":
            n_clusters = kwargs.get("n_clusters", min(10, n_samples))
            model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(embeddings)

        elif method == "hdbscan":
            try:
                from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
            except ImportError:
                try:
                    import hdbscan as hdbscan_lib  # type: ignore

                    min_cluster_size = kwargs.get("min_cluster_size", 5)
                    min_samples = kwargs.get("min_samples", 3)
                    clusterer = hdbscan_lib.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                    )
                    labels = clusterer.fit_predict(embeddings)
                    return labels.astype(np.int32)
                except ImportError:
                    raise ImportError(
                        "HDBSCAN requires scikit-learn >= 1.3 or the 'hdbscan' package. "
                        "Install with: pip install hdbscan"
                    )

            min_cluster_size = kwargs.get("min_cluster_size", 5)
            min_samples = kwargs.get("min_samples", 3)
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            labels = model.fit_predict(embeddings)

        elif method == "spectral":
            n_clusters = kwargs.get("n_clusters", min(10, n_samples))
            model = SpectralClustering(
                n_clusters=n_clusters,
                random_state=self.random_state,
                affinity="nearest_neighbors",
                n_neighbors=min(10, n_samples - 1),
            )
            labels = model.fit_predict(embeddings)

        else:
            raise ValueError(
                f"Unsupported clustering method '{method}'. "
                f"Supported: 'kmeans', 'hdbscan', 'spectral'"
            )

        return np.asarray(labels, dtype=np.int32)

    def cluster_by_tags(
        self,
        annotations: List[ScenarioAnnotation],
        n_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cluster recordings by their tag vectors using multi-hot encoding.

        Each recording's tags are encoded as a binary vector indicating
        which scenario nodes are present. KMeans is used on the resulting
        binary feature matrix.

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            Annotations to cluster.
        n_clusters : int or None
            Number of clusters. If None, uses sqrt(n_samples) heuristic.

        Returns
        -------
        np.ndarray
            Cluster labels of shape (n_annotations,).
        """
        if not annotations:
            return np.array([], dtype=np.int32)

        # Extract tag sets for each annotation
        tag_sets: List[List[str]] = []
        for ann in annotations:
            node_ids = [tag.node_id for tag in ann.tags]
            tag_sets.append(node_ids)

        # Multi-hot encode
        mlb = MultiLabelBinarizer()
        feature_matrix = mlb.fit_transform(tag_sets).astype(np.float64)

        # Determine n_clusters
        n_samples = len(annotations)
        if n_clusters is None:
            n_clusters = max(2, min(int(np.sqrt(n_samples)), n_samples))

        n_clusters = min(n_clusters, n_samples)

        model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = model.fit_predict(feature_matrix)

        return np.asarray(labels, dtype=np.int32)

    def label_clusters(
        self,
        clusters: np.ndarray,
        annotations: List[ScenarioAnnotation],
    ) -> Dict[int, str]:
        """
        Auto-label clusters with their dominant scenario attributes.

        For each cluster, finds the most frequently occurring tags and
        produces a human-readable label from the dominant attributes.

        Parameters
        ----------
        clusters : np.ndarray
            Cluster labels of shape (n_samples,).
        annotations : list of ScenarioAnnotation
            Corresponding annotations (same order as clusters).

        Returns
        -------
        dict
            Mapping from cluster_id to descriptive label string.
        """
        clusters = np.asarray(clusters)
        unique_labels = set(clusters.tolist())

        labels: Dict[int, str] = {}

        for cluster_id in sorted(unique_labels):
            if cluster_id == -1:
                labels[-1] = "Noise (unassigned)"
                continue

            # Get indices of samples in this cluster
            mask = clusters == cluster_id
            cluster_annotations = [
                ann for ann, m in zip(annotations, mask) if m
            ]

            if not cluster_annotations:
                labels[cluster_id] = f"Cluster {cluster_id} (empty)"
                continue

            # Count tag frequencies in this cluster
            tag_counter: Counter = Counter()
            for ann in cluster_annotations:
                for tag in ann.tags:
                    tag_counter[tag.node_id] += 1

            # Find dominant tags (top 3 by frequency, present in >50% of cluster)
            cluster_size = len(cluster_annotations)
            dominant_tags: List[str] = []

            for node_id, count in tag_counter.most_common(5):
                if count >= cluster_size * 0.4:  # Present in 40%+ of cluster
                    dominant_tags.append(node_id)
                if len(dominant_tags) >= 3:
                    break

            if dominant_tags:
                labels[cluster_id] = " + ".join(dominant_tags)
            else:
                # Fall back to most common tag
                most_common = tag_counter.most_common(1)
                if most_common:
                    labels[cluster_id] = most_common[0][0]
                else:
                    labels[cluster_id] = f"Cluster {cluster_id}"

        return labels

    def find_cluster_exemplars(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
    ) -> Dict[int, List[int]]:
        """
        Find representative samples (exemplars) for each cluster.

        Exemplars are the samples closest to the cluster centroid.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix of shape (n_samples, n_features).
        clusters : np.ndarray
            Cluster labels of shape (n_samples,).

        Returns
        -------
        dict
            Mapping from cluster_id to list of sample indices (sorted by
            proximity to centroid). Up to 5 exemplars per cluster.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        clusters = np.asarray(clusters)

        unique_labels = set(clusters.tolist())
        exemplars: Dict[int, List[int]] = {}

        for cluster_id in sorted(unique_labels):
            if cluster_id == -1:
                continue

            # Get indices and embeddings for this cluster
            mask = clusters == cluster_id
            cluster_indices = np.where(mask)[0]

            if len(cluster_indices) == 0:
                exemplars[cluster_id] = []
                continue

            cluster_embeddings = embeddings[cluster_indices]

            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)

            # Compute distances to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            # Sort by distance (closest to centroid first)
            sorted_local_indices = np.argsort(distances)

            # Take up to 5 exemplars
            n_exemplars = min(5, len(sorted_local_indices))
            exemplar_indices = cluster_indices[sorted_local_indices[:n_exemplars]]

            exemplars[cluster_id] = exemplar_indices.tolist()

        return exemplars

    def compute_cluster_statistics(
        self,
        clusters: np.ndarray,
        annotations: List[ScenarioAnnotation],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute descriptive statistics for each cluster.

        Parameters
        ----------
        clusters : np.ndarray
            Cluster labels of shape (n_samples,).
        annotations : list of ScenarioAnnotation
            Corresponding annotations (same order as clusters).

        Returns
        -------
        dict
            Mapping from cluster_id to statistics dict containing:
            - 'size': number of samples in cluster
            - 'tag_distribution': Counter of node_ids in the cluster
            - 'layer_distribution': dict mapping layer -> count
            - 'unique_recordings': number of unique recording IDs
            - 'avg_tags_per_recording': average number of tags
            - 'dominant_tags': top-5 most frequent tags with counts
        """
        clusters = np.asarray(clusters)
        unique_labels = set(clusters.tolist())

        statistics: Dict[int, Dict[str, Any]] = {}

        for cluster_id in sorted(unique_labels):
            mask = clusters == cluster_id
            cluster_annotations = [
                ann for ann, m in zip(annotations, mask) if m
            ]

            if not cluster_annotations:
                statistics[cluster_id] = {
                    "size": 0,
                    "tag_distribution": Counter(),
                    "layer_distribution": {},
                    "unique_recordings": 0,
                    "avg_tags_per_recording": 0.0,
                    "dominant_tags": [],
                }
                continue

            cluster_size = len(cluster_annotations)

            # Tag distribution
            tag_counter: Counter = Counter()
            layer_counter: Counter = Counter()
            total_tags = 0
            recording_ids: Set[str] = set()

            for ann in cluster_annotations:
                recording_ids.add(ann.recording_id)
                total_tags += len(ann.tags)
                for tag in ann.tags:
                    tag_counter[tag.node_id] += 1
                    layer_counter[_extract_layer(tag.node_id)] += 1

            avg_tags = total_tags / cluster_size if cluster_size > 0 else 0.0

            statistics[cluster_id] = {
                "size": cluster_size,
                "tag_distribution": tag_counter,
                "layer_distribution": dict(layer_counter),
                "unique_recordings": len(recording_ids),
                "avg_tags_per_recording": round(avg_tags, 2),
                "dominant_tags": tag_counter.most_common(5),
            }

        return statistics
