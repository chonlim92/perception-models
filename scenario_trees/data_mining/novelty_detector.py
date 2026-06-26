"""
Novelty detection for identifying unusual or rare scenarios.

Combines embedding-space anomaly detection (IsolationForest) with
tag-based rarity scoring to surface corner cases in driving data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from ..taxonomy.scenario_schema import ScenarioAnnotation, ScenarioTag
from .embedding_store import EmbeddingStore


@dataclass
class NoveltyResult:
    """Result from novelty detection analysis."""

    recording_id: str
    novelty_score: float
    reason: str


class TagDatabase(Protocol):
    """Protocol for tag database objects used in novelty detection."""

    def get_annotations(self) -> List[ScenarioAnnotation]: ...


class NoveltyDetector:
    """
    Detect unusual or rare scenarios using embedding-space anomaly detection
    and tag frequency analysis.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the training data.
        Passed to IsolationForest. Default 0.1.
    n_neighbors : int
        Number of neighbors for k-NN density estimation. Default 10.
    random_state : int or None
        Random state for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        random_state: Optional[int] = 42,
    ) -> None:
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self._isolation_forest: Optional[IsolationForest] = None
        self._tag_frequencies: Dict[str, int] = {}
        self._total_samples: int = 0
        self._is_fitted: bool = False

    def fit(self, embeddings: np.ndarray, tags: List[Dict[str, Any]]) -> None:
        """
        Fit novelty detection models on training data.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix of shape (n_samples, n_features).
        tags : list of dict
            List of tag dictionaries, one per sample. Each dict should have
            a 'node_ids' key with a list of node_id strings, or be a flat dict
            mapping attribute names to values.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        n_samples = embeddings.shape[0]
        self._total_samples = n_samples

        # Fit IsolationForest on embedding space
        self._isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self._isolation_forest.fit(embeddings)

        # Compute tag frequencies
        self._tag_frequencies = Counter()
        for tag_dict in tags:
            if "node_ids" in tag_dict:
                for node_id in tag_dict["node_ids"]:
                    self._tag_frequencies[node_id] += 1
            else:
                for key, value in tag_dict.items():
                    tag_key = f"{key}={value}"
                    self._tag_frequencies[tag_key] += 1

        self._is_fitted = True

    def score_novelty(self, embedding: np.ndarray, tags: Dict[str, Any]) -> float:
        """
        Compute a novelty score for a single sample.

        Combines IsolationForest anomaly score with tag rarity.
        Output is in range [0, 1] where 1 = most novel.

        Parameters
        ----------
        embedding : np.ndarray
            Embedding vector of shape (n_features,).
        tags : dict
            Tag dictionary for this sample. Can contain 'node_ids' key
            with list of node IDs, or attribute key-value pairs.

        Returns
        -------
        float
            Novelty score between 0 and 1.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted yet.
        """
        if not self._is_fitted or self._isolation_forest is None:
            raise RuntimeError("NoveltyDetector must be fitted before scoring")

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        # IsolationForest score: decision_function returns negative for anomalies
        # More negative = more anomalous
        raw_score = self._isolation_forest.decision_function(embedding)[0]
        # Convert to [0, 1] where 1 is most novel
        # decision_function typically ranges from about -0.5 to 0.5
        embedding_novelty = 1.0 / (1.0 + np.exp(5.0 * raw_score))

        # Tag rarity score
        tag_novelty = self._compute_tag_rarity(tags)

        # Combine scores: weighted average
        combined = 0.6 * embedding_novelty + 0.4 * tag_novelty
        return float(np.clip(combined, 0.0, 1.0))

    def _compute_tag_rarity(self, tags: Dict[str, Any]) -> float:
        """
        Compute tag-based rarity score.

        Tags with fewer occurrences in the training data receive higher scores.

        Returns
        -------
        float
            Rarity score between 0 and 1.
        """
        if self._total_samples == 0:
            return 0.5

        rarity_scores: List[float] = []

        if "node_ids" in tags:
            for node_id in tags["node_ids"]:
                freq = self._tag_frequencies.get(node_id, 0)
                # Rarity = 1 - (frequency / total), so rare tags score high
                rarity = 1.0 - (freq / self._total_samples)
                rarity_scores.append(rarity)
        else:
            for key, value in tags.items():
                tag_key = f"{key}={value}"
                freq = self._tag_frequencies.get(tag_key, 0)
                rarity = 1.0 - (freq / self._total_samples)
                rarity_scores.append(rarity)

        if not rarity_scores:
            return 0.5

        return float(np.mean(rarity_scores))

    def find_rare_scenarios(
        self,
        embedding_store: EmbeddingStore,
        tag_database: Any,
        top_k: int = 100,
    ) -> List[NoveltyResult]:
        """
        Find the most novel/rare scenarios across the entire dataset.

        Parameters
        ----------
        embedding_store : EmbeddingStore
            Store containing all embeddings.
        tag_database : object
            Object with a `get_annotations()` method returning a list of
            ScenarioAnnotation objects.
        top_k : int
            Number of most novel scenarios to return.

        Returns
        -------
        list of NoveltyResult
            Top-k most novel scenarios, sorted by descending novelty score.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("NoveltyDetector must be fitted before finding rare scenarios")

        annotations: List[ScenarioAnnotation] = tag_database.get_annotations()
        annotation_map: Dict[str, ScenarioAnnotation] = {
            ann.recording_id: ann for ann in annotations
        }

        all_ids = embedding_store.get_all_ids()
        all_embeddings = embedding_store.get_all_embeddings()

        results: List[NoveltyResult] = []

        for i, recording_id in enumerate(all_ids):
            embedding = all_embeddings[i]

            # Build tags dict from annotation if available
            tags: Dict[str, Any] = {}
            if recording_id in annotation_map:
                ann = annotation_map[recording_id]
                tags = {"node_ids": [t.node_id for t in ann.tags]}

            novelty_score = self.score_novelty(embedding, tags)

            # Determine reason
            reason = self._determine_novelty_reason(embedding, tags)

            results.append(NoveltyResult(
                recording_id=recording_id,
                novelty_score=novelty_score,
                reason=reason,
            ))

        # Sort by novelty score descending and return top_k
        results.sort(key=lambda r: r.novelty_score, reverse=True)
        return results[:top_k]

    def _determine_novelty_reason(
        self, embedding: np.ndarray, tags: Dict[str, Any]
    ) -> str:
        """Determine the primary reason a sample is novel."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        # Check embedding-space novelty
        raw_score = self._isolation_forest.decision_function(embedding)[0]  # type: ignore
        embedding_novelty = 1.0 / (1.0 + np.exp(5.0 * raw_score))

        # Check tag rarity
        tag_novelty = self._compute_tag_rarity(tags)

        reasons = []
        if embedding_novelty > 0.6:
            reasons.append("embedding outlier in feature space")
        if tag_novelty > 0.6:
            # Find the rarest tags
            rare_tags = self._find_rarest_tags(tags, top_n=3)
            if rare_tags:
                reasons.append(f"rare tag combination: {', '.join(rare_tags)}")
            else:
                reasons.append("rare tag combination")

        if not reasons:
            reasons.append("marginally unusual combination of features and tags")

        return "; ".join(reasons)

    def _find_rarest_tags(self, tags: Dict[str, Any], top_n: int = 3) -> List[str]:
        """Find the rarest individual tags in a tag dict."""
        tag_items: List[tuple] = []

        if "node_ids" in tags:
            for node_id in tags["node_ids"]:
                freq = self._tag_frequencies.get(node_id, 0)
                tag_items.append((node_id, freq))
        else:
            for key, value in tags.items():
                tag_key = f"{key}={value}"
                freq = self._tag_frequencies.get(tag_key, 0)
                tag_items.append((tag_key, freq))

        # Sort by frequency ascending (rarest first)
        tag_items.sort(key=lambda x: x[1])
        return [item[0] for item in tag_items[:top_n]]

    def find_low_density_regions(
        self, embeddings: np.ndarray, threshold: float = 0.1
    ) -> np.ndarray:
        """
        Find embeddings in low-density regions using k-NN density estimation.

        The density for each point is estimated as the inverse of its average
        distance to its k nearest neighbors. Points below the given threshold
        percentile of density are flagged.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix of shape (n_samples, n_features).
        threshold : float
            Fraction of samples to flag as low-density (0 to 1).
            E.g., 0.1 means the 10% of points with lowest density.

        Returns
        -------
        np.ndarray
            Boolean mask of shape (n_samples,) where True indicates
            low-density region membership.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        n_samples = embeddings.shape[0]
        if n_samples == 0:
            return np.array([], dtype=bool)

        # Adjust n_neighbors if we have fewer samples
        k = min(self.n_neighbors, n_samples - 1)
        if k < 1:
            return np.ones(n_samples, dtype=bool)

        # k-NN density estimation
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)

        # Density is inverse of mean distance to k neighbors
        mean_distances = distances.mean(axis=1)
        # Avoid division by zero
        density = 1.0 / (mean_distances + 1e-10)

        # Find threshold percentile
        n_low_density = max(1, int(n_samples * threshold))
        density_threshold = np.sort(density)[n_low_density - 1]

        return density <= density_threshold
