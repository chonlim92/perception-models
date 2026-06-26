"""
Embedding store for scene similarity search.

Stores normalized scene embeddings backed by numpy arrays, with optional
FAISS support for accelerated approximate nearest-neighbor search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    """Result from an embedding similarity search."""

    recording_id: str
    score: float
    metadata: Dict[str, Any]


class EmbeddingStore:
    """
    Store scene embeddings for cosine similarity search.

    Embeddings are L2-normalized on insertion so that cosine similarity
    reduces to a simple dot product.

    Parameters
    ----------
    dimension : int
        Dimensionality of the embedding vectors.
    use_faiss : bool
        If True and faiss is importable, use a FAISS IndexFlatIP for search.
        Otherwise falls back to pure numpy dot-product search.
    """

    def __init__(self, dimension: int, use_faiss: bool = False) -> None:
        self.dimension = dimension
        self._ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._embeddings: List[np.ndarray] = []
        self._id_to_index: Dict[str, int] = {}

        self._faiss_index: Optional[Any] = None
        self._use_faiss = use_faiss

        if use_faiss:
            try:
                import faiss  # type: ignore

                self._faiss_index = faiss.IndexFlatIP(dimension)
            except ImportError:
                self._faiss_index = None
                self._use_faiss = False

    @property
    def size(self) -> int:
        """Number of embeddings currently stored."""
        return len(self._ids)

    def add(self, recording_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        """
        Store a normalized embedding with associated metadata.

        Parameters
        ----------
        recording_id : str
            Unique identifier for the recording.
        embedding : np.ndarray
            Embedding vector of shape (dimension,). Will be L2-normalized.
        metadata : dict
            Arbitrary metadata associated with this recording.

        Raises
        ------
        ValueError
            If the embedding dimension does not match the store dimension,
            or if the recording_id already exists in the store.
        """
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} does not match "
                f"store dimension {self.dimension}"
            )

        if recording_id in self._id_to_index:
            raise ValueError(f"Recording '{recording_id}' already exists in the store")

        # L2 normalize for cosine similarity via dot product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            raise ValueError("Cannot add a zero-magnitude embedding")

        index = len(self._ids)
        self._ids.append(recording_id)
        self._metadata.append(metadata)
        self._embeddings.append(embedding)
        self._id_to_index[recording_id] = index

        if self._faiss_index is not None:
            self._faiss_index.add(embedding.reshape(1, -1))

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Find the top_k most similar embeddings to the query via cosine similarity.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query vector of shape (dimension,). Will be L2-normalized.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of SearchResult
            Sorted by descending similarity score.
        """
        if self.size == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match "
                f"store dimension {self.dimension}"
            )

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        else:
            return []

        top_k = min(top_k, self.size)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query.reshape(1, -1), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                results.append(SearchResult(
                    recording_id=self._ids[idx],
                    score=float(score),
                    metadata=self._metadata[idx],
                ))
            return results

        # Numpy fallback: dot product on normalized vectors = cosine similarity
        matrix = np.array(self._embeddings, dtype=np.float32)
        scores = matrix @ query  # shape (n,)

        # Get top_k indices by descending score
        if top_k >= self.size:
            top_indices = np.argsort(scores)[::-1]
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                recording_id=self._ids[idx],
                score=float(scores[idx]),
                metadata=self._metadata[idx],
            ))
        return results

    def search_by_id(self, recording_id: str, top_k: int = 10) -> List[SearchResult]:
        """
        Find recordings most similar to an existing recording in the store.

        Parameters
        ----------
        recording_id : str
            ID of an existing recording in the store.
        top_k : int
            Number of results to return (excluding the query recording itself).

        Returns
        -------
        list of SearchResult
            Sorted by descending similarity, excluding the query recording.

        Raises
        ------
        KeyError
            If the recording_id is not found in the store.
        """
        if recording_id not in self._id_to_index:
            raise KeyError(f"Recording '{recording_id}' not found in the store")

        idx = self._id_to_index[recording_id]
        query_embedding = self._embeddings[idx]

        # Search for top_k + 1 since we will exclude the query itself
        results = self.search(query_embedding, top_k=top_k + 1)

        # Filter out the query recording itself
        return [r for r in results if r.recording_id != recording_id][:top_k]

    def get_embedding(self, recording_id: str) -> np.ndarray:
        """
        Retrieve the stored embedding for a recording.

        Parameters
        ----------
        recording_id : str
            ID of the recording.

        Returns
        -------
        np.ndarray
            The normalized embedding vector.

        Raises
        ------
        KeyError
            If the recording_id is not found.
        """
        if recording_id not in self._id_to_index:
            raise KeyError(f"Recording '{recording_id}' not found in the store")
        return self._embeddings[self._id_to_index[recording_id]].copy()

    def get_all_embeddings(self) -> np.ndarray:
        """
        Return the full embedding matrix.

        Returns
        -------
        np.ndarray
            Matrix of shape (n_embeddings, dimension) with all stored embeddings.
        """
        if self.size == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.array(self._embeddings, dtype=np.float32)

    def get_all_ids(self) -> List[str]:
        """Return list of all recording IDs in insertion order."""
        return list(self._ids)

    def get_metadata(self, recording_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a recording.

        Raises
        ------
        KeyError
            If the recording_id is not found.
        """
        if recording_id not in self._id_to_index:
            raise KeyError(f"Recording '{recording_id}' not found in the store")
        return self._metadata[self._id_to_index[recording_id]]

    def save(self, path: str) -> None:
        """
        Persist the embedding store to disk in numpy .npz format.

        Parameters
        ----------
        path : str
            File path for the output .npz file.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        embeddings_matrix = self.get_all_embeddings()
        ids_array = np.array(self._ids, dtype=object)
        metadata_array = np.array(self._metadata, dtype=object)

        np.savez(
            filepath,
            embeddings=embeddings_matrix,
            ids=ids_array,
            metadata=metadata_array,
            dimension=np.array([self.dimension]),
        )

    @classmethod
    def load(cls, path: str, use_faiss: bool = False) -> "EmbeddingStore":
        """
        Load an embedding store from a .npz file on disk.

        Parameters
        ----------
        path : str
            Path to the .npz file.
        use_faiss : bool
            Whether to use FAISS for the loaded store.

        Returns
        -------
        EmbeddingStore
            A fully reconstructed embedding store.
        """
        data = np.load(path, allow_pickle=True)

        dimension = int(data["dimension"][0])
        store = cls(dimension=dimension, use_faiss=use_faiss)

        ids = data["ids"]
        metadata = data["metadata"]
        embeddings = data["embeddings"]

        for i in range(len(ids)):
            recording_id = str(ids[i])
            emb = embeddings[i]
            meta = metadata[i] if isinstance(metadata[i], dict) else dict(metadata[i])

            # Embeddings are already normalized from the save, but add() re-normalizes
            # so we bypass add() to avoid double-normalization overhead
            store._ids.append(recording_id)
            store._metadata.append(meta)
            store._embeddings.append(emb.astype(np.float32))
            store._id_to_index[recording_id] = i

            if store._faiss_index is not None:
                store._faiss_index.add(emb.reshape(1, -1).astype(np.float32))

        return store

    def remove(self, recording_id: str) -> None:
        """
        Remove a recording from the store.

        Note: This operation is O(n) and rebuilds internal indices.
        For FAISS-backed stores, the FAISS index is rebuilt from scratch.

        Parameters
        ----------
        recording_id : str
            ID of the recording to remove.

        Raises
        ------
        KeyError
            If the recording_id is not found.
        """
        if recording_id not in self._id_to_index:
            raise KeyError(f"Recording '{recording_id}' not found in the store")

        idx = self._id_to_index[recording_id]

        self._ids.pop(idx)
        self._metadata.pop(idx)
        self._embeddings.pop(idx)

        # Rebuild index mapping
        self._id_to_index = {rid: i for i, rid in enumerate(self._ids)}

        # Rebuild FAISS index if applicable
        if self._faiss_index is not None:
            import faiss  # type: ignore

            self._faiss_index = faiss.IndexFlatIP(self.dimension)
            if self._embeddings:
                matrix = np.array(self._embeddings, dtype=np.float32)
                self._faiss_index.add(matrix)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, recording_id: str) -> bool:
        return recording_id in self._id_to_index
