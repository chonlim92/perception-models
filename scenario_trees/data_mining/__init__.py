"""
Data mining module for Functional Scenario Trees.

Provides tools for discovering corner cases, analyzing scenario coverage,
scoring difficulty, and clustering recordings by similarity.

Components:
    EmbeddingStore: Store and search scene embeddings via cosine similarity
    NoveltyDetector: Identify unusual or rare scenarios
    CoverageAnalyzer: Analyze scenario coverage and identify gaps
    DifficultyScorer: Score recordings by difficulty level
    ScenarioClusterAnalyzer: Cluster recordings into scenario groups
"""

from .embedding_store import EmbeddingStore, SearchResult
from .novelty_detector import NoveltyDetector, NoveltyResult
from .coverage_analyzer import CoverageAnalyzer, CoverageGap, CollectionPriority
from .difficulty_scorer import DifficultyScorer
from .cluster_analysis import ScenarioClusterAnalyzer

__all__ = [
    "EmbeddingStore",
    "SearchResult",
    "NoveltyDetector",
    "NoveltyResult",
    "CoverageAnalyzer",
    "CoverageGap",
    "CollectionPriority",
    "DifficultyScorer",
    "ScenarioClusterAnalyzer",
]
