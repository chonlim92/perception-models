"""
Query engine for searching recordings by scenario criteria.

Supports structured queries, composite boolean expressions, temporal queries,
and similarity-based search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..taxonomy.scenario_schema import ScenarioQuery
from .database import ScenarioDatabase


class EmbeddingStore(Protocol):
    """Protocol for embedding-based similarity search backends."""

    def get_embedding(self, recording_id: str) -> Optional[List[float]]:
        """Get the embedding vector for a recording."""
        ...

    def find_nearest(
        self, embedding: List[float], top_k: int
    ) -> List[tuple[str, float]]:
        """Find top_k nearest recordings by embedding distance. Returns (id, distance) pairs."""
        ...


class _TokenType(Enum):
    """Token types for boolean expression parsing."""

    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    TERM = auto()
    EOF = auto()


@dataclass
class _Token:
    """A single token in a boolean expression."""

    type: _TokenType
    value: str


class _ASTNode:
    """Base class for AST nodes."""
    pass


@dataclass
class _TermNode(_ASTNode):
    """A terminal node representing a scenario term."""

    term: str


@dataclass
class _NotNode(_ASTNode):
    """A NOT operation node."""

    child: _ASTNode


@dataclass
class _BinaryNode(_ASTNode):
    """A binary operation node (AND/OR)."""

    op: _TokenType
    left: _ASTNode
    right: _ASTNode


class _Tokenizer:
    """Tokenize boolean expressions like 'pedestrian AND rain AND NOT highway'."""

    _KEYWORDS = {
        "AND": _TokenType.AND,
        "OR": _TokenType.OR,
        "NOT": _TokenType.NOT,
    }

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._tokens: List[_Token] = []
        self._tokenize()

    def _tokenize(self) -> None:
        """Parse the input text into tokens."""
        while self._pos < len(self._text):
            ch = self._text[self._pos]

            # Skip whitespace
            if ch.isspace():
                self._pos += 1
                continue

            # Parentheses
            if ch == "(":
                self._tokens.append(_Token(_TokenType.LPAREN, "("))
                self._pos += 1
                continue
            if ch == ")":
                self._tokens.append(_Token(_TokenType.RPAREN, ")"))
                self._pos += 1
                continue

            # Words (keywords or terms)
            if ch.isalnum() or ch in ("_", "-", ".", "/"):
                start = self._pos
                while self._pos < len(self._text) and (
                    self._text[self._pos].isalnum()
                    or self._text[self._pos] in ("_", "-", ".", "/")
                ):
                    self._pos += 1
                word = self._text[start : self._pos]
                upper = word.upper()
                if upper in self._KEYWORDS:
                    self._tokens.append(_Token(self._KEYWORDS[upper], upper))
                else:
                    self._tokens.append(_Token(_TokenType.TERM, word))
                continue

            # Skip unknown characters
            self._pos += 1

        self._tokens.append(_Token(_TokenType.EOF, ""))

    @property
    def tokens(self) -> List[_Token]:
        return self._tokens


class _Parser:
    """
    Recursive descent parser for boolean expressions.

    Grammar:
        expr     -> or_expr
        or_expr  -> and_expr (OR and_expr)*
        and_expr -> not_expr (AND not_expr)*
        not_expr -> NOT not_expr | primary
        primary  -> TERM | LPAREN expr RPAREN
    """

    def __init__(self, tokens: List[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _current(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _expect(self, token_type: _TokenType) -> _Token:
        token = self._current()
        if token.type != token_type:
            raise ValueError(
                f"Expected {token_type.name}, got {token.type.name} ('{token.value}')"
            )
        return self._advance()

    def parse(self) -> _ASTNode:
        """Parse the token stream into an AST."""
        node = self._or_expr()
        if self._current().type != _TokenType.EOF:
            raise ValueError(
                f"Unexpected token after expression: '{self._current().value}'"
            )
        return node

    def _or_expr(self) -> _ASTNode:
        left = self._and_expr()
        while self._current().type == _TokenType.OR:
            self._advance()
            right = self._and_expr()
            left = _BinaryNode(op=_TokenType.OR, left=left, right=right)
        return left

    def _and_expr(self) -> _ASTNode:
        left = self._not_expr()
        while self._current().type == _TokenType.AND:
            self._advance()
            right = self._not_expr()
            left = _BinaryNode(op=_TokenType.AND, left=left, right=right)
        return left

    def _not_expr(self) -> _ASTNode:
        if self._current().type == _TokenType.NOT:
            self._advance()
            child = self._not_expr()
            return _NotNode(child=child)
        return self._primary()

    def _primary(self) -> _ASTNode:
        token = self._current()
        if token.type == _TokenType.TERM:
            self._advance()
            return _TermNode(term=token.value)
        if token.type == _TokenType.LPAREN:
            self._advance()
            node = self._or_expr()
            self._expect(_TokenType.RPAREN)
            return node
        raise ValueError(
            f"Unexpected token in expression: '{token.value}' ({token.type.name})"
        )


class ScenarioQueryEngine:
    """
    Query engine for searching recordings by scenario criteria.

    Supports structured queries via ScenarioQuery objects, composite boolean
    expressions, temporal queries, and similarity-based search.
    """

    def __init__(self, database: ScenarioDatabase) -> None:
        """
        Initialize the query engine.

        Args:
            database: The ScenarioDatabase instance to query against.
        """
        self.database = database

    def query(self, criteria: ScenarioQuery) -> List[str]:
        """
        Find recording IDs matching the given ScenarioQuery criteria.

        Applies required_tags (AND), excluded_tags (NOT), min_confidence,
        and layer_filters to find matching recordings.

        Args:
            criteria: A ScenarioQuery object specifying the search criteria.

        Returns:
            List of recording IDs that satisfy all criteria.
        """
        all_recordings = self.database.get_all_recordings()

        if not all_recordings:
            return []

        # Start with all recording IDs
        candidate_ids = {rec["id"] for rec in all_recordings}

        # Apply required_tags: each required tag must be present with min_confidence
        for required_node_id in criteria.required_tags:
            matching = set(
                self.database.get_recordings_with_tag(
                    required_node_id, criteria.min_confidence
                )
            )
            candidate_ids &= matching

        if not candidate_ids:
            return []

        # Apply excluded_tags: remove recordings that have any excluded tag
        for excluded_node_id in criteria.excluded_tags:
            excluded = set(
                self.database.get_recordings_with_tag(excluded_node_id, 0.0)
            )
            candidate_ids -= excluded

        if not candidate_ids:
            return []

        # Apply layer_filters if specified: only keep recordings that have at least
        # one tag in the specified layers with min_confidence
        if criteria.layer_filters:
            layer_filtered: set[str] = set()
            for rec_id in candidate_ids:
                tags = self.database.get_tags_for_recording(rec_id)
                for tag in tags:
                    if tag["confidence"] >= criteria.min_confidence:
                        layer = _extract_layer_from_node_id(tag["node_id"])
                        if layer in criteria.layer_filters:
                            layer_filtered.add(rec_id)
                            break
            candidate_ids &= layer_filtered

        return sorted(candidate_ids)

    def query_composite(self, expression: str) -> List[str]:
        """
        Parse and evaluate a boolean expression against the database.

        Supports AND, OR, NOT operators and parentheses.
        Terms are matched case-insensitively against node IDs and node names
        (via tags in the database).

        Example: "pedestrian AND rain AND NOT highway"

        Args:
            expression: Boolean expression string.

        Returns:
            List of recording IDs matching the expression.
        """
        tokenizer = _Tokenizer(expression)
        parser = _Parser(tokenizer.tokens)
        ast = parser.parse()

        # Get all recordings
        all_recordings = self.database.get_all_recordings()
        all_ids = {rec["id"] for rec in all_recordings}

        # Build a lookup: recording_id -> set of terms (node_ids + lowered names)
        recording_terms: Dict[str, set[str]] = {}
        for rec_id in all_ids:
            tags = self.database.get_tags_for_recording(rec_id)
            terms: set[str] = set()
            for tag in tags:
                node_id = tag["node_id"]
                terms.add(node_id.lower())
                # Also add individual parts of the node_id for matching
                terms.add(node_id)
            recording_terms[rec_id] = terms

        # Evaluate the AST against each recording
        def evaluate(node: _ASTNode, rec_id: str) -> bool:
            if isinstance(node, _TermNode):
                term_lower = node.term.lower()
                rec_terms = recording_terms.get(rec_id, set())
                # Check if the term matches any node_id (exact or substring)
                for t in rec_terms:
                    if term_lower in t.lower():
                        return True
                return False
            elif isinstance(node, _NotNode):
                return not evaluate(node.child, rec_id)
            elif isinstance(node, _BinaryNode):
                left_val = evaluate(node.left, rec_id)
                if node.op == _TokenType.AND:
                    return left_val and evaluate(node.right, rec_id)
                else:  # OR
                    return left_val or evaluate(node.right, rec_id)
            return False

        results = [rec_id for rec_id in sorted(all_ids) if evaluate(ast, rec_id)]
        return results

    def query_temporal(
        self, event_type: str, time_window: float
    ) -> List[str]:
        """
        Find recordings containing temporal events of a specific type
        that occur within a given time window (in seconds).

        Uses frame_start/frame_end from tags. A tag is considered to be
        within the time window if its frame range duration (assuming 10 fps)
        is less than or equal to time_window.

        Args:
            event_type: The node_id (or substring) of the event to search for.
            time_window: Maximum duration in seconds for the event occurrence.

        Returns:
            List of recording IDs with matching temporal events.
        """
        # Assume 10 FPS for frame-to-time conversion if not otherwise specified
        fps = 10.0
        max_frames = time_window * fps

        all_recordings = self.database.get_all_recordings()
        results: List[str] = []

        event_lower = event_type.lower()

        for rec in all_recordings:
            rec_id = rec["id"]
            tags = self.database.get_tags_for_recording(rec_id)
            for tag in tags:
                node_id_lower = tag["node_id"].lower()
                if event_lower not in node_id_lower:
                    continue
                frame_start = tag.get("frame_start")
                frame_end = tag.get("frame_end")
                if frame_start is not None and frame_end is not None:
                    duration_frames = frame_end - frame_start
                    if duration_frames <= max_frames:
                        results.append(rec_id)
                        break
                else:
                    # If no frame info, include if the tag matches
                    # (the event exists in the recording)
                    results.append(rec_id)
                    break

        return sorted(results)

    def find_similar(
        self,
        recording_id: str,
        top_k: int,
        embedding_store: EmbeddingStore,
    ) -> List[str]:
        """
        Find recordings most similar to a reference recording using embeddings.

        Args:
            recording_id: The reference recording to find similar items for.
            top_k: Number of similar recordings to return.
            embedding_store: Backend providing embedding vectors and nearest-neighbor search.

        Returns:
            List of recording IDs most similar to the reference, ordered by similarity.
        """
        embedding = embedding_store.get_embedding(recording_id)
        if embedding is None:
            return []

        nearest = embedding_store.find_nearest(embedding, top_k + 1)
        # Exclude the query recording itself
        results = [
            rid for rid, _dist in nearest if rid != recording_id
        ]
        return results[:top_k]

    def count_by_attribute(self, attribute: str) -> Dict[str, int]:
        """
        Count recordings grouped by a specific attribute.

        Supported attributes:
        - 'node_id': count per scenario tag node
        - 'location': count per recording location
        - 'source': count per tag source (auto/manual/model)
        - 'layer': count per scenario layer

        Args:
            attribute: The attribute to group by.

        Returns:
            Dictionary mapping attribute values to recording counts.
        """
        if attribute == "node_id":
            stats = self.database.get_statistics()
            return stats.get("tag_counts", {})

        elif attribute == "location":
            stats = self.database.get_statistics()
            return stats.get("location_counts", {})

        elif attribute == "source":
            stats = self.database.get_statistics()
            return stats.get("source_counts", {})

        elif attribute == "layer":
            # Count distinct recordings per layer
            all_recordings = self.database.get_all_recordings()
            layer_recordings: Dict[str, set[str]] = {}
            for rec in all_recordings:
                tags = self.database.get_tags_for_recording(rec["id"])
                for tag in tags:
                    layer = _extract_layer_from_node_id(tag["node_id"])
                    layer_key = f"Layer {layer}"
                    if layer_key not in layer_recordings:
                        layer_recordings[layer_key] = set()
                    layer_recordings[layer_key].add(rec["id"])
            return {k: len(v) for k, v in sorted(layer_recordings.items())}

        else:
            raise ValueError(
                f"Unsupported attribute: '{attribute}'. "
                f"Supported: node_id, location, source, layer"
            )


def _extract_layer_from_node_id(node_id: str) -> int:
    """
    Extract the layer number from a node ID.

    Node IDs follow the pattern 'L{layer}.{sub}...' (e.g., 'L4.3.1' -> 4).
    """
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
