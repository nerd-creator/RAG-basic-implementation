"""Hybrid retriever combining BM25 and vector search."""

import time
import logging
from typing import List, Tuple
from .bm25 import BM25Index
from ..indexing.embeddings import get_embedding
from ..indexing.vector_store import similarity_search, get_all_chunks

logger = logging.getLogger(__name__)

# Weights for hybrid fusion
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7


class HybridRetriever:
    """Hybrid retrieval combining keyword (BM25) and semantic (vector) search."""

    def __init__(self):
        self.bm25_index = BM25Index()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize retriever by loading chunks and building BM25 index."""
        chunks = get_all_chunks()

        if not chunks:
            logger.warning("No chunks found in database")
            return

        self.bm25_index.build_index(chunks)
        self._initialized = True
        logger.info(f"Hybrid retriever initialized with {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> Tuple[List[dict], float]:
        """
        Perform hybrid search combining BM25 and vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Tuple of (results list, retrieval time in seconds)
        """
        start_time = time.time()

        if not self._initialized:
            self.initialize()

        # Get vector search results
        query_embedding = get_embedding(query)
        vector_results = similarity_search(query_embedding, top_k=top_k * 2)

        # Get BM25 search results
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

        # Normalize and fuse scores
        fused_results = self._fuse_results(vector_results, bm25_results, top_k)

        retrieval_time = time.time() - start_time
        return fused_results, retrieval_time

    def _fuse_results(
        self,
        vector_results: List[dict],
        bm25_results: List[dict],
        top_k: int
    ) -> List[dict]:
        """Fuse vector and BM25 results with weighted scoring."""

        # Normalize scores
        vector_scores = self._normalize_scores(
            [r.get('similarity', 0) for r in vector_results]
        )
        bm25_scores = self._normalize_scores(
            [r.get('bm25_score', 0) for r in bm25_results]
        )

        # Create score maps by chunk_id
        score_map = {}

        for i, result in enumerate(vector_results):
            chunk_id = result['chunk_id']
            score_map[chunk_id] = {
                'data': result,
                'vector_score': vector_scores[i] if i < len(vector_scores) else 0,
                'bm25_score': 0
            }

        for i, result in enumerate(bm25_results):
            chunk_id = result['chunk_id']
            if chunk_id in score_map:
                score_map[chunk_id]['bm25_score'] = bm25_scores[i] if i < len(bm25_scores) else 0
            else:
                score_map[chunk_id] = {
                    'data': result,
                    'vector_score': 0,
                    'bm25_score': bm25_scores[i] if i < len(bm25_scores) else 0
                }

        # Calculate fused scores
        for chunk_id, scores in score_map.items():
            scores['fused_score'] = (
                VECTOR_WEIGHT * scores['vector_score'] +
                BM25_WEIGHT * scores['bm25_score']
            )

        # Sort by fused score and return top_k
        sorted_items = sorted(
            score_map.items(),
            key=lambda x: x[1]['fused_score'],
            reverse=True
        )[:top_k]

        results = []
        for chunk_id, scores in sorted_items:
            result = scores['data'].copy()
            result['fused_score'] = scores['fused_score']
            result['vector_score'] = scores['vector_score']
            result['bm25_score'] = scores['bm25_score']
            results.append(result)

        return results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def is_ready(self) -> bool:
        """Check if retriever is ready."""
        return self._initialized and self.bm25_index.is_ready()
