"""BM25 keyword search implementation."""

import re
import logging
from typing import List, Optional
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 index for keyword-based retrieval."""

    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.chunks: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def build_index(self, chunks: List[dict]) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries with chunk_text
        """
        self.chunks = chunks
        self.tokenized_corpus = [
            self._tokenize(chunk['chunk_text'])
            for chunk in chunks
        ]

        self.index = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Built BM25 index with {len(chunks)} documents")

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Search for relevant chunks using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of chunks with BM25 scores
        """
        if self.index is None:
            logger.warning("BM25 index not built")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                chunk = self.chunks[idx].copy()
                chunk['bm25_score'] = float(scores[idx])
                results.append(chunk)

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        # Remove very short tokens and stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'it', 'its'
        }

        tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
        return tokens

    def is_ready(self) -> bool:
        """Check if index is built and ready."""
        return self.index is not None and len(self.chunks) > 0
