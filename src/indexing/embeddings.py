"""Embedding generation using Ollama with nomic-embed-text model."""

import os
import time
import logging
from typing import List
import numpy as np
import ollama

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_TEXT_CHARS = 6000  # Safety limit to stay within model context


def get_embedding(text: str) -> np.ndarray:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Numpy array of shape (768,)
    """
    # Truncate text if too long
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        logger.debug(f"Truncated text to {MAX_TEXT_CHARS} chars")

    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=text
            )
            return np.array(response['embedding'], dtype=np.float32)

        except Exception as e:
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed to generate embedding after {MAX_RETRIES} attempts: {e}")


def get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of texts to embed

    Returns:
        List of numpy arrays, each of shape (768,)
    """
    embeddings = []

    # Process in batches
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = []

        for text in batch:
            embedding = get_embedding(text)
            batch_embeddings.append(embedding)

        embeddings.extend(batch_embeddings)

        # Log progress
        processed = min(i + BATCH_SIZE, len(texts))
        logger.debug(f"Generated embeddings: {processed}/{len(texts)}")

    return embeddings


def check_ollama_model() -> bool:
    """Check if the embedding model is available."""
    try:
        # Try a simple embedding to verify model is loaded
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt="test"
        )
        return len(response.get('embedding', [])) == EMBEDDING_DIM
    except Exception as e:
        logger.error(f"Embedding model check failed: {e}")
        return False
