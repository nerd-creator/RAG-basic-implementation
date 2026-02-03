"""Text chunking module for splitting documents into semantic chunks."""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4
TARGET_TOKENS = 300  # Reduced for nomic-embed-text context limit
OVERLAP_TOKENS = 30

TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # ~1200 chars
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN
MAX_CHUNK_CHARS = 6000  # Hard limit to stay within embedding model context


def chunk_text(text: str, article_id: int) -> List[dict]:
    """
    Split text into semantic chunks with overlap.

    Args:
        text: Full text to chunk
        article_id: ID of the source article

    Returns:
        List of chunk dictionaries with article_id, chunk_text, chunk_index
    """
    # Split into sentences first
    sentences = _split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence exceeds target, save chunk and start new one
        if current_length + sentence_length > TARGET_CHARS and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "article_id": article_id,
                "chunk_text": chunk_text,
                "chunk_index": chunk_index
            })
            chunk_index += 1

            # Keep overlap from end of current chunk
            overlap_text = _get_overlap(current_chunk, OVERLAP_CHARS)
            current_chunk = [overlap_text] if overlap_text else []
            current_length = len(overlap_text) if overlap_text else 0

        # Truncate very long sentences
        if sentence_length > MAX_CHUNK_CHARS:
            sentence = sentence[:MAX_CHUNK_CHARS]
            sentence_length = MAX_CHUNK_CHARS

        current_chunk.append(sentence)
        current_length += sentence_length

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        # Truncate if still too long
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS]
        if len(chunk_text.strip()) > 50:  # Only if substantial
            chunks.append({
                "article_id": article_id,
                "chunk_text": chunk_text,
                "chunk_index": chunk_index
            })

    logger.info(f"Created {len(chunks)} chunks for article {article_id}")
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting on common terminators
    # Handle abbreviations and decimals to avoid false splits
    text = re.sub(r'([.!?])\s+', r'\1|SPLIT|', text)

    # Don't split on common abbreviations
    text = re.sub(r'(Dr|Mr|Mrs|Ms|Prof|et al|vs|Fig|fig|i\.e|e\.g)\|SPLIT\|', r'\1. ', text)

    sentences = text.split('|SPLIT|')

    # Clean up sentences
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s and len(s) > 10:  # Skip very short fragments
            cleaned.append(s)

    return cleaned


def _get_overlap(chunks: List[str], target_chars: int) -> str:
    """Get overlap text from end of chunk list."""
    overlap_parts = []
    total_length = 0

    for chunk in reversed(chunks):
        overlap_parts.insert(0, chunk)
        total_length += len(chunk)
        if total_length >= target_chars:
            break

    return ' '.join(overlap_parts)
