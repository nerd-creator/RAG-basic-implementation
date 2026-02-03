"""Vector store operations using PostgreSQL with pgvector."""

import logging
from typing import List, Optional
import numpy as np
from ..database.db_setup import get_connection

logger = logging.getLogger(__name__)


def insert_article(metadata: dict, full_text: str, pdf_path: str) -> int:
    """
    Insert an article and return its ID.

    Args:
        metadata: Dictionary with title, authors, journal, year
        full_text: Full extracted text
        pdf_path: Path to source PDF

    Returns:
        Article ID
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO articles (title, authors, journal, year, pdf_path, full_text)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            metadata.get('title'),
            metadata.get('authors'),
            metadata.get('journal'),
            metadata.get('year'),
            pdf_path,
            full_text
        ))

        article_id = cur.fetchone()[0]
        conn.commit()
        logger.info(f"Inserted article {article_id}: {metadata.get('title', 'Unknown')[:50]}")
        return article_id

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to insert article: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def insert_chunks(article_id: int, chunks: List[dict], embeddings: List[np.ndarray]) -> None:
    """
    Insert chunks with embeddings for an article.

    Args:
        article_id: ID of the parent article
        chunks: List of chunk dictionaries
        embeddings: List of embedding arrays
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have same length")

    conn = get_connection()
    cur = conn.cursor()

    try:
        for chunk, embedding in zip(chunks, embeddings):
            # Convert numpy array to list for pgvector
            embedding_list = embedding.tolist()

            cur.execute("""
                INSERT INTO chunks (article_id, chunk_text, chunk_index, embedding)
                VALUES (%s, %s, %s, %s);
            """, (
                article_id,
                chunk['chunk_text'],
                chunk['chunk_index'],
                embedding_list
            ))

        conn.commit()
        logger.info(f"Inserted {len(chunks)} chunks for article {article_id}")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to insert chunks: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def similarity_search(query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
    """
    Search for similar chunks using cosine similarity.

    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return

    Returns:
        List of dictionaries with chunk info and similarity scores
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        embedding_list = query_embedding.tolist()

        cur.execute("""
            SELECT
                c.id,
                c.chunk_text,
                c.chunk_index,
                c.article_id,
                a.title,
                a.authors,
                a.year,
                1 - (c.embedding <=> %s::vector) as similarity
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """, (embedding_list, embedding_list, top_k))

        results = []
        for row in cur.fetchall():
            results.append({
                'chunk_id': row[0],
                'chunk_text': row[1],
                'chunk_index': row[2],
                'article_id': row[3],
                'title': row[4],
                'authors': row[5],
                'year': row[6],
                'similarity': float(row[7])
            })

        return results

    finally:
        cur.close()
        conn.close()


def get_all_chunks() -> List[dict]:
    """Get all chunks for BM25 indexing."""
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                c.id,
                c.chunk_text,
                c.article_id,
                a.title,
                a.authors,
                a.year
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            ORDER BY c.id;
        """)

        results = []
        for row in cur.fetchall():
            results.append({
                'chunk_id': row[0],
                'chunk_text': row[1],
                'article_id': row[2],
                'title': row[3],
                'authors': row[4],
                'year': row[5]
            })

        return results

    finally:
        cur.close()
        conn.close()
