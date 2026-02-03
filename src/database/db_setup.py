"""Database setup and connection management for RAG Clinical Literature Demo."""

import os
import logging
from typing import Optional
import psycopg2
from psycopg2.extensions import connection as PgConnection
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_connection() -> PgConnection:
    """Create and return a PostgreSQL database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "rag_clinical"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )


def init_database() -> None:
    """Initialize database with required tables and extensions."""
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector extension enabled")

        # Create articles table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                title VARCHAR(500),
                authors TEXT,
                journal VARCHAR(300),
                year INTEGER,
                pdf_path VARCHAR(500),
                full_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("Articles table created")

        # Create chunks table with vector embedding
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("Chunks table created")

        # Create vector similarity index for fast retrieval
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        logger.info("Vector similarity index created")

        conn.commit()
        logger.info("Database initialization complete")

    except Exception as e:
        conn.rollback()
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def reset_database() -> None:
    """Drop and recreate all tables (for reindexing)."""
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
        cur.execute("DROP TABLE IF EXISTS articles CASCADE;")
        conn.commit()
        logger.info("Tables dropped")
    finally:
        cur.close()
        conn.close()

    init_database()


def get_stats() -> dict:
    """Get database statistics."""
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM articles;")
        article_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM chunks;")
        chunk_count = cur.fetchone()[0]

        return {
            "articles": article_count,
            "chunks": chunk_count
        }
    finally:
        cur.close()
        conn.close()
