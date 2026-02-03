#!/usr/bin/env python3
"""
RAG Clinical Literature Demo

Interactive CLI for querying clinical research literature using
hybrid retrieval (BM25 + vector search) and LLM generation.
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.db_setup import init_database, reset_database, get_stats
from src.ingestion.pdf_parser import process_pdf_directory, get_pdf_count
from src.ingestion.metadata_extractor import extract_metadata
from src.indexing.chunker import chunk_text
from src.indexing.embeddings import get_embeddings_batch, check_ollama_model
from src.indexing.vector_store import insert_article, insert_chunks
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_generator import generate_answer, check_llm_model

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PDF_DIR = Path(__file__).parent / "data" / "pdfs"


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 50)
    print("       RAG Clinical Literature Demo")
    print("=" * 50 + "\n")


def print_progress(current: int, total: int, prefix: str = ""):
    """Print a simple progress bar."""
    bar_length = 30
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r{prefix}[{bar}] {current}/{total}", end="", flush=True)
    if current == total:
        print()


def check_prerequisites() -> bool:
    """Check that required services are available."""
    print("Checking prerequisites...")

    # Check embedding model
    print("  - Checking nomic-embed-text model...", end=" ")
    if not check_ollama_model():
        print("FAILED")
        print("\nError: nomic-embed-text model not available.")
        print("Run: ollama pull nomic-embed-text")
        return False
    print("OK")

    # Check LLM model
    print("  - Checking llama3 model...", end=" ")
    if not check_llm_model():
        print("FAILED")
        print("\nError: llama3 model not available.")
        print("Run: ollama pull llama3")
        return False
    print("OK")

    return True


def process_pdfs() -> int:
    """Process all PDFs and return chunk count."""
    pdf_count = get_pdf_count(str(PDF_DIR))

    if pdf_count == 0:
        print(f"\nNo PDFs found in {PDF_DIR}")
        print("Please add PDF files and run 'reindex'")
        return 0

    print(f"\nProcessing {pdf_count} PDFs from {PDF_DIR}...")

    total_chunks = 0
    processed = 0

    for pdf_data in process_pdf_directory(str(PDF_DIR)):
        processed += 1
        print_progress(processed, pdf_count, "  ")

        # Extract metadata
        metadata = extract_metadata(pdf_data['filename'], pdf_data['full_text'])

        # Insert article
        article_id = insert_article(
            metadata,
            pdf_data['full_text'],
            str(PDF_DIR / pdf_data['filename'])
        )

        # Create chunks
        chunks = chunk_text(pdf_data['full_text'], article_id)

        if chunks:
            # Generate embeddings
            chunk_texts = [c['chunk_text'] for c in chunks]
            embeddings = get_embeddings_batch(chunk_texts)

            # Store chunks with embeddings
            insert_chunks(article_id, chunks, embeddings)
            total_chunks += len(chunks)

    print(f"  Generated {total_chunks} chunks with embeddings")
    return total_chunks


def show_stats():
    """Display database statistics."""
    stats = get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Articles: {stats['articles']}")
    print(f"  Chunks:   {stats['chunks']}")


def query_loop(retriever: HybridRetriever):
    """Main query interaction loop."""
    print("\nReady for queries!")
    print("Commands: 'exit' to quit, 'reindex' to reprocess PDFs, 'stats' for info")
    print("-" * 50)

    while True:
        try:
            query = input("\nQuery: ").strip()

            if not query:
                continue

            if query.lower() == 'exit':
                print("Goodbye!")
                break

            if query.lower() == 'stats':
                show_stats()
                continue

            if query.lower() == 'reindex':
                print("\nReindexing...")
                reset_database()
                chunk_count = process_pdfs()
                if chunk_count > 0:
                    retriever.initialize()
                    print("Reindexing complete!")
                continue

            # Perform retrieval
            print("\n[Retrieving...]", end=" ", flush=True)
            results, retrieval_time = retriever.search(query, top_k=3)

            if not results:
                print("No relevant documents found.")
                continue

            print(f"Found {len(results)} relevant chunks ({retrieval_time:.2f}s)")

            # Generate answer
            print("[Generating...]\n")
            answer, gen_time, citations = generate_answer(query, results)

            # Display answer
            print("Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)

            # Display citations
            if citations:
                print("\nSources:")
                for cit in citations:
                    year = cit.get('year', 'N/A')
                    print(f"  [Source: {cit['title']}, {year}]")

            print(f"\nRetrieval: {retrieval_time:.2f}s | Generation: {gen_time:.2f}s")

            # Offer to show chunks
            show_chunks = input("\nShow retrieved chunks? (y/n): ").strip().lower()
            if show_chunks == 'y':
                print("\nRetrieved Chunks:")
                for i, chunk in enumerate(results, 1):
                    print(f"\n[{i}] {chunk.get('title', 'Unknown')} (score: {chunk.get('fused_score', 0):.3f})")
                    print(f"    {chunk['chunk_text'][:300]}...")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Query error: {e}")
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    print_banner()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Initialize database
    print("\nInitializing database...")
    try:
        init_database()
        print("  Database ready")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        print("\nMake sure PostgreSQL is running and the database exists:")
        print("  createdb rag_clinical")
        sys.exit(1)

    # Check if we need to process PDFs
    stats = get_stats()

    if stats['chunks'] == 0:
        chunk_count = process_pdfs()
        if chunk_count == 0:
            print("\nNo documents indexed. Add PDFs to data/pdfs/ and run 'reindex'")
    else:
        print(f"\nLoaded existing index: {stats['articles']} articles, {stats['chunks']} chunks")

    # Initialize retriever
    print("\nBuilding search index...")
    retriever = HybridRetriever()
    retriever.initialize()

    if retriever.is_ready():
        print("  Search index ready")
    else:
        print("  Warning: Search index empty")

    # Start query loop
    query_loop(retriever)


if __name__ == "__main__":
    main()
