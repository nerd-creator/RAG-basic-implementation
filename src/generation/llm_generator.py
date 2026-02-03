"""LLM-based answer generation using Ollama."""

import time
import logging
from typing import List, Generator, Tuple
import ollama

logger = logging.getLogger(__name__)

LLM_MODEL = "llama3.2:1b"
MAX_CONTEXT_CHARS = 2000  # Limit context size for faster generation
MAX_CHUNK_CHARS = 600  # Limit each chunk

PROMPT_TEMPLATE = """Answer based on the context. Cite sources as [Title, Year].

Context:
{context}

Question: {query}

Answer:"""


def format_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into context string."""
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks[:3], 1):  # Max 3 chunks
        title = chunk.get('title', 'Unknown')[:50]  # Truncate title
        year = chunk.get('year', 'N/A')
        text = chunk.get('chunk_text', '')[:MAX_CHUNK_CHARS]  # Truncate chunk

        part = f"[{i}] {title} ({year}): {text}"

        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n\n".join(context_parts)


def generate_answer(query: str, chunks: List[dict]) -> Tuple[str, float, List[dict]]:
    """
    Generate an answer using LLM with retrieved context.

    Args:
        query: User's question
        chunks: Retrieved context chunks

    Returns:
        Tuple of (answer text, generation time, citations)
    """
    start_time = time.time()

    context = format_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    try:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={
                'temperature': 0.5,
                'top_p': 0.9,
                'num_predict': 256,  # Limit response length
            }
        )

        answer = response['response']
        generation_time = time.time() - start_time

        # Extract citations from chunks
        citations = extract_citations(chunks)

        return answer, generation_time, citations

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


def generate_answer_stream(query: str, chunks: List[dict]) -> Generator[str, None, Tuple[float, List[dict]]]:
    """
    Stream answer generation for better UX.

    Args:
        query: User's question
        chunks: Retrieved context chunks

    Yields:
        Answer tokens as they're generated

    Returns:
        Tuple of (generation time, citations) after completion
    """
    start_time = time.time()

    context = format_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    try:
        stream = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            stream=True,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
            }
        )

        for chunk in stream:
            if 'response' in chunk:
                yield chunk['response']

        generation_time = time.time() - start_time
        citations = extract_citations(chunks)

        return generation_time, citations

    except Exception as e:
        logger.error(f"Stream generation failed: {e}")
        raise


def extract_citations(chunks: List[dict]) -> List[dict]:
    """Extract citation information from chunks."""
    seen_titles = set()
    citations = []

    for chunk in chunks:
        title = chunk.get('title', 'Unknown')
        if title not in seen_titles:
            seen_titles.add(title)
            citations.append({
                'title': title,
                'authors': chunk.get('authors'),
                'year': chunk.get('year')
            })

    return citations[:3]  # Top 3 citations


def check_llm_model() -> bool:
    """Check if the LLM model is available."""
    try:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt="Hi",
            options={'num_predict': 1}
        )
        return 'response' in response
    except Exception as e:
        logger.error(f"LLM model check failed: {e}")
        return False
