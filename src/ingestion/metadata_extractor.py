"""Metadata extraction from clinical literature PDFs."""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_metadata(filename: str, full_text: str) -> dict:
    """
    Extract metadata from PDF filename and content.

    Args:
        filename: Original PDF filename
        full_text: Extracted text from PDF

    Returns:
        Dictionary with title, authors, journal, year
    """
    metadata = {
        "title": _extract_title(filename, full_text),
        "authors": _extract_authors(full_text),
        "journal": _extract_journal(full_text),
        "year": _extract_year(filename, full_text)
    }

    logger.info(f"Extracted metadata for {filename}: title='{metadata['title'][:50]}...'")
    return metadata


def _extract_title(filename: str, text: str) -> str:
    """Extract title from text or filename."""
    # Try to get title from first few lines (usually largest/bold text comes first)
    lines = text.split('\n')[:20]

    # Look for a substantial line that could be a title (not too short, not too long)
    for line in lines:
        line = line.strip()
        # Title is usually 10-200 chars, starts with capital, no common noise
        if 10 < len(line) < 200 and line[0].isupper():
            # Skip lines that look like headers/footers
            if not any(skip in line.lower() for skip in ['abstract', 'introduction', 'doi:', 'volume', 'issue']):
                return line

    # Fallback to filename (remove extension, replace underscores)
    title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    return title.strip()


def _extract_authors(text: str) -> Optional[str]:
    """Extract author names from text."""
    # Common author patterns in academic papers
    patterns = [
        # "John Smith, Jane Doe, Bob Wilson"
        r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)+)',
        # "J. Smith, J. Doe"
        r'^([A-Z]\.\s*[A-Z][a-z]+(?:\s*,\s*[A-Z]\.\s*[A-Z][a-z]+)+)',
        # Author section
        r'Authors?:\s*(.+?)(?:\n|$)',
    ]

    lines = text.split('\n')[:30]  # Authors usually in first 30 lines

    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                authors = match.group(1).strip()
                # Clean up and limit length
                if len(authors) > 10 and len(authors) < 500:
                    return authors

    return None


def _extract_journal(text: str) -> Optional[str]:
    """Extract journal name from text."""
    # Common journal name patterns
    patterns = [
        r'(?:Published in|Journal[:\s]+)([A-Z][^,\n]+)',
        r'([A-Z][a-z]+\s+(?:Journal|Review|Letters|Medicine|Research|Science)[^,\n]*)',
    ]

    lines = text.split('\n')[:50]

    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                journal = match.group(1).strip()
                if 5 < len(journal) < 200:
                    return journal

    return None


def _extract_year(filename: str, text: str) -> Optional[int]:
    """Extract publication year from filename or text."""
    # Look for 4-digit year in reasonable range
    year_pattern = r'\b(20[0-2][0-9]|201[0-9])\b'

    # Try filename first
    match = re.search(year_pattern, filename)
    if match:
        return int(match.group(1))

    # Then first 50 lines of text
    lines = text.split('\n')[:50]
    text_sample = '\n'.join(lines)

    matches = re.findall(year_pattern, text_sample)
    if matches:
        # Return the most common year, preferring recent ones
        years = [int(y) for y in matches]
        # Prefer years 2015-2025
        recent_years = [y for y in years if 2015 <= y <= 2025]
        if recent_years:
            return max(set(recent_years), key=recent_years.count)
        return max(set(years), key=years.count)

    return None
