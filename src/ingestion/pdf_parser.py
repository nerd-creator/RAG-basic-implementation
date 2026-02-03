"""PDF parsing module for extracting text from clinical literature PDFs."""

import os
import logging
from pathlib import Path
from typing import Generator
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract full text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with filename, full_text, and page_count
    """
    try:
        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text with better handling of multi-column layouts
            text = page.get_text("text", sort=True)
            text_parts.append(text)

        full_text = "\n\n".join(text_parts)

        # Clean up common PDF artifacts
        full_text = _clean_text(full_text)

        result = {
            "filename": os.path.basename(pdf_path),
            "full_text": full_text,
            "page_count": len(doc)
        }

        doc.close()
        logger.info(f"Extracted {len(full_text)} chars from {pdf_path}")
        return result

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        raise


def _clean_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    # Join with single newlines, paragraphs separated by double
    text = '\n'.join(cleaned_lines)

    # Fix common hyphenation issues (word- continuation)
    import re
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    return text


def process_pdf_directory(pdf_dir: str) -> Generator[dict, None, None]:
    """
    Process all PDFs in a directory.

    Args:
        pdf_dir: Path to directory containing PDFs

    Yields:
        Dictionary for each PDF with filename, full_text, page_count
    """
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        logger.warning(f"PDF directory does not exist: {pdf_dir}")
        return

    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        try:
            yield extract_text_from_pdf(str(pdf_file))
        except Exception as e:
            logger.error(f"Skipping {pdf_file}: {e}")
            continue


def get_pdf_count(pdf_dir: str) -> int:
    """Get count of PDF files in directory."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        return 0
    return len(list(pdf_path.glob("*.pdf")))
