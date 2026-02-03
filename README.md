# RAG Clinical Literature Demo

A Retrieval-Augmented Generation (RAG) pipeline for querying clinical research literature using hybrid search and local LLMs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Retriever                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   BM25 Search    │    │  Vector Search   │                   │
│  │   (Keywords)     │    │   (Semantic)     │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │     0.3 weight        │    0.7 weight               │
│           └───────────┬───────────┘                             │
│                       ▼                                          │
│              Score Fusion & Ranking                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Generator (llama3)                        │
│                 Context + Query → Answer                         │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **PostgreSQL** with pgvector extension
- **Ollama** with the following models:
  - `llama3.2:1b` (for answer generation - small model for limited RAM)
  - `nomic-embed-text` (for embeddings)
- **Python 3.10+**

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup PostgreSQL Database

```bash
# Create database
createdb rag_clinical

# Enable pgvector extension (in psql)
psql -d rag_clinical -c "CREATE EXTENSION vector;"
```

### 3. Pull Ollama Models

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database credentials if needed
```

### 5. Add PDF Files

Place your clinical literature PDFs in the `data/pdfs/` directory.

### 6. Run Demo

```bash
python demo.py
```

## Usage

```
=== RAG Clinical Literature Demo ===

Checking prerequisites...
  - Checking nomic-embed-text model... OK
  - Checking llama3 model... OK

Initializing database...
  Database ready

Processing 5 PDFs from data/pdfs/...
  [████████████████████████████████] 5/5
  Generated 234 chunks with embeddings

Building search index...
  Search index ready

Ready for queries!
Commands: 'exit' to quit, 'reindex' to reprocess PDFs, 'stats' for info
--------------------------------------------------

Query: What are the biomarkers for breast cancer?

[Retrieving...] Found 5 relevant chunks (0.32s)
[Generating...]

Answer:
------------------------------------------
Based on the research literature, key biomarkers for breast cancer include...
------------------------------------------

Sources:
  [Source: Molecular Profiling of Breast Cancer, 2023]
  [Source: Biomarker Discovery in Oncology, 2022]

Retrieval: 0.32s | Generation: 2.14s

Show retrieved chunks? (y/n):
```

## Example Queries

- "What are the risk factors for cardiovascular disease?"
- "Describe the mechanism of action of checkpoint inhibitors"
- "What biomarkers are used for early cancer detection?"
- "Summarize the latest findings on diabetes treatment"

## CLI Commands

| Command   | Description                      |
|-----------|----------------------------------|
| `exit`    | Quit the application             |
| `reindex` | Reprocess all PDFs               |
| `stats`   | Show database statistics         |

## Project Structure

```
rag-clinical-demo/
├── data/
│   └── pdfs/               # Place PDFs here
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py       # PDF text extraction
│   │   └── metadata_extractor.py
│   ├── indexing/
│   │   ├── chunker.py          # Text chunking
│   │   ├── embeddings.py       # Ollama embeddings
│   │   └── vector_store.py     # pgvector operations
│   ├── retrieval/
│   │   ├── hybrid_retriever.py # Combined search
│   │   └── bm25.py             # Keyword search
│   ├── generation/
│   │   └── llm_generator.py    # LLM answer generation
│   └── database/
│       └── db_setup.py         # PostgreSQL setup
├── demo.py                 # Main CLI application
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Environment variables (`.env`):

| Variable      | Default                    | Description          |
|---------------|----------------------------|----------------------|
| DB_HOST       | localhost                  | PostgreSQL host      |
| DB_PORT       | 5432                       | PostgreSQL port      |
| DB_NAME       | rag_clinical               | Database name        |
| DB_USER       | postgres                   | Database user        |
| DB_PASSWORD   | postgres                   | Database password    |
| OLLAMA_HOST   | http://localhost:11434     | Ollama API endpoint  |

## Technical Details

- **Chunking**: ~500 tokens per chunk with 50-token overlap
- **Embeddings**: 768-dimensional vectors (nomic-embed-text)
- **Hybrid Search**: 30% BM25 + 70% vector similarity
- **Vector Index**: IVFFlat with 100 lists for fast retrieval

## Troubleshooting

### "Database connection failed"
- Ensure PostgreSQL is running
- Check credentials in `.env`
- Verify database exists: `psql -l | grep rag_clinical`

### "Embedding model not available"
- Run: `ollama pull nomic-embed-text`
- Verify: `ollama list`

### "No PDFs found"
- Add PDF files to `data/pdfs/` directory
- Run `reindex` command in the CLI

## License

MIT
