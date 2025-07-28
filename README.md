# Challenge 1B: Persona-Driven Document Intelligence

## Overview

https://github.com/tronmon/Adobe-India-Hackathon-2025-Challenge_1B/blob/main/assets/demo.mp4

This project implements a system that acts as an intelligent document analyst. Given a collection of PDF documents, a target persona, and a job-to-be-done, the system extracts and ranks the most relevant sections to fulfill the persona’s information needs.

It supports diverse input types (research papers, textbooks, financial reports, etc.) and is designed to generalize across various domains.

---

## Approach

1. **Document Parsing**: Each PDF is processed using heuristics based on font size and layout to segment the text into titled sections. This is implemented in `src/docparse.py`.

2. **Relevance Analysis**: A relevance profile is built using the persona and job-to-be-done. Sections are ranked using semantic similarity (via Sentence-BERT embeddings). Extractive summarization is then performed on top-ranked sections to identify refined key points. This is implemented in `src/relevance.py`.

3. **API Interface**: A FastAPI server provides endpoints to upload files, configure persona/job prompts, and retrieve structured analysis. This is defined in `api.py`.

---

## Models and Libraries Used

- **Sentence-BERT** (`all-MiniLM-L6-v2`) via `sentence-transformers` for semantic similarity
- **PyMuPDF** (`pymupdf`) for PDF parsing
- **NLTK** for sentence tokenization
- **FastAPI** for web API interface
- **Uvicorn** for serving the API
- **Torch** for tensor computation (runs on CPU)

---

## Build & Run Instructions

```bash
uv sync
uv run download_model.py
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

The server will start at `http://0.0.0.0:8000`. You can access endpoints such as `/health`, `/analyze`, and the frontend UI served from the root (`/`).

---

## Directory Structure

```
.
├── api.py                   # Main FastAPI server
├── download_model.py        # (Optional) Script to pre-download model
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package configuration (if using uv/poetry)
├── templates/               # Jinja2 HTML templates
├── static/                  # Static assets
├── uploaded_files/          # Uploaded PDF input files
├── src/
│   ├── docparse.py          # PDF section extractor
│   └── relevance.py         # Semantic relevance scoring and summarization
```

---

