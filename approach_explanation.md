# Approach

## Methodology

This solution addresses the Persona-Driven Document Intelligence challenge by combining document parsing, semantic relevance analysis, and summarization into an end-to-end system. The pipeline is designed to be generic, allowing it to handle a wide variety of document types, personas, and tasks.

### Document Parsing

The system first processes each input PDF document using the PyMuPDF library. A heuristic-based parser segments the text into titled sections based on font size, layout, and formatting cues. The assumption is that section titles tend to use larger fonts and appear in shorter, more concise lines. This lightweight approach avoids the need for fine-tuned OCR or document layout models, enabling fast and CPU-efficient parsing.

### Relevance Profiling and Ranking

To identify sections relevant to a specific persona and their job-to-be-done, we construct a relevance profile from their description (e.g., “As a PhD student, I want to...”). This profile is encoded using a pretrained Sentence-BERT model (`all-MiniLM-L6-v2`), which provides a compact semantic representation.

Each section’s content is similarly embedded, and cosine similarity is computed between the profile and all sections. Sections are then ranked by their semantic closeness to the profile, giving us an importance-ranked list of the most relevant content.

### Subsection Analysis

To provide refined insights, the top-ranked sections are passed through an extractive summarization process. Sentences within each section are scored against the relevance profile, and the top-k most semantically similar sentences are selected to form a coherent, condensed summary. This provides more actionable information for the persona without overwhelming them with full-length sections.

### API Design

The system is wrapped in a FastAPI-based server, which supports file uploads, persona/job input via form or config files (JSON/YAML), and returns structured output in the required JSON format. The endpoints are designed for flexibility and testability, and a basic UI is available for interactive use.

## Models and Constraints

The core model (`all-MiniLM-L6-v2`) is under 100MB, CPU-compatible, and supports fast inference even with multiple documents. All processing occurs offline, and the entire pipeline can handle 3–5 documents in under 60 seconds on standard hardware, satisfying the competition constraints.

## Conclusion

This solution balances efficiency, generalization, and interpretability. It leverages pre-trained lightweight language models and heuristics instead of relying on large-scale document understanding systems. The modular architecture allows easy adaptation to new document types, personas, and tasks, making it a practical and scalable approach for real-world persona-driven document analysis.

