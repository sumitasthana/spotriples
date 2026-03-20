# SPO Relationship Extractor

Extract **Subject–Predicate–Object (SPO) triplets** from unstructured text using a multi-pass LLM pipeline powered by OpenAI. Supports explicit relationships, cross-chunk derived relationships, negation detection, and optional implicit relationship inference.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI](#cli)
  - [FastAPI Backend](#fastapi-backend)
  - [React Frontend](#react-frontend)
  - [Streamlit App](#streamlit-app)
  - [Python API](#python-api)
- [Output Formats](#output-formats)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Overview

The SPO Relationship Extractor analyses text of any length and returns structured triplets of the form:

```
Apple  |  was founded by  |  Steve Jobs
Steve Jobs  |  served as  |  CEO of Apple
```

It handles long documents by splitting them into overlapping chunks and running a **three-pass extraction pipeline**:

| Pass | Description |
|------|-------------|
| **Pass 1** | Extract explicit SPO triplets from each chunk independently |
| **Pass 2** | Cross-chunk reasoning — derive new relationships by combining results from Pass 1 |
| **Pass 3** | *(optional)* Implicit relationship inference — surface strongly implied but unstated facts |

---

## Architecture

```
Text Input
    │
    ▼
chunk_text()          — split into overlapping chunks (default 800 tokens, 100 overlap)
    │
    ▼
Pass 1: Per-chunk extraction (parallel tqdm loop)
    │   → LLM pronoun resolution per chunk
    │   → Structured JSON via response_format=json_object
    │   → Few-shot examples in system prompt
    │
    ▼
Pass 2: Cross-chunk reasoning
    │   → tiktoken-budget-aware seed slice (fits within model context)
    │   → Derives new relationships not present in any single chunk
    │
    ▼
Pass 3 (optional): Implicit extraction
    │   → Finds strongly implied but literally unstated relationships
    │
    ▼
Deduplication (case-insensitive) + merge negated relationships
    │
    ▼
pandas DataFrame  →  CLI / API / UI
```

**Key design decisions:**
- **LLM-based pronoun resolution** — each chunk is pre-processed to replace pronouns with their referents before extraction (more accurate than spaCy rule-based coref)
- **Structured JSON output** — `response_format={"type": "json_object"}` eliminates regex fragility
- **Tiktoken budgeting** — Pass 2 seed slice uses actual token counts to avoid context overflow
- **No spaCy / langchain dependency** — text splitting is an inline implementation; pronoun resolution is fully LLM-driven

---

## Features

- Multi-pass SPO extraction from text of any length
- LLM-based pronoun resolution for accurate entity references
- Negation detection (`"Apple did NOT acquire Microsoft"` → `negated: true`)
- Cross-document relationship derivation (Pass 2)
- Optional implicit relationship extraction (Pass 3)
- Four output formats: table, CSV, JSON, triplets
- File input support: `.txt`, `.md`, `.pdf`
- REST API (FastAPI) with JSON and file upload endpoints
- React + Tailwind UI with sortable table, charts, and CSV/JSON export
- Legacy Streamlit UI
- Progress bars via `tqdm`

---

## Project Structure

```
SPO/
├── relationship_extractor.py   # Core multi-pass extraction engine
├── api.py                      # FastAPI REST backend
├── cli.py                      # Command-line interface
├── app.py                      # Legacy Streamlit web app
├── examples.py                 # Runnable usage examples
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
├── .gitignore
├── data/                       # Sample input files
│   └── 2025-11-04T03-14_export.csv
└── frontend/                   # React / Vite UI
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── index.css
```

---

## Prerequisites

- Python **3.10+** (tested on 3.13)
- Node.js **18+** (for the React frontend only)
- An **OpenAI API key**

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sumitasthana/spotriples.git
cd spotriples
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # optional, defaults to gpt-4o-mini
```

Supported models: `gpt-4o-mini` (default, fast/cheap), `gpt-4o`, `gpt-4-turbo`, or any OpenAI chat model.

---

## Usage

### CLI

The CLI reads from a text string or a file and writes results to stdout.

```bash
# Extract from a string
python cli.py --text "Apple was founded by Steve Jobs in 1976."

# Extract from a text or markdown file
python cli.py --file report.txt

# Extract from a PDF
python cli.py --file report.pdf

# Include implicit relationships (Pass 3)
python cli.py --file report.txt --implicit

# Choose output format
python cli.py --file report.txt --format table      # default
python cli.py --file report.txt --format csv
python cli.py --file report.txt --format json
python cli.py --file report.txt --format triplets

# Save output to a file
python cli.py --file report.txt --format csv --output results.csv

# Use a different model
python cli.py --text "..." --model gpt-4o

# Pass an API key directly (overrides .env)
python cli.py --text "..." --api-key sk-...
```

**CLI options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--text TEXT` | Raw text to extract from | — |
| `--file PATH` | Path to `.txt`, `.md`, or `.pdf` file | — |
| `--format` | Output format: `table`, `csv`, `json`, `triplets` | `table` |
| `--output PATH` | Write output to file instead of stdout | stdout |
| `--implicit` | Enable implicit relationship extraction (Pass 3) | off |
| `--model MODEL` | OpenAI model name | `gpt-4o-mini` |
| `--api-key KEY` | OpenAI API key (overrides `.env`) | from `.env` |

---

### FastAPI Backend

Start the backend server:

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

### React Frontend

With the backend running, start the React dev server:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

**UI features:**
- **Extract tab** — paste text or upload a file; toggle implicit extraction; view sortable/filterable results table
- **Analyze tab** — bar charts of top subjects, predicates, and objects
- **Guide tab** — built-in usage instructions
- Export results as CSV or JSON

---

### Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

### Python API

Use `RelationshipExtractor` directly in your own code:

```python
from relationship_extractor import RelationshipExtractor

extractor = RelationshipExtractor(api_key="sk-...", model="gpt-4o-mini")

df = extractor.extract(
    text="Elon Musk founded SpaceX in 2002. He also leads Tesla.",
    include_implicit=False,
)

print(df[["subject", "predicate", "object", "negated"]])
```

**Output (DataFrame):**

| subject | predicate | object | negated | source_quote |
|---------|-----------|--------|---------|--------------|
| Elon Musk | founded | SpaceX | False | Elon Musk founded SpaceX in 2002. |
| Elon Musk | leads | Tesla | False | He also leads Tesla. |

---

## Output Formats

### `table` (default)

```
+------------+--------------+--------+---------+
| subject    | predicate    | object | negated |
+------------+--------------+--------+---------+
| Apple      | was founded  | Steve Jobs | False |
+------------+--------------+--------+---------+
```

### `triplets`

```
Apple | was founded by | Steve Jobs
Steve Jobs | served as | CEO of Apple
```

### `csv`

```csv
"subject","predicate","object","negated","source_quote"
"Apple","was founded by","Steve Jobs","False","Apple was founded by Steve Jobs..."
```

### `json`

```json
[
  {
    "subject": "Apple",
    "predicate": "was founded by",
    "object": "Steve Jobs",
    "negated": false,
    "source_quote": "Apple was founded by Steve Jobs in 1976."
  }
]
```

---

## Examples

Run the included examples to see all features in action:

```bash
python examples.py
```

The file contains six progressive examples:
1. Basic extraction from a short paragraph
2. Multi-chunk extraction from a longer text
3. Negation detection
4. Implicit relationship extraction
5. Cross-chunk reasoning
6. DataFrame post-processing

---

## API Reference

### `GET /health`

Returns the server status and configured model.

```json
{ "status": "ok", "model": "gpt-4o-mini" }
```

### `POST /extract`

Extract relationships from a JSON body.

**Request:**
```json
{
  "text": "Apple was founded by Steve Jobs.",
  "include_implicit": false
}
```

**Response:**
```json
{
  "relationships": [...],
  "count": 1,
  "stats": {
    "unique_subjects": ["Apple"],
    "unique_predicates": ["was founded by"],
    "unique_objects": ["Steve Jobs"],
    "negated_count": 0
  }
}
```

### `POST /extract/file`

Extract relationships from a file upload (multipart/form-data).

**Form fields:**
- `file` — `.txt`, `.md`, or `.pdf` file
- `include_implicit` — `"true"` or `"false"` (optional, default `"false"`)

Response format is identical to `POST /extract`.
