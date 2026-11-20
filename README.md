# Flash Vector DB

A working **vector database** built from scratch in Python, with a web UI.

It implements **HNSW**, **KD-Tree**, and **Brute Force** side by side, three distance metrics, a live 2D PCA scatter plot, and an optional **RAG** pipeline through DeepInfra (OpenAI-compatible API).

---

## Features

| Feature | Description |
|---|---|
| **3 search algorithms** | HNSW, KD-Tree, Brute Force — compare latency live |
| **3 distance metrics** | Cosine distance, Euclidean, Manhattan |
| **16D demo vectors** | 20 preloaded items across CS, Math, Food, Sports |
| **2D PCA scatter plot** | Watch semantic clusters form in the browser |
| **Real embeddings** | Paste text → DeepInfra `BAAI/bge-base-en-v1.5` |
| **RAG pipeline** | Retrieve with HNSW → answer with DeepInfra chat model |
| **REST API** | insert, delete, search, benchmark, hnsw-info, docs |

---

## How it works

```
Your Text
    │
    ▼
DeepInfra embeddings          ← text → embedding vector
    │
    ▼
HNSW index (Python)           ← multilayer small-world graph
    │
    ▼
Semantic search               ← nearest neighbors in vector space
    │
    ▼
DeepInfra chat model          ← retrieved chunks → answer
    │
    ▼
Answer
```

Demo search (left panel) does not need DeepInfra. It uses the built-in 16D categorical vectors.

---

## Setup

### 1. Python deps

```powershell
cd flash_vector_db
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. DeepInfra keys (Documents + Ask AI tabs)

Copy `.env.example` to `.env` and fill in:

```
LLM_API_KEY=...
LLM_BASE_URL=https://api.deepinfra.com/v1/openai
LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
EMBED_MODEL=BAAI/bge-base-en-v1.5
```

`.env` is gitignored — never commit it.

---

## Run

```powershell
python server.py
```

Open http://localhost:8080

---

## Using the UI

### Search (demo vectors)

- Query concepts like `binary tree`, `sushi`, `basketball`
- Pick algorithm: HNSW / KD-Tree / Brute Force
- Pick metric: Cosine / Euclidean / Manhattan
- **COMPARE ALL ALGOS** times all three on the same query

### Documents

- Paste a title + text → chunked, embedded via DeepInfra, stored in HNSW

### Ask AI (RAG)

1. Insert documents first
2. Ask a question
3. Flow: embed question → HNSW top-k chunks → chat model answer

---

## REST API

### Demo vectors

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert demo vector |
| `DELETE` | `/delete/:id` | Delete by id |
| `GET` | `/items` | List demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare algorithms |
| `GET` | `/hnsw-info` | Graph layers / edges |
| `GET` | `/stats` | Counts and options |

### Documents + RAG

| Method | Endpoint | Body |
|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` |
| `GET` | `/doc/list` | — |
| `DELETE` | `/doc/delete/:id` | — |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` |
| `GET` | `/status` | LLM provider + counts |

---

## Project layout

```
flash_vector_db/
├── engine.py         ← distances, BruteForce, KD-Tree, HNSW, VectorDB, DocumentDB, LLMClient
├── server.py         ← Flask REST API + UI host
├── index.html        ← PCA scatter, search, docs, RAG chat
├── .env.example      ← template for DeepInfra keys
├── requirements.txt
└── README.md
```

---

## Algorithms (short)

**Brute Force** — score every vector. Exact. O(N·d). Baseline.

**KD-Tree** — binary space splits cycling through axes. Strong in low dimensions; weak at high-d.

**HNSW** — multilayer navigable small-world graph. Approximate, O(log N) in practice.

---

## License

MIT
