"""
HTTP API for the vector database. Serves the web UI and REST endpoints.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from engine import (
    DIMS,
    DocumentDB,
    LLMClient,
    VectorDB,
    chunk_text,
    get_dist_fn,
    load_demo,
)

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)

db = VectorDB(DIMS)
doc_db = DocumentDB()
llm = LLMClient()
load_demo(db)


def _parse_vec(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/", methods=["OPTIONS"])
@app.route("/<path:_any>", methods=["OPTIONS"])
def options(_any=None):
    return ("", 204)


@app.get("/search")
def search():
    q = _parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"
    out = db.search(q, k, metric, algo)
    return jsonify(
        {
            "results": [
                {
                    "id": h.id,
                    "metadata": h.meta,
                    "category": h.cat,
                    "distance": round(h.dist, 6),
                    "embedding": [round(x, 4) for x in h.emb],
                }
                for h in out.hits
            ],
            "latencyUs": out.us,
            "algo": out.algo,
            "metric": out.metric,
        }
    )


@app.post("/insert")
def insert():
    body = request.get_json(force=True, silent=True) or {}
    meta = body.get("metadata") or ""
    cat = body.get("category") or ""
    emb = body.get("embedding") or []
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    item_id = db.insert(meta, cat, list(map(float, emb)), get_dist_fn("cosine"))
    return jsonify({"id": item_id})


@app.delete("/delete/<int:item_id>")
def delete(item_id: int):
    return jsonify({"ok": db.remove(item_id)})


@app.get("/items")
def items():
    return jsonify(
        [
            {
                "id": v.id,
                "metadata": v.metadata,
                "category": v.category,
                "embedding": [round(x, 4) for x in v.emb],
            }
            for v in db.all()
        ]
    )


@app.get("/benchmark")
def benchmark():
    q = _parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric") or "cosine"
    b = db.benchmark(q, k, metric)
    return jsonify(
        {
            "bruteforceUs": b.bf_us,
            "kdtreeUs": b.kd_us,
            "hnswUs": b.hnsw_us,
            "itemCount": b.n,
        }
    )


@app.get("/hnsw-info")
def hnsw_info():
    gi = db.hnsw_info()
    return jsonify(
        {
            "topLayer": gi.top_layer,
            "nodeCount": gi.node_count,
            "nodesPerLayer": gi.nodes_per_layer,
            "edgesPerLayer": gi.edges_per_layer,
            "nodes": gi.nodes,
            "edges": gi.edges,
        }
    )


@app.post("/doc/insert")
def doc_insert():
    body = request.get_json(force=True, silent=True) or {}
    title = body.get("title") or ""
    text = body.get("text") or ""
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids: list[int] = []
    for i, chunk in enumerate(chunks):
        emb = llm.embed(chunk)
        if not emb:
            return jsonify(
                {
                    "error": (
                        "DeepInfra unavailable. Set LLM_API_KEY and LLM_BASE_URL in .env "
                        "(see .env.example)."
                    )
                }
            ), 503
        chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.delete("/doc/delete/<int:item_id>")
def doc_delete(item_id: int):
    return jsonify({"ok": doc_db.remove(item_id)})


@app.get("/doc/list")
def doc_list():
    docs = []
    for d in doc_db.all():
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        words = len(d.text.split())
        docs.append({"id": d.id, "title": d.title, "preview": preview, "words": words})
    return jsonify(docs)


@app.post("/doc/search")
def doc_search():
    body = request.get_json(force=True, silent=True) or {}
    question = body.get("question") or ""
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb = llm.embed(question)
    if not q_emb:
        return jsonify({"error": "DeepInfra unavailable"}), 503
    hits = doc_db.search(q_emb, k)
    return jsonify(
        {
            "contexts": [
                {
                    "id": item.id,
                    "title": item.title,
                    "distance": round(dist, 4),
                }
                for dist, item in hits
            ]
        }
    )


@app.post("/doc/ask")
def doc_ask():
    body = request.get_json(force=True, silent=True) or {}
    question = body.get("question") or ""
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    q_emb = llm.embed(question)
    if not q_emb:
        return jsonify({"error": "DeepInfra unavailable"}), 503

    hits = doc_db.search(q_emb, k)
    ctx_parts = []
    for i, (_, item) in enumerate(hits, start=1):
        ctx_parts.append(f"[{i}] {item.title}:\n{item.text}\n")
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{''.join(ctx_parts)}"
        f"Question: {question}\n\n"
        "Answer:"
    )
    answer = llm.generate(prompt)
    return jsonify(
        {
            "answer": answer,
            "model": llm.gen_model,
            "contexts": [
                {
                    "id": item.id,
                    "title": item.title,
                    "text": item.text,
                    "distance": round(dist, 4),
                }
                for dist, item in hits
            ],
            "docCount": doc_db.size(),
        }
    )


@app.get("/status")
def status():
    available = llm.is_available()
    return jsonify(
        {
            "ollamaAvailable": available,  # UI field name (legacy)
            "llmAvailable": available,
            "provider": "deepinfra",
            "embedModel": llm.embed_model,
            "genModel": llm.gen_model,
            "docCount": doc_db.size(),
            "docDims": doc_db.get_dims(),
            "demoDims": DIMS,
            "demoCount": db.size(),
        }
    )


@app.get("/stats")
def stats():
    return jsonify(
        {
            "count": db.size(),
            "dims": DIMS,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics": ["euclidean", "cosine", "manhattan"],
        }
    )


def main() -> None:
    up = llm.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(
        f"DeepInfra: {'ONLINE' if up else 'OFFLINE (check .env LLM_API_KEY / LLM_BASE_URL)'}"
    )
    if up:
        print(f"  embed model: {llm.embed_model}  gen model: {llm.gen_model}")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)


if __name__ == "__main__":
    main()
