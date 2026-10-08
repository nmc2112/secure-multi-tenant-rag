# Secure Multi-Tenant RAG Assistant

**Status:** local AI engineering prototype; NOT production ready. Tested by project author on Intel Mac (October 2026). Built with Python, FastAPI, Qdrant local, Sentence Transformers, optional CrossEncoder reranking and Ollama Qwen2.5 3B. Supports structured JSON responses, source citations, evidence-aware fallback and tenant-scoped retrieval (demo tokens only).

## Architecture

`API auth -> tenant-scoped retrieval (lexical OR embeddings/Qdrant) -> optional CrossEncoder rerank -> evidence/context -> optional Ollama chat completions -> JSON/citation validation -> answer or fallback/review`

The `/ask` response includes `retrieval_mode`, `generation_mode` (`llm`, `fallback`, `not_attempted`) and model information when the LLM succeeds. This is **not** an autonomous agent and there is **no function calling** yet.

## Local baseline (no Ollama needed)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs. Use demonstration bearer tokens `demo-acme` and `demo-orbit` with synthetic, non-sensitive data only.

## Local semantic + Ollama (run outside Docker)

Install compatible `requirements-semantic.txt` dependencies for your OS. For an Intel Mac, the tested setup in this project's development was NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.2.2, Transformers 4.44.2, sentence-transformers 3.0.1. The optional semantic dependency file may need environment-specific pinning.

Start Ollama separately and pull `qwen2.5:3b`. In the API terminal:

```bash
export RETRIEVER=semantic RERANK=1
export LLM_BASE_URL=http://127.0.0.1:11434/v1
export LLM_API_KEY=ollama LLM_MODEL=qwen2.5:3b
python -m uvicorn app.main:app --reload
```

Qdrant local is single-process storage. Do not let the API and a benchmark access the same QDRANT_PATH concurrently. Benchmark example: `QDRANT_PATH=./.qdrant_eval_ext python evaluate_extended.py --mode semantic --top-k 3 --output reports/semantic.json`.

## Docker baseline only

The supplied Docker image deliberately runs lexical mode, not semantic+Ollama. Stop the existing host API on port 8000 first, then:

```bash
docker compose up --build -d
curl http://127.0.0.1:8000/health
docker compose logs --tail=50
docker compose down
```

Port is bound to loopback to reduce accidental external exposure. **Do not expose demo authentication on the public internet.**

## Measured local evaluations

| Evaluation | Result |
|---|---|
| 36 synthetic queries, Lexical Recall@3 / MRR@3 | 30.56% / 0.2361 |
| 36 synthetic queries, Semantic Recall@3 / MRR@3 | 94.44% / 0.8750 |
| Semantic mean retrieval latency | 107.77 ms |
| 18 synthetic answer queries, LLM successes | 16/18 (88.89%) |
| 18 synthetic answer queries, citation heuristic | 18/18 |
| 18 synthetic answer queries, fact heuristic | originally 17/18; 18/18 after accepting the answer `16` as equivalent to `16 characters` |
| LLM answer API mean latency | 3,216.15 ms |

See `reports/benchmark_summary.json` for provenance. These are **local, synthetic, limited-sample observations**, not generalization claims. No adversarial security audit or cloud production deployment was performed. Reranking was enabled in local tests but its independent lift was not benchmarked.

## Security and limitations

- Tenant identity comes from hardcoded demonstration bearer tokens, not verified OIDC/JWT.
- Local tenant-filtered retrieval and basic prompt-injection screening have been checked with simple cases, **not** a security guarantee.
- Human review is a flag, not a fully implemented review workflow.
- JSON schema and citation IDs are validated, but factual grounding is not formally verified.
- No agent tool execution or function calling. IoT integration is future work.
- Docker image is a reproducible lightweight **lexical baseline**; local semantic + Ollama requires a separate runtime.

## CI

GitHub Actions runs compile, pytest, a lexical mini-benchmark and a Docker image build. Semantic/LLM tests are not run in CI because they need model downloads, compute and a running Ollama service.

## Roadmap

1. More realistic, larger, annotated evaluation with negatives, French queries, p95 latency and model/reranker ablations.
2. JWT/OIDC, better multi-tenant isolation, prompt-injection red-team testing and audit logging.
3. Compose with Qdrant server, a separately hosted LLM and production-grade observability.
4. Constrained function calling and human approval workflow.
