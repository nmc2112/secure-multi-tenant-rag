LLM ANSWER EVALUATION — SECURE RAG
===================================
This adds a standalone test script; it does NOT overwrite app/main.py or alter your existing project files.

1. Place evaluate_llm.py in ~/Projects/secure-rag-portfolio
2. Keep your FastAPI server running in semantic mode with Ollama settings.
3. In a new terminal:

cd ~/Projects/secure-rag-portfolio
source ~/.venv/bin/activate
python evaluate_llm.py --output reports/llm_evaluation.json

The script inserts 18 fictional policy documents into DEMO-ORBIT, not demo-acme, then sends 18 questions to /ask. It writes all responses and summary metrics to reports/llm_evaluation.json.
Expect several minutes on an Intel Mac CPU; per-request LLM timeout in current app/main.py is 18 seconds. The script prints progress after each answer.

Interpretation:
- generation_modes.llm: valid JSON returned by configured LLM and accepted by app (not proof of factual correctness).
- fallback: app used extractive fallback; often due to LLM timeout or invalid JSON/citations.
- fact_heuristic_pass_rate: basic keyword/number match, NOT comprehensive answer accuracy.
- expected_citation_rate: whether the expected document appears in citations.
- Do not present benchmark results as measured until run on your Mac.

To avoid any mutation of the demo-orbit tenant, restore your demo environment after evaluating or implement a dedicated evaluation tenant in a later upgrade.
