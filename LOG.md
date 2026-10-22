# Project Progress Log

**Project:** Aviation Maintenance Copilot  
**Dataset:** MaintNet logbooks (6,169 records)  
**Branch:** `graph_test`  

---

## What is this?
An assistant to help aircraft technicians diagnose faults by searching historical maintenance logbooks. If there isn't enough proof in past records, it refuses to answer instead of guessing unverified maintenance procedures.

---

## How it works (Quick summary)
1. **Normalizer (`normalizer.py`):** Converts technician shorthand (`r&r`, `cyl #2`, `a/c`, `powerloss`) into plain terms.
2. **Search (`indexer.py`):** Combines BM25 for keyword matching + TF-IDF for fault descriptions, then merges the scores.
3. **Knowledge Graph (`graph_builder.py`):** Maps systems -> components -> symptoms -> clearing actions using NetworkX. Cached locally in `aviation_kg.json`.
4. **Agent (`agent.py`):** 
   - Uses local Gemma 4 (via Ollama) to clean up conversational queries into search terms.
   - Refuses queries if similarity or keyword coverage is below safety thresholds.
   - Summarizes past fixes with verifiable record IDs (`[IDENT: XXXXXX]`).
5. **UI (`app.py`):** Streamlit dashboard showing diagnostic suggestions, the visual graph diagram, and raw log entries side-by-side.

---

## Test Results (`eval.py`)
Tested against 15 diverse pilot discrepancy reports across 15 aircraft subsystems and 8 adversarial non-fleet queries:
- **Search accuracy (Hit@5):** 100.0% (15 out of 15 queries matched the target historical logbook fix; 12/15 at Rank 1)
- **Refusal accuracy:** 100.0% (8/8 adversarial queries for non-existent trainer systems like FADEC, APU, thrust reversers correctly refused)
- **Citation accuracy:** 100.0% (every citation links directly to a real historical record)

---

## Dev Log & Updates

### Oct 22, 2026
- Set up local Streamlit app and verified it runs cleanly on port 8501.
- Built automated evaluation suite in `eval.py` to benchmark hit rate and refusal accuracy.
- Set up GitHub branches: kept `main` untouched with original code, and created `graph_test` branch for the GraphRAG implementation.
- Fixed query refusal bug for `"engine choking and powerloss during departure"`:
  - Split compound words like `powerloss` -> `power loss`.
  - Added suffix handling so words like `choking` match `choke`.
  - Filtered prepositions (`during`, `when`, `while`) so they don't lower coverage scores.
  - Adjusted prompt so Gemma doesn't invent extra terms that dilute search scores.
  - Test passed and retrieved record `100004` (fouled spark plug fix).
- Made all dataset paths relative so the project runs on any machine without path errors.

---

## Next To-Do
- [ ] Add more components to the graph (avionics and hydraulics).
- [ ] Add an option in the UI to run 100% deterministic template responses (skipping LLM).
- [ ] Clean up and merge `graph_test` into `main` when ready.
