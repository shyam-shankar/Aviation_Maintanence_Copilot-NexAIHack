# Aviation Maintenance Copilot

An empirical diagnostic copilot for aircraft technicians, grounded strictly in 6,169 historical maintenance logbook records from the MaintNet / University of North Dakota Aviation Program (2012–2017).

> **SAFETY NOTICE:** This tool provides diagnostic reference based strictly on historical logbook entries. It does **not** provide FAA/EASA certified or manufacturer-approved maintenance procedures. Always consult the official Aircraft Maintenance Manual (AMM).

---

## Key Features (MVP)

1. **Aviation Text Normalizer ([`normalizer.py`](normalizer.py)):**
   - Merges MaintNet abbreviation tables with operational aircraft maintenance shorthand (`R&R` $\rightarrow$ `removed and replaced`, `A/C` $\rightarrow$ `aircraft`, `CYL` $\rightarrow$ `cylinder`, `L/H` $\rightarrow$ `left hand`, `ALT` $\rightarrow$ `alternator`).
   - Normalizes cylinder numbers (`#2` $\rightarrow$ `cylinder 2`) and preserves raw text for truthful citations.

2. **Hybrid Search Index ([`indexer.py`](indexer.py)):**
   - **BM25 Keyword Search** for exact part names, numbers, and identifiers.
   - **TF-IDF Semantic Similarity** (word + bigrams with sublinear TF scaling) for fault concepts and synonyms.
   - **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{TF-IDF}}(d)}$$
   - **Empirical Refusal Gate**: Refuses queries with low token coverage or low semantic similarity, preventing spurious guesses on out-of-domain queries.

3. **Maintenance Agent ([`agent.py`](agent.py)):**
   - Implements two primary tools:
     - `search_records(query, top_k)`
     - `get_record(record_id)`
   - Cites verifiable record numbers (`[IDENT: XXXXXX]`) for every claim.
   - Distinguishes **Likely Causes** from **Historical Clearing Actions**.
   - Refuses explicitly with **"Insufficient evidence"** when records don't support an answer.

4. **Simple & Focused Chat UI ([`app.py`](app.py)):**
   - **Left Panel**: Assessment, likely causes, clearing actions, and confidence score.
   - **Right Panel**: Side-by-side evidence inspection cards displaying verbatim raw logbook records.
   - **Tool Trace Expander**: Inspects agent tool calls, queries, candidate IDs, and RRF calculations.
   - **1-Click Test Queries**: Sample positive faults and negative refusal tests in the sidebar.

5. **Evaluation Suite ([`eval.py`](eval.py)):**
   - **Retrieval Hit@5**: 73.3% across diverse ground-truth logbook faults.
   - **Refusal Accuracy**: 100.0% on out-of-domain and impossible queries.
   - **Citation Precision**: 100.0% (all generated citations verified against historical records).

---

## Running the Copilot

The environment is already configured in `.venv`:

```bash
# Activate environment
source .venv/bin/activate

# Launch UI
streamlit run app.py --server.port 8501
```

Access the UI at: **`http://localhost:8501`**

To run automated benchmarks:
```bash
python eval.py
```
