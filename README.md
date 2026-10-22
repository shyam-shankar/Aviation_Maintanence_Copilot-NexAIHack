# AeroGraph Copilot // Aviation MRO Flight Log Diagnostic Engine

> **Autonomous Knowledge-Graph & Hybrid RAG Diagnostic Engine Grounded in 6,169 UND Aviation Fleet Maintenance Records (2012–2017)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit%201.35+-red.svg)](https://streamlit.io/)
[![Retrieval Hit@5](https://img.shields.io/badge/Retrieval%20Hit@5-100.0%25-brightgreen.svg)]()
[![Refusal Accuracy](https://img.shields.io/badge/Refusal%20Accuracy-100.0%25-brightgreen.svg)]()
[![Citation Precision](https://img.shields.io/badge/Citation%20Precision-100.0%25-brightgreen.svg)]()
[![Local LLM](https://img.shields.io/badge/Local%20LLM-Gemma%204:E2B%20(Ollama)-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> [!WARNING]
> **MANDATORY FLIGHT SAFETY & AIRWORTHINESS ADVISORY**  
> This copilot provides diagnostic reference based strictly on historical logbook entries from the MaintNet / University of North Dakota (UND) Aviation Program dataset. It does **not** provide FAA/EASA certified or manufacturer-approved maintenance procedures. Aviation maintenance technicians and certificated A&P mechanics must **always consult the official Aircraft Maintenance Manual (AMM), Illustrated Parts Catalog (IPC), and applicable Airworthiness Directives (ADs)** before performing maintenance or signing off an aircraft return-to-service.

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. The Problem Space & The Aviation Challenge](#2-the-problem-space--the-aviation-challenge)
  - [The Hazards of Naive LLMs and Standard Vector RAG](#the-hazards-of-naive-llms-and-standard-vector-rag)
- [3. End-to-End System Architecture](#3-end-to-end-system-architecture)
  - [High-Level Architecture Pipeline](#high-level-architecture-pipeline)
  - [Detailed Dataflow & Tool Execution Pipeline](#detailed-dataflow--tool-execution-pipeline)
- [4. Deep-Dive: Architectural Decisions & Technical Rationale](#4-deep-dive-architectural-decisions--technical-rationale)
  - [Decision 1: Domain-Specific Normalization Before Indexing & Querying](#decision-1-domain-specific-normalization-before-indexing--querying)
  - [Decision 2: Hybrid BM25 + Sublinear TF-IDF with Reciprocal Rank Fusion (RRF)](#decision-2-hybrid-bm25--sublinear-tf-idf-with-reciprocal-rank-fusion-rrf)
  - [Decision 3: Multi-Gate Empirical Refusal Safeguard](#decision-3-multi-gate-empirical-refusal-safeguard)
  - [Decision 4: Causal Knowledge Graph (GraphRAG) with NetworkX](#decision-4-causal-knowledge-graph-graphrag-with-networkx)
  - [Decision 5: Dual Synthesis Modes (Local Gemma 4 + Deterministic Template Fallback)](#decision-5-dual-synthesis-modes-local-gemma-4--deterministic-template-fallback)
  - [Decision 6: Glass Cockpit UI with Side-by-Side Verbatim Evidence Cards](#decision-6-glass-cockpit-ui-with-side-by-side-verbatim-evidence-cards)
  - [Decision 7: Fleet Chronic Defect Analytics (Recurring Defect Detector)](#decision-7-fleet-chronic-defect-analytics-recurring-defect-detector)
- [5. Knowledge Graph Schema & Ontology](#5-knowledge-graph-schema--ontology)
- [6. Agentic Tool Specifications](#6-agentic-tool-specifications)
- [7. Benchmark Results & Quantitative Evaluation](#7-benchmark-results--quantitative-evaluation)
  - [Retrieval Accuracy (Hit@5)](#retrieval-accuracy-hit5)
  - [Adversarial Near-Miss Refusal Accuracy](#adversarial-near-miss-refusal-accuracy)
  - [Citation Precision](#citation-precision)
- [8. Repository Structure](#8-repository-structure)
- [9. Installation & Quickstart](#9-installation--quickstart)
  - [Quick Launch](#quick-launch)
  - [Running Automated Benchmarks](#running-automated-benchmarks)
  - [Configuring Local Gemma 4 (Ollama)](#configuring-local-gemma-4-ollama)
- [10. Maintenance & Future Roadmap](#10-maintenance--future-roadmap)

---

## 1. Executive Summary

**AeroGraph Copilot** is an empirical diagnostic copilot for aircraft Maintenance, Repair, and Overhaul (MRO) technicians. Operating across **6,169 verified historical logbook records** from the University of North Dakota (UND) flight training fleet (2012–2017), the system enables technicians to input conversational defect squelches or pilot reports (e.g., *"the bird's left fan is coughing when I push the throttle"* or *"HAD ENGINE CHOKE & BRIEFLY LOSE POWER ON DEPARTURE"*), maps them onto a structured causal graph of aircraft systems and failure modes, and retrieves verifiable historical clearing actions with **100% auditable record citations (`[IDENT: XXXXXX]`)**.

Unlike generic RAG implementations that hallucinate plausible answers on out-of-domain or unseen queries, AeroGraph incorporates a **mathematical multi-gate refusal engine**: if the historical records lack sufficient token coverage or semantic precedent, the system strictly refuses to answer, protecting technicians from ungrounded advice.

---

## 2. The Problem Space & The Aviation Challenge

Aircraft maintenance records are notoriously difficult for conventional Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) systems:

1. **Telegraphic, Unstandardized Shorthand:** Discrepancy logs are composed under time pressure in hangars or on flight lines using idiosyncratic abbreviations (e.g., `R&R` for "removed and replaced", `CYL #2` for "cylinder 2", `A/C` for "aircraft", `INOP` for "inoperative", `VAC` for "vacuum", `ALT` for "alternator").
2. **High Domain Specificity:** Standard off-the-shelf tokenizers and lemmatizers (such as NLTK WordNet or spaCy) routinely mangle aviation terms (e.g., parsing `R&R` as punctuation, or treating `A/C` as air conditioning / alternating current).
3. **Compound Term Concatenation:** Pilot squelches frequently concatenate terms (e.g., `powerloss`, `stopdrilled`, `runup`), causing standard keyword indices to miss matches.
4. **Morphological Variations:** Technicians interchange verb tenses and participles (e.g., `choking` vs. `choke`, `leaking` vs. `leaks`, `hesitating` vs. `hesitation`, `vibrating` vs. `vibration`).

### The Hazards of Naive LLMs and Standard Vector RAG

In consumer applications, an AI hallucination is an inconvenience. **In aviation maintenance, an AI hallucination is a catastrophic airworthiness hazard.**

- **Hallucination Under Uncertainty:** Off-the-shelf generative LLMs (GPT-4, Claude, Llama) are trained to be "helpful" and will readily interpolate unverified torque specifications, plausible-sounding clearing procedures, or non-existent component part numbers.
- **The Vector Drift Dilemma in Dense Retrieval:** Modern embedding models (e.g., Sentence-BERT, OpenAI text-embedding-3, Nomic) project text into continuous dense vector spaces. In specialized technical domains, dense embeddings frequently suffer from *semantic drift*: an adversarial or non-existent fault (e.g., *"Magneto internal plastic distributor gear teeth stripped and seized"*) gets mapped closely to generic magneto records (e.g., *"Magneto RPM drop on runup"*), leading naive dense vector search to return irrelevant records with misleadingly high cosine similarity (>0.75).
- **The Opaque Black Box:** Standard RAG pipelines pass retrieved chunks directly to an LLM without giving the technician visibility into token coverage, graph relationships, or verifiable logbook IDs.

---

## 3. End-to-End System Architecture

AeroGraph resolves these challenges through a hybrid architecture combining rule-based normalization, dual-channel hybrid retrieval, a mathematical refusal safeguard, causal knowledge-graph traversal, and dual synthesis options.

### High-Level Architecture Pipeline

```mermaid
flowchart TD
    UserQuery["Technician Defect Report / Squelch<br><i>'Left fan is coughing on throttle push'</i>"]
    
    subgraph Step0 ["Phase 0: Query Preprocessing & Rewriting"]
        GemmaRewriter["Gemma 4:E2B Query Rewriter (Ollama)<br><i>Extracts root symptoms without speculation</i>"]
        Normalizer["Aviation Text Normalizer<br><i>Expands shorthand, compounds & cylinder notations</i>"]
    end
    
    subgraph Step1 ["Phase 1: Dual-Channel Hybrid Retrieval"]
        BM25["BM25Okapi Index<br><i>Exact technical keywords & part names</i>"]
        TFIDF["Sublinear TF-IDF Vectorizer<br><i>N-gram (1,2) phrasal semantic similarity</i>"]
        RRF["Reciprocal Rank Fusion (RRF, k=60)<br><i>Score-invariant rank aggregation</i>"]
    end
    
    subgraph Step2 ["Phase 2: Empirical Refusal Gate (Safety Guardrail)"]
        CoverageCheck{"Coverage >= 0.32 &<br>(Confidence >= 0.35 | CosSim >= 0.52)?"}
        RefusalOutput["Zero-Hallucination Refusal<br><i>Explicitly refuses & directs to AMM</i>"]
    end
    
    subgraph Step3 ["Phase 3: Knowledge Graph Traversal (GraphRAG)"]
        AnchorNodes["Anchor Query to Entity Nodes<br><i>(System, Component, Symptom)</i>"]
        GraphTraversal["Causal Subgraph Traversal<br><i>System -> Component -> Symptom -> Action</i>"]
        RecordIntersection["Edge Record ID Intersection<br><i>Extracts 100% causal logbook IDs</i>"]
    end
    
    subgraph Step4 ["Phase 4: Synthesis & Glass Cockpit UI"]
        DualSynth{"Local Ollama<br>Online?"}
        LLMSynth["Gemma 4 Grounded LLM Report<br><i>Verifiable citations [IDENT: XXXXXX]</i>"]
        DetSynth["Deterministic Template Report<br><i>100% offline airworthiness mode</i>"]
        GlassCockpit["Streamlit Glass Cockpit HUD<br><i>Diagnosis | Visual DOT Graph | Verbatim Evidence</i>"]
    end

    UserQuery --> GemmaRewriter
    GemmaRewriter --> Normalizer
    Normalizer --> BM25
    Normalizer --> TFIDF
    BM25 --> RRF
    TFIDF --> RRF
    RRF --> CoverageCheck
    
    CoverageCheck -- "No (Insufficient Evidence)" --> RefusalOutput
    CoverageCheck -- "Yes (Sufficient Evidence)" --> AnchorNodes
    
    AnchorNodes --> GraphTraversal
    GraphTraversal --> RecordIntersection
    RecordIntersection --> DualSynth
    
    DualSynth -- "Yes" --> LLMSynth
    DualSynth -- "No / Offline" --> DetSynth
    
    LLMSynth --> GlassCockpit
    DetSynth --> GlassCockpit
    RefusalOutput --> GlassCockpit
```

---

### Detailed Dataflow & Tool Execution Pipeline

The `MaintenanceCopilotAgent` executes a sequential 5-step tool-orchestrated loop:

```mermaid
sequenceDiagram
    autonumber
    actor Tech as Aircraft Technician
    participant Agent as MaintenanceCopilotAgent
    participant Rewriter as Gemma 4 (Ollama)
    participant Norm as AviationNormalizer
    participant Index as HybridIndex (BM25 + TF-IDF)
    participant Gate as Refusal Gate
    participant KG as AviationKnowledgeGraph
    participant UI as Glass Cockpit UI

    Tech->>UI: Enter discrepancy query
    UI->>Agent: answer_fault(query, enable_llm)
    
    opt LLM Enabled & Ollama Online
        Agent->>Rewriter: Tool 0: rewrite_query(raw_query)
        Rewriter-->>Agent: Concise maintenance search terms
    end
    
    Agent->>Norm: normalize(query)
    Norm-->>Agent: Normalized token stream
    
    Agent->>Index: Tool 1: search_records(query, top_k=5)
    Index->>Index: Compute BM25 + TF-IDF Cosine Sim + RRF
    Index-->>Agent: Candidates + Coverage & Confidence Metrics
    
    Agent->>Gate: Evaluate refusal criteria
    alt Insufficient Evidence (Coverage < 0.32 or Confidence < 0.35)
        Gate-->>Agent: is_confident = False
        Agent-->>UI: Return Refusal Verdict + Recommend AMM
    else Sufficient Evidence
        Gate-->>Agent: is_confident = True
        Agent->>KG: Tool 3: query_graph(query)
        KG->>KG: Anchor (System, Component, Symptom) & Traverse Causal Edges
        KG-->>Agent: Causal Path + Intersected Record IDs + DOT Digraph
        
        loop For Top-3 Verified Candidates
            Agent->>Index: Tool 2: get_record(record_id)
            Index-->>Agent: Raw problem & action text
        end
        
        alt Gemma 4 Online
            Agent->>Rewriter: synthesize_with_llm(records, graph_res)
            Rewriter-->>Agent: Grounded structured report
        else Deterministic Mode
            Agent->>Agent: Compile deterministic diagnostic template
        end
        
        Agent-->>UI: Full Diagnostic Assessment + Graphviz Digraph + Evidence Cards
    end
```

---

## 4. Deep-Dive: Architectural Decisions & Technical Rationale

Every component in AeroGraph was selected based on strict aviation safety criteria, deterministic auditability, and empirical performance benchmarks.

### Decision 1: Domain-Specific Normalization Before Indexing & Querying

* **Implementation:** [`normalizer.py`](normalizer.py)
* **What it does:**
  - Loads the MaintNet aviation abbreviation lexicon (`Aviation_Abbreviation_Dataset.csv`) merged with operational MRO shorthand.
  - Expands compound and slashed shorthand: `R&R` / `R/R` $\rightarrow$ `removed and replaced`, `A/C` $\rightarrow$ `aircraft`, `L/H` $\rightarrow$ `left hand`, `R/H` $\rightarrow$ `right hand`, `P/N` $\rightarrow$ `part number`, `S/N` $\rightarrow$ `serial number`, `W/O` $\rightarrow$ `without`.
  - Normalizes cylinder designations: `#2` $\rightarrow$ `cylinder 2`.
  - Resolves concatenated compounds and morphological variations: `powerloss` $\rightarrow$ `power loss`, `choking` $\rightarrow$ `choke`, `loss of power` $\rightarrow$ `lose power`, `shaking` / `vibrating` $\rightarrow$ `vibration`, `stalling` $\rightarrow$ `stall`, `overheating` $\rightarrow$ `overheat`.
  - Strips stop words and prepositions (`during`, `when`, `while`, `after`, `before`) that dilute keyword density.
* **Why this decision was made:**
  - Standard lemmatizers (e.g. WordNet, spaCy, Stanza) are trained on news and web text. They fail completely on aviation logs: `R&R` is split into punctuation tokens, `#2` is discarded, and `A/C` is misclassified as "air conditioning".
  - If abbreviations were not normalized before indexing, a search for *"cylinder 2 rocker cover removed and replaced"* would completely miss historical logbook entry `#2 CYL ROCKER COVER GASKET R&R`.
  - Crucially, normalization produces clean tokens for search while **preserving the original verbatim text for truthful user citations**.

---

### Decision 2: Hybrid BM25 + Sublinear TF-IDF with Reciprocal Rank Fusion (RRF)

* **Implementation:** [`indexer.py`](indexer.py)
* **What it does:**
  - **BM25Okapi Channel:** Operates on normalized query tokens to find exact technical matches (part names, component designations, assembly numbers).
  - **TF-IDF Vector Channel:** Operates with word + bigram features (`ngram_range=(1, 2)`) and sublinear term frequency scaling (`sublinear_tf=True`) to capture phrasal co-occurrence (e.g., `power loss`, `rocker cover`, `oil pressure`, `starter ring gear`).
  - **Reciprocal Rank Fusion (RRF):**
    $$\text{RRF}(d) = \sum_{m \in \{\text{BM25}, \text{TF-IDF}\}} \frac{\mathbb{I}(\text{score}_m(d) > \tau_m)}{k + \text{rank}_m(d)}$$
    where $k = 60$, $\tau_{\text{BM25}} = 0$, and $\tau_{\text{TF-IDF}} = 0.05$. Candidate pools from the top 100 results of each channel are merged.
* **Why this decision was made (over Dense Neural Embeddings):**
  - **The Peril of Dense Semantic Drift:** In early prototypes, dense neural embeddings (e.g. Sentence-BERT, Nomic, FAISS) were tested. While effective on conversational queries, dense embeddings exhibited severe failure modes on specialized technical components:
    - *Example:* An adversarial test query *"Magneto internal plastic distributor gear teeth stripped and seized"* was retrieved by dense embeddings with high similarity to *"Magneto RPM drop on runup"* simply because both texts shared dense vector representations of "magneto" and "engine".
    - Dense vectors cannot distinguish between a real failure mode present in the fleet and a plausible-sounding near-miss that never occurred.
  - **Explainability & Auditing:** BM25 and TF-IDF provide exact token-level matched features. Technicians can see exactly which words triggered the retrieval (`matched_tokens`).
  - **Rank-Invariant Score Fusion:** BM25 produces unbounded positive log-odds scores ($0$ to $30+$), whereas TF-IDF Cosine Similarity is bounded ($0.0$ to $1.0$). Normalizing these scores linearly is unstable across varied query lengths. RRF eliminates scale variance by operating purely on ordinal ranks.
  - **Sub-15ms In-Memory Execution:** The entire 6,169-record corpus is indexed in memory with zero GPU requirement, enabling deployment in remote hangars or on ruggedized tablets.

---

### Decision 3: Multi-Gate Empirical Refusal Safeguard

* **Implementation:** [`indexer.py`](indexer.py) & [`agent.py`](agent.py)
* **What it does:**
  - Evaluates the top candidate against calibrated mathematical safety gates:
    $$\text{Coverage}(q, d) = \frac{|\text{Matched Tokens}(q, d)|}{|q_{\text{tokens}}|}$$
    $$\text{Confidence}(q, d) = 0.5 \times \text{CosineSim}(q, d) + 0.5 \times \text{Coverage}(q, d)$$
  - **Refusal Condition:** A query is accepted **only** if it satisfies the anchor coverage requirement and at least one confidence criteria:
    $$\text{is\_confident} = (\text{Coverage} \ge 0.32) \land \Big[(\text{Confidence} \ge 0.35) \lor (\text{Coverage} \ge 0.45 \land \text{BM25} > 2.0) \lor (\text{CosineSim} \ge 0.52)\Big]$$
  - If `is_confident == False`, the copilot **refuses to synthesize an answer**, halts LLM execution, and returns:
    > *"Refusal: Insufficient Evidence in Logbooks. The historical maintenance logbooks do not contain sufficiently verified precedents matching this fault description... Required Action: Consult the official Aircraft Maintenance Manual (AMM)."*
* **Why this decision was made:**
  - Standard RAG systems blindly answer every prompt, hallucinating advice when presented with out-of-domain queries or unverified failure modes.
  - In aviation, **saying "I don't know / Insufficient evidence" is a life-safety requirement.**
  - The coverage threshold ($\ge 0.32$) acts as an anchor gate: even if an embedding model or TF-IDF reports moderate similarity due to common stop-words or general terms, a candidate matching fewer than a third of the query's informative tokens is rejected.
  - The refusal gate incorporates **suffix-aware morphological root matching** (`ing`, `ed`, `es`, `s`), ensuring that legitimate pilot queries (e.g., *"engine choking on departure"*) match root forms (`choke`) without triggering false refusals.

---

### Decision 4: Causal Knowledge Graph (GraphRAG) with NetworkX

* **Implementation:** [`graph_builder.py`](graph_builder.py)
* **What it does:**
  - Parses all 6,169 records into a directed ontology graph (`nx.DiGraph`), serialized to [`aviation_kg.json`](aviation_kg.json).
  - Implements a 4-tier causal hierarchy:
    $$\text{System} \xrightarrow{\text{HAS\_PART}} \text{Component} \xrightarrow{\text{EXHIBITED}} \text{Symptom} \xrightarrow{\text{RESOLVED\_BY}} \text{Action}$$
  - **Edge Properties:**
    - `weight`: Occurrence frequency across fleet logs.
    - `record_ids`: The exact logbook `IDENT` numbers where this specific relationship occurred.
  - **GraphRAG Traversal:**
    1. Anchors query to canonical System, Component, and Symptom nodes via regex entity resolution.
    2. Refuses queries with unknown components or disconnected graph topology.
    3. Traverses the graph from the anchored component through symptoms to historical clearing actions.
    4. Computes the **set intersection of record IDs** across edges:
       $$\text{Causal Records} = \text{record\_ids}(\text{Component} \to \text{Symptom}) \cap \text{record\_ids}(\text{Symptom} \to \text{Action})$$
  - Generates Graphviz DOT markup for real-time visualization in the Streamlit UI.
* **Why this decision was made:**
  - Text retrieval alone retrieves isolated strings without causal understanding. It cannot answer questions like: *"What is the most frequent clearing action for this symptom across the fleet?"*
  - The graph enforces **causal structural grounding**: actions are explicitly tied to the symptoms they resolved, backed by historical counts.
  - NetworkX provides in-memory traversal in microseconds (<1ms) without the infrastructure overhead or latency of external graph databases (like Neo4j).
  - The JSON cache ([`aviation_kg.json`](aviation_kg.json)) ensures instant startup times without re-parsing the dataset on every application boot.

---

### Decision 5: Dual Synthesis Modes (Local Gemma 4 + Deterministic Template Fallback)

* **Implementation:** [`agent.py`](agent.py)
* **What it does:**
  - **Query Rewriter (Tool 0):** Uses local Gemma 4 (`gemma4:e2b` via Ollama) to translate conversational pilot slang into concise logbook search terms under strict negative constraints:
    - *Input:* `"the bird's left fan is coughing when I push the throttle"`
    - *Output:* `"left engine choke throttle power loss"`
    - *Rule:* The prompt forbids speculative extrapolation (e.g. no adding "fuel system", "electrical failure", etc.).
  - **Synthesis Engine (Tool 5):**
    - **LLM Mode (Local Gemma 4):** Prompts the model with only the verified cited records, requiring it to format findings into Observed Causes, Historical Clearing Actions, and Primary Recommendation while citing `[IDENT: XXXXXX]`.
    - **Deterministic Mode (Template Fallback):** If Ollama is offline or disabled via the UI toggle, the agent uses a deterministic template that synthesizes the diagnostic report directly from the graph traversal and top records.
* **Why this decision was made:**
  - **Zero Cloud Dependency & Air-Gapped Operation:** Flight line hangars and military MRO depots frequently operate without internet connectivity. AeroGraph can run 100% locally with zero external API calls.
  - **Deterministic Safety Guarantee:** If the local LLM server is unavailable or runs out of VRAM, the system degrades gracefully into a deterministic template engine with zero loss of retrieval accuracy or citation precision.
  - **Strict Citation Grounding:** By feeding only retrieved records into the prompt with temperature $0.2$, Gemma 4 acts solely as a summarizer rather than a knowledge source.

---

### Decision 6: Glass Cockpit UI with Side-by-Side Verbatim Evidence Cards

* **Implementation:** [`app.py`](app.py)
* **What it does:**
  - Designed with an aviation "Glass Cockpit" dark HUD theme (`#080d1a`, cyan telemetry tags, alert badges).
  - Main diagnostic screen split into two synchronized columns:
    - **Left Column:** Synthesized diagnostic assessment, anchored system/component badges, and the **interactive Graphviz causal subgraph**.
    - **Right Column:** **Verbatim Evidence Cards** showing the exact raw problem and action text from historical logbooks with their `IDENT` numbers and match percentages.
  - **Tool-Call Trace Expander:** Provides a step-by-step audit log of every tool invoked by the agent (`rewrite_query`, `search_records`, `query_graph`, `get_record`, `synthesize_evidence`).
* **Why this decision was made:**
  - **Human-in-the-Loop Verification:** Aviation regulations require certificated mechanics to sign off maintenance. A mechanic must never be forced to trust an AI summary blindly. Showing raw logbook entries side-by-side allows instant verification against real historical precedent.
  - **Explainability Compliance:** The visual knowledge graph and tool execution trace demystify the agent's internal reasoning, satisfying strict aviation auditing standards.

---

### Decision 7: Fleet Chronic Defect Analytics (Recurring Defect Detector)

* **Implementation:** [`graph_builder.py`](graph_builder.py) & [`app.py`](app.py) (Tab 2)
* **What it does:**
  - Extracts edge weights on `(Component) --[EXHIBITED]--> (Symptom)` relationships across all 6,169 records.
  - Ranks fleet components by recurring defect frequency and presents an interactive analytics dashboard with a bar chart and sample citations.
* **Top Recurring Defects Identified in the UND Fleet:**
  1. **Rocker Cover & Gaskets $\to$ Fluid / Pressure Leak** (471 incidents)
  2. **Engine Baffles & Air Seals $\to$ Structural Crack** (246 incidents)
  3. **Magnetos & Ignition System $\to$ Rough Running & Vibration / RPM Drop** (164 incidents)
  4. **Alternator & Drive Belt $\to$ Loose Hardware / Slipping Belt** (112 incidents)
* **Why this decision was made:**
  - Moves the copilot from reactive troubleshooting to **proactive fleet reliability engineering**.
  - Maintenance directors can quickly spot recurring component failure trends across airframe types.

---

## 5. Knowledge Graph Schema & Ontology

The AeroGraph Knowledge Graph adheres to a strict 4-tier ontological model:

```mermaid
classDiagram
    direction LR
    class System {
        +String name
        +String type = "System"
    }
    class Component {
        +String name
        +String type = "Component"
        +String system
    }
    class Symptom {
        +String name
        +String type = "Symptom"
    }
    class Action {
        +String name
        +String type = "Action"
    }

    System --> Component : HAS_PART
    Component --> Symptom : EXHIBITED [weight, record_ids]
    Symptom --> Action : RESOLVED_BY [weight, record_ids]
```

### Ontology Specifications

| Entity Type | Description | Sample Entities |
| :--- | :--- | :--- |
| **System** | High-level aircraft functional system | Powerplant & Engine, Electrical & Ignition, Fuel System, Landing Gear & Controls, Airframe & Structure, Exhaust System, Pneumatics & Instruments |
| **Component** | Physical replaceable part, assembly, or line item | Rocker Cover & Gaskets, Magnetos & Ignition System, Fuel Servo & Metering Unit, Starter Ring Gear, Alternator & Drive Belt, Engine Baffles |
| **Symptom** | Observed discrepancy, defect mode, or pilot report | Fluid / Pressure Leak, Structural Crack, Rough Running & Vibration, Engine Choke & Power Loss, Loose Hardware / Fastener, Inoperative |
| **Action** | Corrective maintenance action taken by technician | Removed & Replaced Component, Tightened & Torqued Hardware, Adjusted & Calibrated Setting, Cleaned & Serviced Assembly, Stop-Drilled Crack |

### Graph Topology Statistics

- **Total Graph Nodes:** 44
- **Causal Edges:** 128
- **Component Classes Indexed:** 19
- **Symptom Patterns Indexed:** 14
- **Action Classes Indexed:** 8
- **Underlying Grounded Fleet Records:** 6,169

---

## 6. Agentic Tool Specifications

The copilot agent operates through five modular, auditable tools:

| Tool Name | Input Parameters | Return Value | Functional Purpose |
| :--- | :--- | :--- | :--- |
| `rewrite_query` | `raw_query: str` | `Tuple[str, bool, str]` | Uses local Gemma 4 to convert conversational pilot slang into concise logbook keywords under strict negative anti-hallucination constraints. |
| `search_records` | `query: str, top_k: int = 5` | `Dict[str, Any]` | Executes hybrid BM25 + TF-IDF retrieval with RRF rank fusion and evaluates the multi-gate empirical refusal criteria. |
| `get_record` | `record_id: str` | `Dict[str, Any]` | Fetches the verbatim raw problem description and technician clearing action by unique record `IDENT`. |
| `query_graph` | `query: str` | `Dict[str, Any]` | Anchors query to graph entities, traverses causal paths (`System` $\to$ `Component` $\to$ `Symptom` $\to$ `Action`), and intersects edge `record_ids`. |
| `synthesize_with_llm`| `fault_query, cited_records, graph_res` | `Optional[str]` | Synthesizes a grounded report with local Gemma 4 citing `[IDENT: XXXXXX]`. Degrades gracefully to a deterministic template if offline. |

---

## 7. Benchmark Results & Quantitative Evaluation

The system is rigorously benchmarked via [`eval.py`](eval.py), evaluating three core metrics across 15 real-world aircraft subsystems and 10 adversarial near-miss test cases:

```bash
.venv/bin/python eval.py
```

```
============================================================
RUNNING COPILOT BENCHMARK & EVALUATION SUITE
============================================================

[Metric 1] Retrieval Hit@5: 15/15 (100.0%) [12/15 at Rank 1]
[Metric 2] Refusal Accuracy: 10/10 (100.0%)
[Metric 3] Citation Precision: 100.0%
============================================================
EVALUATION COMPLETED
============================================================
```

### Retrieval Accuracy (Hit@5)

Evaluated across 15 diverse pilot discrepancy reports spanning every major aircraft subsystem:

| # | Aircraft Subsystem | Discrepancy Query | Target IDENT | Retrieval Rank | Result |
| :-: | :--- | :--- | :-: | :-: | :-: |
| 1 | **Ignition** | *Left magneto excessive RPM drop during engine run-up checks* | `100892` | **Rank 1** | HIT |
| 2 | **Fuel System** | *Fuel pressure indicator needle spiking into the red arc intermittently* | `100067` | **Rank 1** | HIT |
| 3 | **Lubrication** | *Abnormally high engine oil pressure observed during ground run* | `100063` | **Rank 1** | HIT |
| 4 | **Electrical** | *Ammeter showing continuous discharge after engine start and battery discharging* | `100523` | **Rank 1** | HIT |
| 5 | **Landing Gear** | *Main landing gear brake linings excessively worn and thin* | `100147` | **Rank 1** | HIT |
| 6 | **Starter** | *Grinding noise on engine start, teeth sheared on starter ring gear* | `100309` | **Rank 1** | HIT |
| 7 | **Engine Cooling** | *Right cowl flap creeps shut and will not stay open in flight* | `101613` | **Rank 1** | HIT |
| 8 | **Flight Controls**| *Top and bottom rudder aerodynamic balance covers cracked* | `100823` | **Rank 1** | HIT |
| 9 | **Engine Baffles** | *Crack found on front engine baffle near the muffler heat shroud* | `100825` | **Rank 3** | HIT |
| 10 | **Pneumatics** | *Vacuum instrument hose chafing against oil cooler line* | `100065` | **Rank 4** | HIT |
| 11 | **Powerplant** | *Engine hesitates and briefly loses power during takeoff roll* | `100004` | **Rank 3** | HIT |
| 12 | **Induction** | *Induction air filter element clogged and due for replacement* | `101381` | **Rank 1** | HIT |
| 13 | **Airframe Cooling**| *L/H engine #3 aft cooling baffle has fatigue crack* | `104638` | **Rank 1** | HIT |
| 14 | **Alternator** | *Alternator drive belt slipping and loose bracket bolts* | `101071` | **Rank 1** | HIT |
| 15 | **Exhaust** | *Exhaust gas leak at cylinder 3 exhaust riser seal flange* | `100193` | **Rank 2** | HIT |

* **Overall Retrieval Hit@5:** **100.0%** (15/15)
* **First-Rank Retrieval (Hit@1):** **80.0%** (12/15)

---

### Adversarial Near-Miss Refusal Accuracy

To test the refusal safeguard against "hallucination under uncertainty", the evaluation suite challenges the copilot with **10 adversarial near-miss queries** ([`refusal_tests.txt`](refusal_tests.txt)). These queries describe mechanically plausible general aviation failure modes on real fleet parts, but have **zero historical precedent in the 2012–2017 UND fleet logbooks**:

| # | Adversarial Near-Miss Query | Real Part? | Logbook Precedent? | Expected Behavior | Model Behavior |
| :-: | :--- | :-: | :-: | :---: | :---: |
| 1 | *Fuel tank selector valve jammed with fuel leaking into cockpit carpet* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 2 | *Magneto internal plastic distributor gear teeth stripped and seized* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 3 | *Propeller governor internal flyweights separated and jammed at high pitch stop* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 4 | *Rudder pedal toe brake master cylinder piston snapped at cylinder base* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 5 | *Starter motor bendix drive clutch fractured and disintegrated inside accessory case* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 6 | *Muffler internal flame cone collapsed and completely plugged tailpipe outlet* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 7 | *Oil filter adapter bypass valve spring broken open with zero oil pressure reading* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 8 | *Carburetor brass float punctured and sunk causing continuous fuel overflow through throat* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 9 | *Alternator internal stator windings melted with smoke and electrical fire in cockpit* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |
| 10 | *Throttle mixture vernier control inner wire sheared inside cockpit quadrant* | Yes | None | **REFUSE** | **REFUSED (100% Correct)** |

* **Refusal Accuracy:** **100.0%** (10/10 adversarial queries correctly refused).
* **Resulting Hallucination Rate:** **0.0%**.

---

### Citation Precision

- **Tested Query:** `"Engine choke lose power on departure"`
- **Generated Citations:** 3
- **Verifiable Real Logbook Records:** 3 (`100004`, `100005`, `100008`)
- **Citation Precision:** **100.0%** (every single cited record exists in the dataset and contains verified technician clearing actions).

---

## 8. Repository Structure

```
Aviation_Maintanence_Copilot-NexAIHack/
├── README.md                 # System architecture, design decisions & benchmarks
├── LOG.md                    # Engineering development log & milestone history
├── LICENSE                   # MIT License
├── requirements.txt          # Python runtime dependencies
├── run.sh                    # Launcher script for Streamlit Glass Cockpit UI
├── refusal_tests.txt         # 10 adversarial near-miss test queries
├── normalizer.py             # MaintNet abbreviation & aviation shorthand normalizer
├── indexer.py                # Hybrid BM25 + TF-IDF indexer, RRF & refusal gate
├── graph_builder.py          # NetworkX Knowledge Graph builder & causal traversal
├── agent.py                  # Agent loop, Gemma 4 rewrite & dual synthesis
├── app.py                    # Streamlit Glass Cockpit HUD & chronic analytics
├── eval.py                   # Automated benchmark suite (Hit@5, refusal, citation)
├── aviation_kg.json          # Cached Knowledge Graph topology (960 KB)
└── dataset/                  # MaintNet & Aviation Lexicon CSV datasets
    ├── Aircraft_Annotation_DataFile.csv       # 6,169 UND maintenance records
    ├── Aviation_Abbreviation_Dataset.csv      # MaintNet abbreviation mappings
    ├── Aviation_TermBanks_Dataset.csv          # Standard aviation domain term bank
    ├── Aviation_grammar_Dataset.csv            # Grammatical syntax patterns
    └── Aviation_Morphosyntactic_Dataset.csv    # Morphological part-of-speech mappings
```

---

## 9. Installation & Quickstart

### Prerequisites

- **Operating System:** Linux, macOS, or Windows (WSL2 recommended)
- **Python:** Python 3.10, 3.11, or 3.12
- **Hardware:** Runs entirely on CPU with <500MB RAM. (No GPU required).
- **Optional Local LLM:** [Ollama](https://ollama.com/) with model `gemma4:e2b` for conversational slang query expansion.

### Quick Launch

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/shyam-shankar/Aviation_Maintanence_Copilot-NexAIHack.git
   cd Aviation_Maintanence_Copilot-NexAIHack
   ```

2. **Set Up Python Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Launch the Glass Cockpit UI:**
   ```bash
   bash run.sh
   # Or directly:
   streamlit run app.py --server.port 8501
   ```

4. **Access the Application:**  
   Open your browser and navigate to **`http://localhost:8501`**.

---

### Running Automated Benchmarks

To execute the automated evaluation suite against the 15 subsystem queries and 10 adversarial near-miss refusals:

```bash
# Ensure virtual environment is active
source .venv/bin/activate

# Run evaluation
python eval.py
```

---

### Configuring Local Gemma 4 (Ollama)

To enable conversational slang query rewriting and local LLM report synthesis:

1. Install Ollama from [ollama.com](https://ollama.com/).
2. Pull the compact Gemma 4 model:
   ```bash
   ollama pull gemma4:e2b
   ```
3. Start the Ollama daemon:
   ```bash
   ollama serve
   ```
4. The copilot automatically detects `http://localhost:11434` on startup and enables the Gemma 4 status badge in the Glass Cockpit header. If Ollama is not running, the application gracefully operates in **Deterministic Mode** with zero performance degradation.

---

## 10. Maintenance & Future Roadmap

- [x] **Sublinear TF-IDF + BM25 Hybrid Retrieval** with Reciprocal Rank Fusion.
- [x] **NetworkX Causal Knowledge Graph** mapping `System` $\to$ `Component` $\to$ `Symptom` $\to$ `Action`.
- [x] **Multi-Gate Refusal Gate** achieving 100% refusal accuracy on out-of-domain and adversarial near-miss queries.
- [x] **Fleet Chronic Defect Analytics** identifying top recurring fleet failure modes.
- [x] **Gemma 4 Integration** for conversational slang query rewriting.
- [ ] **Expansion to Avionics & Hydraulics:** Ingest additional ATA chapter lexicons for Glass Cockpit displays (Garmin G1000) and landing gear hydraulic actuators.
- [ ] **Illustrated Parts Catalog (IPC) Cross-Referencing:** Link verified logbook component entries directly to manufacturer IPC part callouts.
- [ ] **Voice-to-Text Squelch Logging:** Enable hands-free voice transcription for hangar floor mechanics via local Whisper.cpp.

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).

---

> [!NOTE]  
> **Attribution & Acknowledgments:** Maintenance dataset derived from the MaintNet aviation maintenance research initiative conducted in cooperation with the University of North Dakota (UND) Department of Aviation (2012–2017).
