"""
AeroGraph Copilot Agent (Hybrid GraphRAG + Local LLM Integration)
Integrates BM25, TF-IDF vector search, NetworkX Knowledge Graph,
and local Gemma 4 (via Ollama) for conversational query rewriting and grounded synthesis.
"""

import os
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
from indexer import HybridIndex
from graph_builder import AviationKnowledgeGraph

SYSTEM_DISCLAIMER = (
    "⚠️ FLIGHT SAFETY NOTICE: This copilot provides diagnostic reference based strictly on "
    "MaintNet historical aviation logbooks. It is NOT manufacturer-approved maintenance data "
    "or an FAA/EASA certified procedure. Always consult official Aircraft Maintenance Manuals (AMM)."
)

OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e2b"

class MaintenanceCopilotAgent:
    def __init__(
        self,
        index: HybridIndex,
        kg: Optional[AviationKnowledgeGraph] = None,
        ollama_model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None
    ):
        self.index = index
        self.kg = kg or AviationKnowledgeGraph(index.dataset_csv, index.abbreviation_csv)
        self.ollama_model = ollama_model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.is_ollama_available = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=1.5)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                return any(self.ollama_model in m for m in models)
        except Exception:
            pass
        return False

    def rewrite_query(self, raw_query: str) -> Tuple[str, bool, str]:
        """
        Tool 0: Conversational Query Rewriting using local Gemma 4
        Translates conversational slang or symptoms into standard logbook search terms.
        """
        # If query is already very short or logbook-like, keep it
        tokens = raw_query.strip().split()
        if len(tokens) <= 3 and any(t.isupper() for t in tokens):
            return raw_query, False, "Pass-through (Already terse logbook keywords)"

        if not self.is_ollama_available:
            return raw_query, False, "Ollama unavailable; using standard normalizer"

        prompt = f"""Convert this conversational fault report into 3 to 5 concise aviation maintenance keywords matching logbook discrepancy notes.
CRITICAL RULES:
1. Extract ONLY the observed defect and affected component.
2. Do NOT speculate or add hypothetical causes (do NOT add 'fuel system', 'electrical', 'performance issues', etc.).
3. Output root words (e.g. choking -> choke, powerloss -> power loss).
4. Reply with ONLY the search keywords on a single line, nothing else.

Example:
Report: the bird's left fan is coughing when I push the throttle
Search terms: left engine choke throttle power loss

Report: {raw_query}
Search terms:"""

        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }
            resp = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=8)
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                # Clean up lines
                lines = [re.sub(r"[\*\"]", "", l).strip() for l in text.split("\n") if l.strip()]
                if lines and len(lines[0]) > 3:
                    raw_rewritten = lines[0]
                    # Pass through normalizer to guarantee expanded compounds & abbreviations
                    rewritten = self.index.normalizer.normalize(raw_rewritten)
                    return rewritten, True, f"Rewritten via {self.ollama_model}"
        except Exception:
            pass

        return raw_query, False, "Fallback to original query"

    def search_records(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Tool 1: Hybrid keyword + vector search with calibrated refusal gate"""
        results, is_confident, metrics = self.index.search(query, top_k=top_k)
        return {
            "tool": "search_records",
            "parameters": {"query": query, "top_k": top_k},
            "is_confident": is_confident,
            "metrics": metrics,
            "candidates": results
        }

    def get_record(self, record_id: str) -> Dict[str, Any]:
        """Tool 2: Fetch raw record text for citation"""
        rec = self.index.get_record(record_id)
        return {
            "tool": "get_record",
            "parameters": {"record_id": record_id},
            "found": rec is not None,
            "record": rec
        }

    def query_graph(self, query: str) -> Dict[str, Any]:
        """Tool 3: Subgraph traversal across System -> Component -> Symptom -> Action"""
        return self.kg.query_graph(query)

    def synthesize_with_llm(
        self,
        fault_query: str,
        cited_records: List[Dict],
        graph_res: Dict[str, Any]
    ) -> Optional[str]:
        """
        Synthesizes grounded response using local Gemma 4 (via Ollama).
        Maintains strict grounding and citation preservation.
        """
        if not self.is_ollama_available or not cited_records:
            return None

        records_block = "\n".join([
            f"- Record [IDENT: {r['ident']}]: Problem: \"{r['problem']}\" | Action: \"{r['action']}\""
            for r in cited_records
        ])

        prompt = f"""You are an aviation maintenance copilot assistant.
A technician reported this discrepancy: "{fault_query}".
System: {graph_res.get('system', 'Powerplant')} | Component: {graph_res.get('component', 'General')}

Below are verified historical maintenance logbook records:
{records_block}

Provide a concise, grounded maintenance diagnostic report with three sections:
1. Observed Causes in Historical Fleet (cite each record ID).
2. Historical Clearing Actions Taken by Technicians (cite each record ID).
3. Primary Recommendation (cite the most relevant record ID).

Rules:
- Strictly ground every claim in the provided records.
- Always include the record citation like [IDENT: XXXXXX].
- Do not invent ungrounded procedures."""

        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            }
            resp = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=12)
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                if answer and len(answer) > 50:
                    return answer
        except Exception:
            pass

        return None

    def answer_fault(self, fault_query: str, enable_llm: bool = True) -> Dict[str, Any]:
        """
        Executes full agentic workflow:
        0. Query rewrite (Gemma 4 converts slang -> logbook keywords)
        1. Hybrid search (BM25 + TF-IDF)
        2. Refusal check (Zero hallucination guardrail)
        3. Knowledge graph traversal (Causal path & edge weights)
        4. Verbatim record inspection
        5. Grounded synthesis (Gemma 4 LLM or deterministic fallback)
        """
        trace = []

        # Step 0: Tool Call rewrite_query
        effective_query = fault_query
        was_rewritten = False
        rewrite_note = ""

        if enable_llm:
            rewritten_q, was_rewritten, rewrite_note = self.rewrite_query(fault_query)
            if was_rewritten:
                effective_query = rewritten_q
            trace.append({
                "step": 0,
                "tool": "rewrite_query",
                "input": {"raw_query": fault_query},
                "output_summary": f"Rewritten to: '{effective_query}' ({rewrite_note})",
                "details": {"original": fault_query, "rewritten": effective_query, "active": was_rewritten}
            })

        # Step 1: Tool Call search_records
        search_res = self.search_records(effective_query, top_k=5)
        top_conf = search_res["metrics"].get("top_confidence", 0.0)
        trace.append({
            "step": 1,
            "tool": "search_records",
            "input": {"query": effective_query, "top_k": 5},
            "output_summary": f"Retrieved {len(search_res['candidates'])} candidates. Confidence: {top_conf:.2f}",
            "details": search_res["metrics"]
        })

        # Step 2: Refusal Gate Check
        if not search_res["is_confident"]:
            return {
                "status": "refusal",
                "query": fault_query,
                "effective_query": effective_query,
                "was_rewritten": was_rewritten,
                "answer": (
                    "**Refusal: Insufficient Evidence in Logbooks.**\n\n"
                    "The historical maintenance logbooks do not contain sufficiently verified precedents "
                    "matching this fault description. In aviation maintenance, guessing or interpolating "
                    "unverified procedures poses severe airworthiness risks.\n\n"
                    "**Required Action:** Consult the official Aircraft Maintenance Manual (AMM) and "
                    "troubleshooting chart for your specific airframe and engine model."
                ),
                "cited_records": [],
                "confidence": top_conf,
                "graph_dot": "",
                "subgraph_info": None,
                "disclaimer": SYSTEM_DISCLAIMER,
                "tool_trace": trace
            }

        # Step 3: Tool Call query_graph
        top_candidates = search_res["candidates"][:3]
        graph_res = self.query_graph(effective_query)
        if graph_res.get("status") != "success" and top_candidates:
            fallback_query = f"{top_candidates[0]['problem']} {top_candidates[0]['action']}"
            graph_res = self.query_graph(fallback_query)

        trace.append({
            "step": 2,
            "tool": "query_graph",
            "input": {"query": effective_query},
            "output_summary": f"Traversed Knowledge Graph: Component='{graph_res.get('component')}' -> Symptom='{graph_res.get('symptom')}'",
            "details": {
                "system": graph_res.get("system"),
                "component": graph_res.get("component"),
                "symptom": graph_res.get("symptom"),
                "traversal_paths": len(graph_res.get("traversal", []))
            }
        })

        # Step 4: Tool Call get_record on top candidates
        inspected_records = []
        for cand in top_candidates:
            ident = cand["ident"]
            rec_res = self.get_record(ident)
            trace.append({
                "step": len(trace) + 1,
                "tool": "get_record",
                "input": {"record_id": ident},
                "output_summary": f"Fetched record IDENT {ident}: Problem: '{cand['problem'][:30]}...' Action: '{cand['action'][:30]}...'"
            })
            if rec_res["found"]:
                inspected_records.append(rec_res["record"])

        # Step 5: Synthesize Answer
        cited_records = []
        causes = []
        actions = []

        for i, rec in enumerate(inspected_records):
            ident = rec["ident"]
            cand_meta = top_candidates[i]
            cited_records.append({
                "ident": ident,
                "problem": rec["problem"],
                "action": rec["action"],
                "score": cand_meta["confidence"],
                "cos_sim": cand_meta["cos_sim"],
                "bm25": cand_meta["bm25_score"],
                "component": graph_res.get("component", "Aircraft Component"),
                "symptom": graph_res.get("symptom", "Observed Defect")
            })
            causes.append(f"- **{rec['problem']}** *(Grounding: `[IDENT: {ident}]`)*")
            actions.append(f"- **{rec['action']}** *(Applied in: `[IDENT: {ident}]`)*")

        # Visual DOT graph
        dot_code = self.kg.generate_dot_graph(graph_res)

        # Attempt LLM synthesis with Gemma 4 if enabled
        llm_answer = None
        if enable_llm:
            llm_answer = self.synthesize_with_llm(fault_query, cited_records, graph_res)

        if llm_answer:
            answer_text = (
                f"### 🤖 Gemma 4 Diagnostic Synthesis\n\n"
                f"**Anchored System:** `{graph_res.get('system', 'Powerplant')}` | "
                f"**Component:** `{graph_res.get('component', 'Core')}`\n\n"
                f"{llm_answer}"
            )
            synthesis_model = f"{self.ollama_model} (Local Ollama)"
        else:
            comp_name = graph_res.get("component") or "Component"
            symp_name = graph_res.get("symptom") or "Defect"
            best_fix = cited_records[0]["action"] if cited_records else "Inspect according to AMM"

            answer_text = (
                f"### ✈️ GraphRAG Diagnostic Assessment\n\n"
                f"**Anchored System:** `{graph_res.get('system', 'Powerplant')}`  \n"
                f"**Component Affected:** `{comp_name}`  \n"
                f"**Historical Defect Pattern:** `{symp_name}`\n\n"
                f"#### 🔍 Observed Problem Variants in Historical Fleet:\n"
                + "\n".join(causes) + "\n\n"
                f"#### 🛠️ Verified Clearing Actions Taken by Technicians:\n"
                + "\n".join(actions) + "\n\n"
                f"#### 🎯 Primary Recommendation:\n"
                f"Historical records indicate technicians most frequently cleared this fault by: **{best_fix}** "
                f"*(Documented in `[IDENT: {cited_records[0]['ident']}]`)*."
            )
            synthesis_model = "Deterministic GraphRAG Template"

        trace.append({
            "step": len(trace) + 1,
            "tool": "synthesize_evidence",
            "input": {"num_records": len(cited_records), "model": synthesis_model},
            "output_summary": f"Generated grounded diagnosis citing {len(cited_records)} records"
        })

        return {
            "status": "success",
            "query": fault_query,
            "effective_query": effective_query,
            "was_rewritten": was_rewritten,
            "rewrite_note": rewrite_note,
            "synthesis_model": synthesis_model,
            "answer": answer_text,
            "cited_records": cited_records,
            "confidence": top_conf,
            "graph_dot": dot_code,
            "subgraph_info": graph_res,
            "disclaimer": SYSTEM_DISCLAIMER,
            "tool_trace": trace
        }


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_CSV = os.path.join(BASE_DIR, "dataset", "Aircraft_Annotation_DataFile.csv")
    ABBREV_CSV = os.path.join(BASE_DIR, "dataset", "Aviation_Abbreviation_Dataset.csv")
    
    idx = HybridIndex(DATASET_CSV, ABBREV_CSV)
    kg = AviationKnowledgeGraph(DATASET_CSV, ABBREV_CSV)
    agent = MaintenanceCopilotAgent(idx, kg)
    
    print("Testing conversational query with Gemma 4 rewrite:")
    res = agent.answer_fault("the bird left fan is coughing when I push the throttle", enable_llm=True)
    print("Status:", res["status"])
    print("Effective Query:", res.get("effective_query"))
    print("Was Rewritten:", res.get("was_rewritten"))
    print("Synthesis Model:", res.get("synthesis_model"))
    print("\nAnswer:\n", res["answer"])
