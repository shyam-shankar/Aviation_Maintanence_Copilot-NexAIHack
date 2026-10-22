"""
AeroGraph Copilot - Glass Cockpit Maintenance Diagnostics
GraphRAG Architecture combining BM25, Semantic TF-IDF, NetworkX Knowledge Graph,
and local Gemma 4 (via Ollama) for Conversational Slang Rewriting & Grounded Synthesis.
"""

import os
import streamlit as st
import pandas as pd
from indexer import HybridIndex
from graph_builder import AviationKnowledgeGraph
from agent import MaintenanceCopilotAgent, SYSTEM_DISCLAIMER

# Page config
st.set_page_config(
    page_title="AeroGraph Copilot | Aviation MRO Diagnostics",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glass Cockpit CSS
st.markdown("""
<style>
    /* Dark Cockpit Theme */
    .stApp {
        background-color: #080d1a;
        color: #e2e8f0;
    }
    
    /* Top Banner */
    .hud-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .hud-title {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 26px;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: 0.5px;
        margin: 0;
    }
    
    .hud-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Status Pills */
    .telemetry-tag {
        display: inline-block;
        font-family: monospace;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        margin-right: 6px;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
    }
    
    .telemetry-alert {
        background-color: #450a0a;
        color: #f87171;
        border: 1px solid #b91c1c;
    }
    
    .telemetry-success {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }

    .telemetry-purple {
        background-color: #3b0764;
        color: #c084fc;
        border: 1px solid #9333ea;
    }

    /* Evidence Cards */
    .evidence-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    
    .evidence-id {
        font-family: monospace;
        font-size: 12px;
        font-weight: bold;
        color: #38bdf8;
    }
    
    .evidence-problem {
        font-size: 13px;
        color: #f1f5f9;
        margin: 6px 0 4px 0;
    }
    
    .evidence-action {
        font-size: 13px;
        color: #34d399;
        font-weight: 500;
    }

    .rewrite-banner {
        background: #0f2438;
        border-left: 4px solid #38bdf8;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        margin-bottom: 12px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(BASE_DIR, "dataset", "Aircraft_Annotation_DataFile.csv")
ABBREV_CSV = os.path.join(BASE_DIR, "dataset", "Aviation_Abbreviation_Dataset.csv")

@st.cache_resource(show_spinner="Initializing AeroGraph Knowledge Graph & Hybrid Index...")
def load_system():
    idx = HybridIndex(DATASET_CSV, ABBREV_CSV)
    kg = AviationKnowledgeGraph(DATASET_CSV, ABBREV_CSV)
    agent = MaintenanceCopilotAgent(idx, kg)
    return idx, kg, agent

index, kg, agent = load_system()

# Cockpit HUD Header
llm_status_badge = (
    '<span class="telemetry-tag telemetry-purple">🤖 GEMMA 4:E2B READY (OLLAMA)</span>'
    if agent.is_ollama_available
    else '<span class="telemetry-tag">DETERMINISTIC MODE</span>'
)

st.markdown(f"""
<div class="hud-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="hud-title">🛫 AEROGRAPH // MRO FLIGHT LOG COPILOT</div>
            <div class="hud-subtitle">Autonomous Knowledge-Graph & Hybrid RAG Diagnostic Engine • UND Aviation Fleet (2012–2017)</div>
        </div>
        <div style="text-align: right;">
            {llm_status_badge}
            <span class="telemetry-tag telemetry-success">KNOWLEDGE GRAPH ACTIVE</span>
            <span class="telemetry-tag">6,169 FLIGHT LOGS</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Mandatory Airworthiness Disclaimer
st.caption(f"🛡️ **MANDATORY AIRWORTHINESS ADVISORY**: {SYSTEM_DISCLAIMER}")

# Sidebar Telemetry & Quick Diagnostic Buttons
with st.sidebar:
    st.markdown("### 🤖 Local LLM Settings")
    enable_llm = st.toggle(
        "Enable Gemma 4 (Ollama)",
        value=agent.is_ollama_available,
        help="Enables conversational slang query rewriting and natural language synthesis."
    )
    if agent.is_ollama_available:
        st.success("🟢 Local Ollama Online: `gemma4:e2b`", icon="✅")
    else:
        st.info("⚪ Ollama offline; using deterministic GraphRAG engine.")

    st.divider()
    st.markdown("### 🎛️ Diagnostic Scenarios")
    st.markdown("Select a real flight discrepancy to test:")
    
    sample_queries = [
        ("⚡ Ignition: Mag RPM drop on runup", "Left magneto excessive RPM drop during engine run-up checks"),
        ("⛽ Fuel: Pressure needle spiking red", "Fuel pressure indicator needle spiking into the red arc intermittently"),
        ("🛞 Gear/Brakes: Worn brake linings", "Main landing gear brake linings excessively worn and thin"),
        ("🛢️ Lube: Abnormally high oil pressure", "Abnormally high engine oil pressure observed during ground run"),
        ("⚙️ Starter: Grinding sheared ring gear", "Grinding noise on engine start, teeth sheared on starter ring gear"),
        ("🌬️ Cooling: Cowl flap creeps shut in flight", "Right cowl flap creeps shut and will not stay open in flight"),
        ("🛡️ Near-Miss Refusal: Cockpit fuel selector leak", "Fuel tank selector valve jammed with fuel leaking into cockpit carpet"),
        ("🛡️ Near-Miss Refusal: Magneto plastic gear seized", "Magneto internal plastic distributor gear teeth stripped and seized"),
        ("🛡️ Near-Miss Refusal: Governor flyweight separation", "Propeller governor internal flyweights separated and jammed at high pitch stop"),
        ("🛡️ Near-Miss Refusal: Sunk carburetor brass float", "Carburetor brass float punctured and sunk causing continuous fuel overflow through throat"),
    ]
    
    for label, query_text in sample_queries:
        if st.button(label, use_container_width=True):
            st.session_state["active_query"] = query_text

    st.divider()
    st.markdown("### 📡 System Architecture")
    st.markdown(
        """
        - **Ontology Nodes:** `System` ➔ `Component` ➔ `Symptom` ➔ `Action`
        - **Edge Properties:** `weight` (occurrences) + `record_ids`
        - **Query Rewriter:** Gemma 4:E2B translates conversational slang
        - **Refusal Gate:** Deterministic topological check + confidence threshold
        """
    )

# Active query management
if "active_query" not in st.session_state:
    st.session_state["active_query"] = "HAD ENGINE CHOKE & BRIEFLY LOSE POWER ON DEPARTURE. FULL THR"

# Navigation Tabs
tab_diag, tab_chronic, tab_ontology = st.tabs([
    "🕹️ Diagnostic Console",
    "📊 Fleet Chronic Defect Analytics",
    "🕸️ Knowledge Graph Ontology"
])

# -------------------------------------------------------------
# TAB 1: Diagnostic Console
# -------------------------------------------------------------
with tab_diag:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query_val = st.text_input(
            "Enter Aircraft Defect, Squelch or Pilot Report:",
            value=st.session_state.get("active_query", ""),
            placeholder="e.g. Engine choke on departure, rocker gasket leak, loose alternator belt...",
            label_visibility="collapsed"
        )
    with col_btn:
        run_diag = st.button("RUN DIAGNOSTIC", type="primary", use_container_width=True)

    if query_val.strip():
        with st.spinner("Executing GraphRAG traversal & verifying citations..."):
            response = agent.answer_fault(query_val.strip(), enable_llm=enable_llm)

        # Status Ribbon
        if response["status"] == "success":
            synth_badge = f"<span class='telemetry-tag telemetry-purple'>SYNTHESIS: {response.get('synthesis_model', 'Gemma 4')}</span>"
            st.markdown(
                f"<div style='margin: 12px 0;'><span class='telemetry-tag telemetry-success'>● STATUS: SUFFICIENT HISTORICAL EVIDENCE IDENTIFIED</span> "
                f"<span class='telemetry-tag'>GRAPH CONFIDENCE: {response['confidence']*100:.1f}%</span> "
                f"<span class='telemetry-tag'>CITATIONS: {len(response['cited_records'])} VERIFIED LOGS</span> "
                f"{synth_badge}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='margin: 12px 0;'><span class='telemetry-tag telemetry-alert'>● STATUS: REFUSAL TRIGGERED (INSUFFICIENT HISTORICAL EVIDENCE)</span> "
                f"<span class='telemetry-tag'>CONFIDENCE: {response['confidence']*100:.1f}%</span> "
                f"<span class='telemetry-tag'>ZERO HALLUCINATION SAFEGUARD ACTIVE</span></div>",
                unsafe_allow_html=True
            )

        # Show Query Rewrite Banner if active
        if response.get("was_rewritten"):
            st.markdown(f"""
            <div class="rewrite-banner">
                🔄 <b>Gemma 4 Conversational Query Expansion Active:</b><br/>
                <i>Original Slang:</i> <code>"{response['query']}"</code> ➔ 
                <i>Search Keywords:</i> <code>"{response['effective_query']}"</code>
            </div>
            """, unsafe_allow_html=True)

        # Main Layout: Left (Verdict + Visual Graph) | Right (Verbatim Evidence Records)
        c_left, c_right = st.columns([6, 5], gap="large")

        with c_left:
            st.markdown(response["answer"])

            # Visual Subgraph Rendering
            if response.get("graph_dot"):
                st.markdown("#### 🌐 Causal Knowledge Subgraph")
                st.caption("Visual path traversed by the agent from Aircraft System ➔ Component ➔ Symptom ➔ Resolving Action:")
                st.graphviz_chart(response["graph_dot"], use_container_width=True)

        with c_right:
            st.markdown("### 📑 Grounded Logbook Records")
            st.caption("Verbatim historical records cited to justify the diagnostic verdict:")
            
            cited = response.get("cited_records", [])
            if cited:
                for rec in cited:
                    st.markdown(f"""
                    <div class="evidence-card">
                        <div style="display: flex; justify-content: space-between;">
                            <span class="evidence-id">IDENT: {rec['ident']}</span>
                            <span style="font-size: 11px; color: #94a3b8;">Match: {rec['score']*100:.0f}%</span>
                        </div>
                        <div class="evidence-problem"><b>Reported:</b> {rec['problem']}</div>
                        <div class="evidence-action"><b>Clearing Action:</b> {rec['action']}</div>
                        <div style="font-size: 10px; color: #64748b; margin-top: 4px;">
                            Component: {rec.get('component', 'N/A')} | Symptom: {rec.get('symptom', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No verified historical logbook records met the confidence threshold. Citations withheld to prevent ungrounded advice.")

        # Tool Execution Trace
        with st.expander("🛠️ Agent Tool-Call Trace (Auditing & Explainability)", expanded=False):
            st.caption("Sequential execution trace of tools called by the agent:")
            for step in response.get("tool_trace", []):
                st.markdown(f"**Step {step['step']}: Tool `{step['tool']}()`**")
                st.json({
                    "parameters": step.get("input"),
                    "result_summary": step.get("output_summary"),
                    "details": step.get("details", {})
                })

# -------------------------------------------------------------
# TAB 2: Fleet Chronic Defect Analytics (The Stretch Goal)
# -------------------------------------------------------------
with tab_chronic:
    st.subheader("📊 Fleet Chronic Defect Analytics (Recurring Defect Detector)")
    st.caption("Top failure modes and high-frequency defect patterns ranked by graph edge weights across all 6,169 aircraft records.")

    chronic_defects = kg.get_chronic_defects(12)
    df_chronic = pd.DataFrame([
        {
            "Aircraft Component": d["component"],
            "Observed Symptom": d["symptom"],
            "Fleet Frequency (Incidents)": d["count"],
            "Sample Citation IDs": ", ".join(d["sample_records"])
        }
        for d in chronic_defects
    ])

    # Key Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Top Failing Component", df_chronic.iloc[0]["Aircraft Component"], f"{df_chronic.iloc[0]['Fleet Frequency (Incidents)']} incidents")
    m2.metric("Primary Defect Type", df_chronic.iloc[0]["Observed Symptom"], "Fluid / Pressure Leaks")
    m3.metric("Fleet Coverage", "6,169 Records", "UND Aviation Program")

    st.dataframe(df_chronic, use_container_width=True, hide_index=True)

    # Visual Bar Chart
    st.bar_chart(
        df_chronic.set_index("Aircraft Component")["Fleet Frequency (Incidents)"],
        color="#38bdf8",
        use_container_width=True
    )

# -------------------------------------------------------------
# TAB 3: Knowledge Graph Ontology
# -------------------------------------------------------------
with tab_ontology:
    st.subheader("🕸️ Aviation Knowledge Graph Schema & Entity Resolution")
    st.markdown(
        """
        The copilot converts unstructured logbook text into four strict node entities linked by typed causal relationships:
        """
    )
    
    schema_dot = """
    digraph Schema {
        rankdir=LR;
        bgcolor="transparent";
        node [fontname="Helvetica", shape=box, style="rounded,filled", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=10, arrowsize=0.8];
        
        System [fillcolor="#1e3a8a", fontcolor="#ffffff", color="#3b82f6", label="System\\n(e.g., Powerplant, Fuel, Electrical)"];
        Component [fillcolor="#065f46", fontcolor="#ffffff", color="#10b981", label="Component\\n(e.g., Cylinder #2, Rocker Gasket, Baffle)"];
        Symptom [fillcolor="#7c2d12", fontcolor="#ffffff", color="#f97316", label="Symptom\\n(e.g., Leak, Choke, Crack, Inop)"];
        Action [fillcolor="#3730a3", fontcolor="#ffffff", color="#818cf8", label="Action\\n(e.g., Replaced, Tightened, Cleaned)"];
        
        System -> Component [color="#60a5fa", fontcolor="#93c5fd", label=" HAS_PART "];
        Component -> Symptom [color="#f97316", fontcolor="#fdba74", label=" EXHIBITED\\n(stores record_ids) "];
        Symptom -> Action [color="#a78bfa", fontcolor="#c4b5fd", label=" RESOLVED_BY\\n(stores record_ids) "];
    }
    """
    st.graphviz_chart(schema_dot, use_container_width=True)

    st.markdown("### 📈 Graph Topology Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Knowledge Nodes", kg.graph.number_of_nodes())
    c2.metric("Causal Edges", kg.graph.number_of_edges())
    c3.metric("Indexed Components", len(kg.components_list))
    c4.metric("Indexed Symptoms", len(kg.symptoms_list))
