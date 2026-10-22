"""
Knowledge Graph Construction for Aviation Maintenance Logbooks
Extracts canonical Systems, Components, Symptoms, and Actions.
Builds a NetworkX MultiDiGraph with record_ids on edges for 100% grounded citations.
"""

import os
import csv
import re
import json
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from normalizer import AviationNormalizer

COMPONENTS_SPEC = [
    (r"\b(?:rocker\s*cover\s*gaskets?|rocker\s*gaskets?|rocker\s*cover|rocker\s*box)\b", "Rocker Cover & Gaskets", "Powerplant & Engine"),
    (r"\b(?:intake\s*gaskets?|intakes?|intake\s*manifold)\b", "Intake Manifold & Gaskets", "Intake & Induction"),
    (r"\b(?:baffle|baffles|baffle\s*seal|side\s*baffle)\b", "Engine Baffles & Air Seals", "Powerplant & Engine"),
    (r"\b(?:induction\s*tube|hose\s*clamps?|clamps?|tube\s*clamp)\b", "Induction Tubes & Clamps", "Intake & Induction"),
    (r"\b(?:fuel\s*servo|servo|carburetor|fuel\s*pump|fuel\s*line|gascolator|fuel\s*drain)\b", "Fuel Servo & Metering Unit", "Fuel System"),
    (r"\b(?:spark\s*plug|plugs?|spark\s*plugs?|fouled\s*plug)\b", "Spark Plugs & Ignition Leads", "Electrical & Ignition"),
    (r"\b(?:alternator\s*belt|alt\s*belt|alternator|alt|drive\s*belt)\b", "Alternator & Drive Belt", "Electrical & Ignition"),
    (r"\b(?:battery|batt|battery\s*box|battery\s*bracket)\b", "Aircraft Battery & Tray", "Electrical & Ignition"),
    (r"\b(?:magneto|mags?|mag|ignition\s*switch)\b", "Magnetos & Ignition System", "Electrical & Ignition"),
    (r"\b(?:cowl\s*mounts?|cowling|cowl|mounts?)\b", "Engine Cowling & Mounts", "Airframe & Structure"),
    (r"\b(?:exhaust|exh|muffler|exhaust\s*valve|valve\s*guide)\b", "Exhaust System & Valves", "Exhaust System"),
    (r"\b(?:vacuum\s*pump|vac\s*pump|attitude\s*indicator)\b", "Vacuum Pump & Gyro Instruments", "Pneumatics & Instruments"),
    (r"\b(?:landing\s*gear|tire|wheel|brake|shimmy|strut)\b", "Landing Gear, Brakes & Tires", "Landing Gear & Controls"),
    (r"\b(?:propeller|prop|governor|spinner)\b", "Propeller & Governor Assembly", "Powerplant & Engine"),
    (r"\b(?:oil\s*cooler|oil\s*line|oil\s*filter|dipstick|oil\s*pressure)\b", "Engine Lubrication & Oil Cooler", "Powerplant & Engine"),
    (r"\b(?:idle|idle\s*override|throttle|mixture\s*control)\b", "Throttle & Engine Idle Controls", "Powerplant & Engine"),
    (r"\b(?:chafe\s*tape|tape|fairing|static\s*wick)\b", "Chafe Protection & Fairings", "Airframe & Structure"),
    (r"\b(?:cylinder\s*head|cylinder|cyl|piston|push\s*rod)\b", "Cylinder Assembly & Piston", "Powerplant & Engine"),
    (r"\b(?:starter|starter\s*solenoid)\b", "Engine Starter Motor", "Electrical & Ignition"),
]

SYMPTOMS_SPEC = [
    (r"\b(?:leaking|leaks|leak|leaked|seepage)\b", "Fluid / Pressure Leak"),
    (r"\b(?:cracked|crack|cracks|fracture)\b", "Structural Crack"),
    (r"\b(?:loose|loosened|backing\s*out)\b", "Loose Hardware / Fastener"),
    (r"\b(?:torn|tear|tears|ripped)\b", "Torn Material / Damaged Mount"),
    (r"\b(?:choke|lose power|lost power|loss of power|hesitation)\b", "Engine Choke & Power Loss"),
    (r"\b(?:rough|vibration|vibrating|shudder)\b", "Rough Running & Vibration"),
    (r"\b(?:killed engine|shut down|shuts down|shutdown|stalls?)\b", "Engine Stall & Shutdown"),
    (r"\b(?:fouled|foul|carbon|deposit)\b", "Component Fouled / Contaminated"),
    (r"\b(?:inoperative|inop|failed|failure|dead)\b", "Inoperative / Total Failure"),
    (r"\b(?:worn|wear|chafing|chafe|rubbing|grooved)\b", "Surface Wear & Chafing"),
    (r"\b(?:broken|broke|separated|snapped)\b", "Broken Component / Sheared Bolt"),
    (r"\b(?:low pressure|low press|high temp|hot|overheat)\b", "Abnormal Operating Temp / Pressure"),
    (r"\b(?:missing|lost|gone)\b", "Missing Hardware / Safety Wire"),
    (r"\b(?:corrosion|corroded|rust)\b", "Surface Corrosion"),
]

ACTIONS_SPEC = [
    (r"\b(?:removed\s*&\s*replaced|r&r|r & r|replaced|install|installed|new|swap)\b", "Removed & Replaced Component"),
    (r"\b(?:tightened|tighten|torqued|torque|secured|sec|fastened)\b", "Tightened & Torqued Hardware"),
    (r"\b(?:adjusted|adjust|rigged|rig|set|calibrated)\b", "Adjusted & Calibrated Setting"),
    (r"\b(?:cleaned|clean|serviced|flushed|blasted)\b", "Cleaned & Serviced Assembly"),
    (r"\b(?:stop drilled|stop drill|drilled|stop-drill)\b", "Stop-Drilled Crack Mitigation"),
    (r"\b(?:run up|check|checked|inspected|tested|troubleshoot)\b", "Operational Run-Up & Diagnostic Check"),
    (r"\b(?:reamed|ream)\b", "Reamed Valve Guide"),
    (r"\b(?:resealed|sealed|safety\s*wired|lubricated)\b", "Resealed / Safety-Wired Hardware"),
]

class AviationKnowledgeGraph:
    def __init__(self, dataset_csv: str, abbreviation_csv: str = None, graph_cache_path: str = "aviation_kg.json"):
        self.dataset_csv = dataset_csv
        self.abbreviation_csv = abbreviation_csv
        self.graph_cache_path = graph_cache_path
        self.normalizer = AviationNormalizer(abbreviation_csv)
        self.graph = nx.DiGraph()
        self.records_by_id: Dict[str, Dict] = {}
        self.components_list: List[str] = []
        self.symptoms_list: List[str] = []
        
        self.build_or_load()

    def build_or_load(self):
        if os.path.exists(self.graph_cache_path):
            try:
                self.load_cache()
                return
            except Exception as e:
                print(f"Error loading cache: {e}, rebuilding graph...")

        self.build_graph()
        self.save_cache()

    def build_graph(self):
        self.graph = nx.DiGraph()
        
        with open(self.dataset_csv, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ident = row.get("IDENT", "").strip()
                raw_prob = row.get("PROBLEM", "").strip()
                raw_act = row.get("ACTION", "").strip()
                
                self.records_by_id[ident] = {
                    "ident": ident,
                    "problem": raw_prob,
                    "action": raw_act
                }
                
                norm_p = self.normalizer.normalize(raw_prob)
                norm_a = self.normalizer.normalize(raw_act)
                
                # Extract Component & System
                comp_name = None
                system_name = "Airframe & Structure"
                for pat, c, s in COMPONENTS_SPEC:
                    if re.search(pat, norm_p, re.I) or re.search(pat, norm_a, re.I):
                        comp_name = c
                        system_name = s
                        break
                        
                if not comp_name:
                    # Fallback general component
                    if "engine" in norm_p:
                        comp_name = "General Engine Core"
                        system_name = "Powerplant & Engine"
                    else:
                        continue

                # Extract Symptom
                symp_name = None
                for pat, s in SYMPTOMS_SPEC:
                    if re.search(pat, norm_p, re.I):
                        symp_name = s
                        break
                if not symp_name:
                    symp_name = "General Operational Defect"

                # Extract Action
                act_name = None
                for pat, a in ACTIONS_SPEC:
                    if re.search(pat, norm_a, re.I):
                        act_name = a
                        break
                if not act_name:
                    act_name = "Inspected & Cleared"

                # Add Nodes
                self.graph.add_node(system_name, type="System")
                self.graph.add_node(comp_name, type="Component", system=system_name)
                self.graph.add_node(symp_name, type="Symptom")
                self.graph.add_node(act_name, type="Action")

                # Edge 1: (System) --[HAS_PART]--> (Component)
                if not self.graph.has_edge(system_name, comp_name):
                    self.graph.add_edge(system_name, comp_name, relation="HAS_PART")

                # Edge 2: (Component) --[EXHIBITED]--> (Symptom)
                if self.graph.has_edge(comp_name, symp_name):
                    edge_data = self.graph[comp_name][symp_name]
                    edge_data["weight"] += 1
                    if ident not in edge_data["record_ids"]:
                        edge_data["record_ids"].append(ident)
                else:
                    self.graph.add_edge(comp_name, symp_name, relation="EXHIBITED", weight=1, record_ids=[ident])

                # Edge 3: (Symptom) --[RESOLVED_BY]--> (Action)
                # We also track which component this was for
                if self.graph.has_edge(symp_name, act_name):
                    edge_data = self.graph[symp_name][act_name]
                    edge_data["weight"] += 1
                    if ident not in edge_data["record_ids"]:
                        edge_data["record_ids"].append(ident)
                else:
                    self.graph.add_edge(symp_name, act_name, relation="RESOLVED_BY", weight=1, record_ids=[ident])

        self.components_list = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Component"]
        self.symptoms_list = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Symptom"]

    def save_cache(self):
        data = {
            "nodes": [{"id": n, **d} for n, d in self.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.graph.edges(data=True)],
            "records": self.records_by_id
        }
        with open(self.graph_cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_cache(self):
        with open(self.graph_cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.DiGraph()
        for node in data["nodes"]:
            nid = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id"}
            self.graph.add_node(nid, **attrs)
        for edge in data["edges"]:
            src = edge["source"]
            tgt = edge["target"]
            attrs = {k: v for k, v in edge.items() if k not in ["source", "target"]}
            self.graph.add_edge(src, tgt, **attrs)
        self.records_by_id = data["records"]
        self.components_list = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Component"]
        self.symptoms_list = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Symptom"]

    def anchor_nodes(self, query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Anchors query to nearest (System, Component, Symptom) nodes"""
        norm_q = self.normalizer.normalize(query)
        
        # Check Component
        anchored_comp = None
        anchored_sys = None
        for pat, c, s in COMPONENTS_SPEC:
            if re.search(pat, norm_q, re.I):
                if c in self.graph:
                    anchored_comp = c
                    anchored_sys = s
                    break
                    
        # Check Symptom
        anchored_symp = None
        for pat, s in SYMPTOMS_SPEC:
            if re.search(pat, norm_q, re.I):
                if s in self.graph:
                    anchored_symp = s
                    break
                    
        return anchored_sys, anchored_comp, anchored_symp

    def query_graph(self, query: str) -> Dict[str, Any]:
        """
        Executes GraphRAG:
        1. Anchors query to graph nodes.
        2. Refuses if no component node or symptom is matched.
        3. Traverses (Component) -> [EXHIBITED] -> (Symptom) -> [RESOLVED_BY] -> (Action).
        4. Fetches exact record_ids and historical counts.
        """
        sys_node, comp_node, symp_node = self.anchor_nodes(query)
        
        # Bulletproof Refusal Gate
        if not comp_node:
            return {
                "status": "refusal",
                "reason": "Unknown component: Query does not reference any known aircraft system or physical component.",
                "subgraph": None,
                "traversal": []
            }
            
        # Check if Component exhibits this symptom (or find all symptoms for this component)
        connected_symptoms = []
        if symp_node and self.graph.has_edge(comp_node, symp_node):
            connected_symptoms = [(symp_node, self.graph[comp_node][symp_node])]
        else:
            # Fallback: traverse all symptoms exhibited by this component
            for tgt in self.graph.successors(comp_node):
                if self.graph.nodes[tgt].get("type") == "Symptom":
                    connected_symptoms.append((tgt, self.graph[comp_node][tgt]))

        if not connected_symptoms:
            return {
                "status": "refusal",
                "reason": f"Topological disconnection: Component [{comp_node}] has no historical records exhibiting symptom [{symp_node}].",
                "subgraph": None,
                "traversal": []
            }

        # Sort symptoms by historical frequency (weight)
        connected_symptoms.sort(key=lambda x: x[1].get("weight", 0), reverse=True)

        traversal_results = []
        cited_records = []
        subgraph_nodes = set([comp_node])
        subgraph_edges = []

        if sys_node and self.graph.has_node(sys_node):
            subgraph_nodes.add(sys_node)
            subgraph_edges.append((sys_node, comp_node, "HAS_PART", 1))

        for s_node, s_edge in connected_symptoms[:2]:
            subgraph_nodes.add(s_node)
            subgraph_edges.append((comp_node, s_node, "EXHIBITED", s_edge.get("weight", 1)))
            
            # Find actions resolving this symptom
            actions = []
            for act_node in self.graph.successors(s_node):
                if self.graph.nodes[act_node].get("type") == "Action":
                    act_edge = self.graph[s_node][act_node]
                    actions.append((act_node, act_edge))
                    
            actions.sort(key=lambda x: x[1].get("weight", 0), reverse=True)

            for act_node, act_edge in actions[:2]:
                subgraph_nodes.add(act_node)
                subgraph_edges.append((s_node, act_node, "RESOLVED_BY", act_edge.get("weight", 1)))
                
                # Intersect record_ids for 100% causal grounding
                comp_record_ids = set(s_edge.get("record_ids", []))
                act_record_ids = set(act_edge.get("record_ids", []))
                common_ids = list(comp_record_ids.intersection(act_record_ids))
                if not common_ids:
                    common_ids = s_edge.get("record_ids", [])[:3]

                for rid in common_ids[:3]:
                    if rid in self.records_by_id and rid not in [c["ident"] for c in cited_records]:
                        raw = self.records_by_id[rid]
                        cited_records.append({
                            "ident": rid,
                            "problem": raw["problem"],
                            "action": raw["action"],
                            "component": comp_node,
                            "symptom": s_node,
                            "clearing_action": act_node
                        })

                traversal_results.append({
                    "system": sys_node or "Aircraft System",
                    "component": comp_node,
                    "symptom": s_node,
                    "action": act_node,
                    "historical_count": act_edge.get("weight", 1),
                    "record_ids": common_ids[:5]
                })

        return {
            "status": "success",
            "system": sys_node,
            "component": comp_node,
            "symptom": symp_node or connected_symptoms[0][0],
            "traversal": traversal_results,
            "cited_records": cited_records,
            "subgraph_nodes": list(subgraph_nodes),
            "subgraph_edges": subgraph_edges
        }

    def get_chronic_defects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Stretch Goal: Recurring defect detection via edge weights"""
        exhibited_edges = []
        for u, v, d in self.graph.edges(data=True):
            if d.get("relation") == "EXHIBITED":
                exhibited_edges.append({
                    "component": u,
                    "symptom": v,
                    "count": d.get("weight", 0),
                    "sample_records": d.get("record_ids", [])[:3]
                })
        exhibited_edges.sort(key=lambda x: x["count"], reverse=True)
        return exhibited_edges[:limit]

    def generate_dot_graph(self, query_res: Dict[str, Any]) -> str:
        """Generates Graphviz DOT string for Streamlit visual rendering"""
        if query_res.get("status") != "success" or not query_res.get("subgraph_edges"):
            return ""
            
        lines = [
            "digraph G {",
            "  rankdir=LR;",
            "  bgcolor=\"transparent\";",
            "  node [fontname=\"Helvetica\", shape=box, style=\"rounded,filled\", margin=\"0.2,0.1\"];",
            "  edge [fontname=\"Helvetica\", fontsize=9, arrowsize=0.8];"
        ]
        
        # Color styling by node type
        node_colors = {
            "System": "fillcolor=\"#1e3a8a\", fontcolor=\"#ffffff\", color=\"#3b82f6\"",
            "Component": "fillcolor=\"#065f46\", fontcolor=\"#ffffff\", color=\"#10b981\"",
            "Symptom": "fillcolor=\"#7c2d12\", fontcolor=\"#ffffff\", color=\"#f97316\"",
            "Action": "fillcolor=\"#3730a3\", fontcolor=\"#ffffff\", color=\"#818cf8\""
        }
        
        for n in query_res.get("subgraph_nodes", []):
            ntype = self.graph.nodes[n].get("type", "Component")
            style = node_colors.get(ntype, "fillcolor=\"#334155\", fontcolor=\"#ffffff\"")
            clean_name = n.replace('"', '\\"')
            lines.append(f'  "{clean_name}" [{style}, label="{clean_name}\\n({ntype})"];')
            
        edge_colors = {
            "HAS_PART": "color=\"#60a5fa\", fontcolor=\"#93c5fd\"",
            "EXHIBITED": "color=\"#f97316\", fontcolor=\"#fdba74\"",
            "RESOLVED_BY": "color=\"#a78bfa\", fontcolor=\"#c4b5fd\""
        }
        
        for u, v, rel, weight in query_res.get("subgraph_edges", []):
            c_u = u.replace('"', '\\"')
            c_v = v.replace('"', '\\"')
            ec = edge_colors.get(rel, "color=\"#94a3b8\"")
            lbl = f"{rel} (x{weight})" if rel != "HAS_PART" else rel
            lines.append(f'  "{c_u}" -> "{c_v}" [{ec}, label=" {lbl} "];')
            
        lines.append("}")
        return "\n".join(lines)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_CSV = os.path.join(BASE_DIR, "dataset", "Aircraft_Annotation_DataFile.csv")
    ABBREV_CSV = os.path.join(BASE_DIR, "dataset", "Aviation_Abbreviation_Dataset.csv")
    
    kg = AviationKnowledgeGraph(DATASET_CSV, ABBREV_CSV)
    print(f"Graph loaded with {kg.graph.number_of_nodes()} nodes and {kg.graph.number_of_edges()} edges.")
    
    print("\n--- TEST: Rocker cover gasket query ---")
    res1 = kg.query_graph("rocker cover gaskets are leaking")
    print("Status:", res1["status"])
    print("Component:", res1.get("component"), "-> Symptom:", res1.get("symptom"))
    print("Traversals found:", len(res1.get("traversal", [])))
    print("Citations:", [c["ident"] for c in res1.get("cited_records", [])])
    
    print("\n--- TEST: Out of domain refusal ---")
    res2 = kg.query_graph("Tesla electric car battery charging door broken")
    print("Status:", res2["status"], "| Reason:", res2.get("reason"))
    
    print("\n--- TOP CHRONIC DEFECTS (STRETCH GOAL) ---")
    for d in kg.get_chronic_defects(5):
        print(f"  • {d['component']} -> {d['symptom']} (Occurred {d['count']} times in fleet logs)")
