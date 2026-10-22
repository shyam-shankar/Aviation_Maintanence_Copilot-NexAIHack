"""
Automated Evaluation Suite for Aviation Maintenance Copilot
Evaluates:
1. Retrieval Hit@5 on 15 technician fault queries
2. Correct refusal rate on 5 out-of-domain / impossible queries
3. Citation precision
"""

from indexer import HybridIndex
from agent import MaintenanceCopilotAgent
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(BASE_DIR, "dataset", "Aircraft_Annotation_DataFile.csv")
ABBREV_CSV = os.path.join(BASE_DIR, "dataset", "Aviation_Abbreviation_Dataset.csv")

# 15 Ground-Truth Test Cases (Query -> Expected Target Record IDENT)
TEST_PAIRS = [
    ("Engine choke and briefly lose power on departure", "100004"),
    ("#2 and 4 cylinder rocker cover gaskets are leaking", "100005"),
    ("Induction tube hose clamps loose all cylinders", "100007"),
    ("#4 rocker cover is leaking", "100010"),
    ("#1, 2, & 6 cowl mounts torn", "100050"),
    ("Oil found on entire right side of fuselage", "100101"),
    ("#2 intake leaking", "100251"),
    ("Aft left hand engine baffle cracked", "100500"),
    ("L/H #1 side baffle seal loose", "100751"),
    ("#2 & 3 rocker covers on L/H engine leaking", "101001"),
    ("#2, 3, 4 rocker gaskets leaking", "101501"),
    ("CYL #1 rocker cover gasket leaking", "102001"),
    ("Upper cowling needs small piece of chafe tape", "103001"),
    ("#2 intake gasket leaking", "104001"),
    ("Alternator belt loose", "101071")
]

# 5 Negative / Out-of-Domain Refusal Cases
REFUSAL_TESTS = [
    "Boeing 777 toilet flush vacuum sensor failed in coach",
    "Tesla Model 3 lithium battery charging port latch stuck",
    "Passenger seat row 14 tray table hinge cracked in cabin",
    "Nuclear submarine propulsion ballast valve pressure drop",
    "Airbus A380 upper deck galley espresso machine water leak"
]

def run_evaluation():
    print("=" * 60)
    print("🚀 RUNNING COPILOT BENCHMARK & EVALUATION SUITE")
    print("=" * 60)

    index = HybridIndex(DATASET_CSV, ABBREV_CSV)
    agent = MaintenanceCopilotAgent(index)

    # 1. Hit@5 Evaluation
    print("\n[Metric 1] Retrieval Hit@5 on 15 Technician Query Pairs:")
    hits = 0
    for query, expected_id in TEST_PAIRS:
        res = agent.search_records(query, top_k=5)
        candidate_ids = [c["ident"] for c in res["candidates"]]
        expected_rec = index.get_record(expected_id)
        expected_prob_norm = expected_rec["norm_problem"] if expected_rec else ""
        
        # A hit occurs if the exact IDENT is in top 5, or an exact equivalent recurring defect record is in top 5
        is_hit = False
        hit_rank = -1
        for rank, cand in enumerate(res["candidates"][:5], start=1):
            if cand["ident"] == expected_id:
                is_hit = True
                hit_rank = rank
                break
            elif cand["norm_problem"] == expected_prob_norm or (expected_prob_norm in cand["norm_problem"] and len(expected_prob_norm) > 10):
                is_hit = True
                hit_rank = rank
                break

        if is_hit:
            hits += 1
            print(f"  ✅ [Hit @ Rank {hit_rank}] Query: '{query}' -> Found: [IDENT: {res['candidates'][hit_rank-1]['ident']}] ('{res['candidates'][hit_rank-1]['problem']}')")
        else:
            print(f"  ❌ [Miss] Query: '{query}' -> Expected: {expected_id}, Got: {candidate_ids[:3]}")

    hit_rate = (hits / len(TEST_PAIRS)) * 100
    print(f"\n📊 Retrieval Hit@5: {hits}/{len(TEST_PAIRS)} ({hit_rate:.1f}%)")

    # 2. Refusal Evaluation
    print("\n" + "-" * 60)
    print("[Metric 2] Correct Refusal Rate on 5 Out-of-Domain Queries:")
    refusals = 0
    for q in REFUSAL_TESTS:
        resp = agent.answer_fault(q, enable_llm=False)
        is_refused = resp["status"] == "refusal"
        if is_refused:
            refusals += 1
            print(f"  🛡️ [Correct Refusal] '{q}' -> Refused: '{resp['answer'][:40]}...'")
        else:
            print(f"  ⚠️ [Hallucination / Failed Refusal] '{q}' -> Answered with {len(resp['cited_records'])} citations")

    refusal_rate = (refusals / len(REFUSAL_TESTS)) * 100
    print(f"\n📊 Refusal Accuracy: {refusals}/{len(REFUSAL_TESTS)} ({refusal_rate:.1f}%)")

    # 3. Citation Precision
    print("\n" + "-" * 60)
    print("[Metric 3] Citation Precision Verification:")
    sample_q = "Engine choke lose power on departure"
    resp = agent.answer_fault(sample_q, enable_llm=False)
    cited = resp["cited_records"]
    valid_citations = 0
    for c in cited:
        # Check that the record actually exists in the database and has non-empty action
        raw = index.get_record(c["ident"])
        if raw and raw["action"].strip():
            valid_citations += 1

    precision = (valid_citations / len(cited)) * 100 if cited else 0
    print(f"  Tested Query: '{sample_q}'")
    print(f"  Total Citations Generated: {len(cited)}")
    print(f"  Verified Valid Citations: {valid_citations}")
    print(f"📊 Citation Precision: {precision:.1f}%\n")

    print("=" * 60)
    print("✅ EVALUATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
