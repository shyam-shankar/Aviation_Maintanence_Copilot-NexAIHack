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

# 15 Hard & Diverse Ground-Truth Test Cases (Conversational Pilot Discrepancy -> Target Record)
TEST_PAIRS = [
    ("Left magneto excessive RPM drop during engine run-up checks", "100892"),
    ("Fuel pressure indicator needle spiking into the red arc intermittently", "100067"),
    ("Abnormally high engine oil pressure observed during ground run", "100062"),
    ("Ammeter showing continuous discharge after engine start and battery discharging", "100523"),
    ("Main landing gear brake linings excessively worn and thin", "100147"),
    ("Grinding noise on engine start, teeth sheared on starter ring gear", "100309"),
    ("Right cowl flap creeps shut and will not stay open in flight", "101613"),
    ("Top and bottom rudder aerodynamic balance covers cracked", "100823"),
    ("Crack found on front engine baffle near the muffler heat shroud", "100825"),
    ("Vacuum instrument hose chafing against oil cooler line", "100065"),
    ("Engine hesitates and briefly loses power during takeoff roll", "100004"),
    ("Induction air filter element clogged and due for replacement", "101381"),
    ("L/H engine #3 aft cooling baffle has fatigue crack", "104638"),
    ("Alternator drive belt slipping and loose bracket bolts", "101071"),
    ("Exhaust gas leak at cylinder 3 exhaust riser seal flange", "100193"),
]

# 10 Adversarial Near-Miss Refusal Cases (Plausible light aircraft faults on fleet components, but zero logbook precedent)
REFUSAL_TESTS = [
    "Fuel tank selector valve jammed with fuel leaking into cockpit carpet",
    "Magneto internal plastic distributor gear teeth stripped and seized",
    "Propeller governor internal flyweights separated and jammed at high pitch stop",
    "Rudder pedal toe brake master cylinder piston snapped at cylinder base",
    "Starter motor bendix drive clutch fractured and disintegrated inside accessory case",
    "Muffler internal flame cone collapsed and completely plugged tailpipe outlet",
    "Oil filter adapter bypass valve spring broken open with zero oil pressure reading",
    "Carburetor brass float punctured and sunk causing continuous fuel overflow through throat",
    "Alternator internal stator windings melted with smoke and electrical fire in cockpit",
    "Throttle mixture vernier control inner wire sheared inside cockpit quadrant",
]

def run_evaluation():
    print("=" * 60)
    print("RUNNING COPILOT BENCHMARK & EVALUATION SUITE")
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
            print(f"  [HIT @ Rank {hit_rank}] Query: '{query}' -> Found: [IDENT: {res['candidates'][hit_rank-1]['ident']}] ('{res['candidates'][hit_rank-1]['problem']}')")
        else:
            print(f"  [MISS] Query: '{query}' -> Expected: {expected_id}, Got: {candidate_ids[:3]}")

    hit_rate = (hits / len(TEST_PAIRS)) * 100
    print(f"\nRetrieval Hit@5: {hits}/{len(TEST_PAIRS)} ({hit_rate:.1f}%)")

    # 2. Refusal Evaluation
    print("\n" + "-" * 60)
    print("[Metric 2] Correct Refusal Rate on 10 Adversarial Near-Miss Queries:")
    refusals = 0
    for q in REFUSAL_TESTS:
        resp = agent.answer_fault(q, enable_llm=False)
        is_refused = resp["status"] == "refusal"
        if is_refused:
            refusals += 1
            print(f"  [CORRECT REFUSAL] '{q}' -> Refused: '{resp['answer'][:40]}...'")
        else:
            print(f"  [FAILED REFUSAL] '{q}' -> Answered with {len(resp['cited_records'])} citations")

    refusal_rate = (refusals / len(REFUSAL_TESTS)) * 100
    print(f"\nRefusal Accuracy: {refusals}/{len(REFUSAL_TESTS)} ({refusal_rate:.1f}%)")

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
    print(f"Citation Precision: {precision:.1f}%\n")

    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
