import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import advanced_rag
from rag import CHUNKS_DIR

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    intersect = retrieved_k.intersection(relevant_set)
    return len(intersect) / len(relevant_set)

def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(k, len(relevant_set))):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(questions_file: Path, strategy: str, k: int, config: dict, mock_deps=None) -> dict:
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, list):
        questions = data
        needs_human_review = any(q.get("needs_human_review", False) for q in questions)
    else:
        questions = data.get("questions", [])
        needs_human_review = data.get("needs_human_review", False)
        
    if not questions:
        raise ValueError("No questions found in file.")
        
    chunks, stats = advanced_rag.load_chunks(CHUNKS_DIR, strategy)
    
    modes = ["bm25", "semantic", "hybrid", "hybrid_rerank"]
    results = {m: {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "latency": []} for m in modes}
    
    has_errors = False
    
    for q in questions:
        query_text = q.get("question")
        relevant_ids = q.get("relevant_chunk_ids", [])
        
        try:
            res = advanced_rag.compare_retrieval(query_text, chunks, strategy, config, mock_deps)
            
            for mode in modes:
                mode_data = res.get(mode, {})
                evidences = mode_data.get("evidence", [])
                
                retrieved_ids = [c["chunk_id"] for c in evidences]
                
                recall = recall_at_k(retrieved_ids, relevant_ids, k)
                mrr = mrr_at_k(retrieved_ids, relevant_ids, k)
                ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)
                
                lat = mode_data.get("trace", {}).get("latency_ms", {}).get("total", 0.0)
                
                results[mode]["recall"] += recall
                results[mode]["mrr"] += mrr
                results[mode]["ndcg"] += ndcg
                results[mode]["latency"].append(lat)
                
        except Exception as e:
            print(f"Error evaluating query '{query_text}': {e}")
            has_errors = True
            
    num_q = len(questions)
    
    metrics_summary = {}
    for m in modes:
        lats = sorted(results[m]["latency"])
        mean_lat = sum(lats) / len(lats) if lats else 0.0
        p50_lat = lats[len(lats)//2] if lats else 0.0
        
        metrics_summary[m] = {
            "Recall@K": results[m]["recall"] / num_q,
            "MRR@K": results[m]["mrr"] / num_q,
            "nDCG@K": results[m]["ndcg"] / num_q,
            "latency_mean_ms": mean_lat,
            "latency_p50_ms": p50_lat
        }
        
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "strategy": strategy,
            "k": k,
            "eval_file": str(questions_file)
        },
        "needs_human_review": needs_human_review,
        "metrics": metrics_summary,
        "has_errors": has_errors
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Evaluate Advanced RAG Pipeline")
    parser.add_argument("--strategy", type=str, default="hierarchical", help="Chunk strategy")
    parser.add_argument("--k", type=int, default=5, help="Top K to evaluate")
    parser.add_argument("--eval_file", type=str, help="Path to eval JSON file", default=None)
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    if args.eval_file:
        eval_path = Path(args.eval_file)
    else:
        eval_path = base_dir / "eval" / "questions.json"
        
    config = advanced_rag.load_config()
    
    print(f"Start evaluation with strategy={args.strategy}, k={args.k}")
    try:
        report = evaluate(eval_path, args.strategy, args.k, config)
        
        reports_dir = base_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        report_file = reports_dir / "report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"\nReport exported at: {report_file}")
        
        if report.get("needs_human_review"):
            print("\nWARNING: Gold labels need human review (needs_human_review=true).")
            print("These metrics are for reference only.")
            
        import pandas as pd
        df = pd.DataFrame(report["metrics"]).T
        print("\n[METRICS]")
        print(df.to_string())
        
        if report["has_errors"]:
            print("\nWARNING: Some queries failed during execution, see logs above.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
