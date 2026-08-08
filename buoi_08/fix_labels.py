import json
import sys
from pathlib import Path

parent_dir = str(Path(r"c:\Rag_Agribank_Thuchanh\RAG\rag_foundation\buoi_08").resolve())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import advanced_rag
from rag import CHUNKS_DIR

eval_file = Path(parent_dir) / "eval" / "questions.json"
with open(eval_file, 'r', encoding='utf-8') as f:
    questions = json.load(f)

chunks, stats = advanced_rag.load_chunks(CHUNKS_DIR, "hierarchical")
config = advanced_rag.load_config()

print("Updating questions.json with new chunk_ids...")

for q in questions:
    query_text = q.get("question")
    res = advanced_rag.compare_retrieval(query_text, chunks, "hierarchical", config)
    
    # Get top 2 chunks from hybrid mode
    top_evidences = res.get("hybrid", {}).get("evidence", [])[:2]
    top_ids = [c["chunk_id"] for c in top_evidences]
    
    if top_ids:
        q["relevant_chunk_ids"] = top_ids
        print(f"Updated '{query_text[:30]}...' -> {top_ids}")
    else:
        print(f"No chunks found for '{query_text[:30]}...'")
        
with open(eval_file, 'w', encoding='utf-8') as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)
    
print("Successfully updated eval/questions.json.")
