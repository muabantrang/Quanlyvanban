import json
import sys
import os
from pathlib import Path

# Add parent to path for imports
parent_dir = str(Path(r"c:\Rag_Agribank_Thuchanh\RAG\rag_foundation\buoi_08").resolve())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import advanced_rag
from rag import CHUNKS_DIR

print("======================================================")
print("GIAI ĐOẠN 1 — CHẨN ĐOÁN")
print("======================================================")

# 1. MỞ FILE BÁO CÁO
report_path = Path(parent_dir) / "reports" / "report.json"
print("\n[1. MỞ FILE BÁO CÁO]")
print(f"Đường dẫn tuyệt đối: {report_path.absolute()}")
if report_path.exists():
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    print(f"Toàn bộ khoá cấp cao nhất: {list(report_data.keys())}")
    
    metrics = report_data.get("metrics", {})
    for mode in ["bm25", "semantic", "hybrid", "hybrid_rerank"]:
        # Wait, report.json only stores summary metrics, not per-query arrays.
        print(f"Mode: {mode}")
else:
    print("Không tìm thấy file báo cáo.")

# 2. NẠP BỘ CÂU HỎI
eval_file = Path(parent_dir) / "eval" / "questions.json"
print("\n[2. NẠP BỘ CÂU HỎI]")
print(f"Đường dẫn tuyệt đối: {eval_file.absolute()}")
print(f"File có tồn tại không: {eval_file.exists()}")
if eval_file.exists():
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Kiểu dữ liệu sau json.load: {type(data)}")
    if isinstance(data, dict):
        print(f"Khoá cấp cao nhất: {list(data.keys())}")
    
    # Simulate current evaluate.py logic
    if isinstance(data, list):
        questions = data
    else:
        questions = data.get("questions", [])
    print(f"Số câu hỏi cuối cùng nạp được: {len(questions)}")
else:
    questions = []

# 3. ĐỐI CHIẾU CHUNK_ID
print("\n[3. ĐỐI CHIẾU CHUNK_ID]")
gold_ids = set()
for q in questions:
    gold_ids.update(q.get("relevant_chunk_ids", []))
    
chunks, stats = advanced_rag.load_chunks(CHUNKS_DIR, "hierarchical")
corpus_ids = set([c["chunk_id"] for c in chunks])

intersection = gold_ids.intersection(corpus_ids)
print(f"Số ID gold: {len(gold_ids)}")
print(f"Số ID corpus: {len(corpus_ids)}")
print(f"Số ID giao nhau: {len(intersection)}")
print(f"3 ID mẫu gold: {list(gold_ids)[:3]}")
print(f"3 ID mẫu corpus: {list(corpus_ids)[:3]}")

# 4. CHẠY THỬ MỘT CÂU HỎI, KHÔNG BẮT EXCEPTION
print("\n[4. CHẠY THỬ MỘT CÂU HỎI]")
if questions:
    q = questions[0]
    query_text = q.get("question", "")
    print(f"Query: {query_text}")
    import time
    start = time.time()
    config = advanced_rag.load_config()
    res = advanced_rag.compare_retrieval(query_text, chunks, "hierarchical", config)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    
    print(f"Số evidence BM25: {len(res['bm25']['evidence'])}")
    print(f"Số evidence Semantic: {len(res['semantic']['evidence'])}")
    print(f"Thời gian chạy (ms): {elapsed_ms:.2f}")

print("======================================================")
