import os
import shutil
import hashlib
from pathlib import Path
import py_compile

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    if not Path(filepath).exists(): return "FILE_NOT_FOUND"
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

rag_dir = Path(r"c:\Rag_Agribank_Thuchanh\RAG")
b08_dir = rag_dir / "rag_foundation" / "buoi_08"
b09_dir = rag_dir / "rag_advanced" / "buoi_09"

print(f"Creating directory {b09_dir}")
b09_dir.mkdir(parents=True, exist_ok=True)

# Directories
for d in ["eval", "reports", "storage/chroma", "storage/hierarchy", "storage/huggingface", "tests/fixtures"]:
    (b09_dir / d).mkdir(parents=True, exist_ok=True)

# .gitkeeps
for k in ["reports/.gitkeep", "storage/chroma/.gitkeep", "storage/hierarchy/.gitkeep", "storage/huggingface/.gitkeep"]:
    (b09_dir / k).touch()
    
# Copy and inject baseline
def copy_baseline(src_name, dest_name):
    src_file = b08_dir / src_name
    dest_file = b09_dir / dest_name
    
    with open(src_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        
    docstring = '"""\nBaseline snapshot từ Buổi 08.\nKhông sửa đổi logic ở bước khởi tạo Buổi 09.\n"""\n'
    
    # Simple injection at top
    new_content = docstring + content
    
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    orig_hash = get_sha256(src_file)
    new_hash = get_sha256(dest_file)
    print(f"{src_name} -> {dest_name}")
    print(f"  Orig SHA-256: {orig_hash}")
    print(f"  New  SHA-256: {new_hash}")

copy_baseline("rag.py", "rag.py")
copy_baseline("advanced_rag.py", "advanced_rag.py")

# .env.example
env_example = """GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=768
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANK_MIN_SCORE=0.50
RERANK_DEVICE=auto

MULTI_QUERY_COUNT=3
MULTI_QUERY_MAX_CHARS=300
MULTI_QUERY_TEMPERATURE=0.2
MULTI_QUERY_ORIGINAL_WEIGHT=1.5
MULTI_QUERY_VARIANT_WEIGHT=1.0
MULTI_QUERY_RRF_K=60
PER_QUERY_CANDIDATES=12
PARENT_MAX_CHARS=6000
PARENT_SCORE_CHILD_LIMIT=3
PARENT_RRF_K=60
PARENT_CANDIDATES=10
FINAL_PARENT_TOP_K=3
TOTAL_CONTEXT_MAX_CHARS=16000
"""
with open(b09_dir / ".env.example", "w", encoding="utf-8") as f:
    f.write(env_example)

# .gitignore
gitignore_content = """__pycache__/
*.pyc
.env
.venv/
storage/chroma/*
!storage/chroma/.gitkeep
storage/hierarchy/*
!storage/hierarchy/.gitkeep
storage/huggingface/*
!storage/huggingface/.gitkeep
reports/*
!reports/.gitkeep
"""
with open(b09_dir / ".gitignore", "w", encoding="utf-8") as f:
    f.write(gitignore_content)
    
# requirements.txt
req_content = """chromadb==0.5.0
google-generativeai==0.8.1
python-dotenv==1.0.1
sentence-transformers==3.0.1
rank_bm25==0.2.2
streamlit==1.37.0
pandas==2.2.2
"""
with open(b09_dir / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(req_content)
    
# SPEC_buoi_09.md
spec_content = """# SPECIFICATION - BUỔI 09: HIERARCHICAL & MULTI-QUERY RAG

## 1. Mục tiêu và khác biệt Buổi 08/09
Mục tiêu của Buổi 09 là khắc phục các nhược điểm đã nhận diện ở Buổi 08 bằng cách kết hợp chiến lược truy vấn phân cấp (Hierarchical Retrieval) và mở rộng truy vấn (Multi-Query Expansion). 
Khác biệt: 
- Buổi 08: Tìm kiếm trên các chunk đơn lẻ (flat), dễ mất ngữ cảnh và bị ảnh hưởng bởi keyword hẹp.
- Buổi 09: Trích xuất các parent document, mở rộng query để bù đắp các keyword còn thiếu, tối ưu hoá kết quả bằng Cross-Query RRF và reranking theo cụm Parent.

## 2. Sơ đồ xử lý
Q0 (Original Query) + variants (LLM generated) 
→ per-query hybrid retrieval (Từng query chạy BM25 + Semantic)
→ cross-query RRF (Tổng hợp kết quả các variant)
→ child-to-parent mapping (Chiếu các chunk con lên parent)
→ parent aggregation (Gom nhóm và tính điểm parent)
→ parent rerank (Dùng Cross-encoder chấm điểm các parent)
→ generation (Tạo đáp án cùng citations).

## 3. Bốn mode đánh giá
- `single_flat`: Truy vấn gốc + Chunk đơn (Giống Buổi 08).
- `multi_flat`: Đa truy vấn + Chunk đơn (Thử nghiệm tác dụng của Multi-query).
- `single_parent`: Truy vấn gốc + Đẩy lên Parent (Thử nghiệm tác dụng của Hierarchy).
- `multi_parent`: Đa truy vấn + Đẩy lên Parent (Pipeline hoàn chỉnh nhất của Buổi 09).

## 4. QueryVariant schema và validation
Định nghĩa cấu trúc cho các truy vấn mở rộng.
Bắt buộc có trường `query`, validate độ dài (`MULTI_QUERY_MAX_CHARS`), loại bỏ các câu hỏi quá giống nhau hoặc không hợp lệ.

## 5. Hierarchy registry schema
Schema lưu trữ cấu trúc cây văn bản, ánh xạ từ `chunk_id` con lên `parent_id`, định nghĩa level và heading.

## 6. ParentDocument schema
Schema chứa dữ liệu của văn bản cha, bao gồm danh sách các child chunks hợp lệ, nội dung ghép lại, độ dài tối đa cho phép.

## 7. MultiQueryChildHit và ParentCandidate schema
- `MultiQueryChildHit`: Biểu diễn kết quả hit của một child từ một variant query nhất định, có trọng số.
- `ParentCandidate`: Cấu trúc nhóm các ChildHit thành một ứng viên Parent để chấm điểm.

## 8. Quy tắc hierarchy resolution và ambiguous warning
Khi một chunk có nhiều parent hoặc không có parent (văn bản lỏng), hệ thống sẽ cảnh báo "ambiguous" và sử dụng fallback strategy (coi nó như một parent độc lập).

## 9. Công thức cross-query RRF và parent aggregation
Sử dụng hằng số RRF (60) để tính điểm cho từng ChildHit từ nhiều queries, sau đó cộng dồn hoặc lấy max (tuỳ cấu trúc điểm) để xếp hạng ParentCandidate. Giới hạn top-k bằng `PARENT_SCORE_CHILD_LIMIT`.

## 10. Context budget và citation contract
Kiểm soát độ dài context gửi vào LLM bằng `TOTAL_CONTEXT_MAX_CHARS`. Nếu các parent vượt budget, hệ thống sẽ cắt gọt hoặc chỉ giữ lại phần mô tả. Citation cần map chuẩn xác về tên văn bản và ID.

## 11. Status/failure contract
Cảnh báo và xử lý lỗi mềm khi API LLM hết hạn mức, hoặc khi Reranker gặp lỗi out of memory. Không đánh sập toàn bộ pipeline.

## 12. Testability/dependency injection
Cho phép truyền mock retriever, mock LLM, và cấu hình tĩnh để dễ dàng unittest.

## 13. Evaluation metrics và acceptance criteria
Sử dụng Recall, MRR, nDCG, và thêm chỉ số đánh giá độ bao phủ Parent.
Acceptance criteria: `multi_parent` phải vượt trội `single_flat` về Recall.

## 14. Xác nhận phạm vi
Chỉ ghi và thực thi logic mới trong phạm vi `rag_advanced/buoi_09/`.
"""
with open(b09_dir / "SPEC_buoi_09.md", "w", encoding="utf-8") as f:
    f.write(spec_content)

# README.md
with open(b09_dir / "README.md", "w", encoding="utf-8") as f:
    f.write("# RAG Advanced - Buổi 09\n\nFramework kết hợp Hierarchical Retrieval và Multi-Query RAG.\n")
    
# Placeholders
py_placeholders = {
    "hierarchical_rag.py": '"""\nTODO: Triển khai pipeline Hierarchical Multi-Query RAG.\nChưa có side effect.\n"""\n',
    "evaluate.py": '"""\nTODO: Module đánh giá 4 modes cho Buổi 09.\nChưa có side effect.\n"""\n',
    "app.py": '"""\nTODO: Giao diện Streamlit Buổi 09.\nChưa có side effect.\n"""\n',
    "tests/__init__.py": ""
}

for name, content in py_placeholders.items():
    with open(b09_dir / name, "w", encoding="utf-8") as f:
        f.write(content)

# Sample JSON
sample_json = """[]"""
with open(b09_dir / "eval" / "questions.json", "w", encoding="utf-8") as f:
    f.write(sample_json)
with open(b09_dir / "tests" / "fixtures" / "hierarchical_sample.json", "w", encoding="utf-8") as f:
    f.write(sample_json)

# Compile check
print("\n--- COMPILE CHECK ---")
for root, dirs, files in os.walk(b09_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                py_compile.compile(path, doraise=True)
                print(f"Compiled successfully: {os.path.relpath(path, b09_dir)}")
            except Exception as e:
                print(f"Compile ERROR in {os.path.relpath(path, b09_dir)}: {e}")
