"""
Logic RAG (Retrieval-Augmented Generation) cho Buổi 07.
"""

import os
import sys
import json
import math
import hashlib
import argparse
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent
FOUNDATION_DIR = BASE_DIR.parent
BUOI_05_DIR = FOUNDATION_DIR / "buoi_05"
CHUNKS_DIR = BUOI_05_DIR / "output" / "chunks"
STORAGE_DIR = BASE_DIR / "storage"

def load_config() -> dict:
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    
    config = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "").strip(),
        "GEMINI_EMBEDDING_MODEL": os.environ.get("GEMINI_EMBEDDING_MODEL", "").strip(),
        "GEMINI_EMBEDDING_DIM": os.environ.get("GEMINI_EMBEDDING_DIM", "").strip(),
        "GEMINI_GENERATION_MODEL": os.environ.get("GEMINI_GENERATION_MODEL", "").strip(),
        "DEFAULT_TOP_K": os.environ.get("DEFAULT_TOP_K", "").strip(),
        "RAG_MAX_DISTANCE": os.environ.get("RAG_MAX_DISTANCE", "").strip(),
    }
    
    # Validation
    if not config["GEMINI_EMBEDDING_MODEL"]:
        raise ValueError("GEMINI_EMBEDDING_MODEL không được rỗng.")
    if not config["GEMINI_GENERATION_MODEL"]:
        raise ValueError("GEMINI_GENERATION_MODEL không được rỗng.")
        
    try:
        dim = int(config["GEMINI_EMBEDDING_DIM"])
        if not (128 <= dim <= 3072):
            raise ValueError
        config["GEMINI_EMBEDDING_DIM"] = dim
    except (ValueError, TypeError):
        raise ValueError("GEMINI_EMBEDDING_DIM phải là integer trong khoảng 128 đến 3072.")
        
    try:
        top_k = int(config["DEFAULT_TOP_K"])
        if not (1 <= top_k <= 20):
            raise ValueError
        config["DEFAULT_TOP_K"] = top_k
    except (ValueError, TypeError):
        raise ValueError("DEFAULT_TOP_K phải là integer từ 1 đến 20.")
        
    try:
        max_dist = float(config["RAG_MAX_DISTANCE"])
        if max_dist < 0:
            raise ValueError
        config["RAG_MAX_DISTANCE"] = max_dist
    except (ValueError, TypeError):
        raise ValueError("RAG_MAX_DISTANCE phải là float không âm.")
        
    return config

def load_chunks(input_dir: Path, strategy: str = "hierarchical") -> tuple[list[dict], dict]:
    stats = {
        "files_read": 0,
        "total_records": 0,
        "selected_records": 0,
        "empty_text_skipped": 0,
        "valid_chunks": 0,
        "skipped_files": 0
    }
    
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục input: {input_dir}")
        
    json_files = sorted(list(input_dir.glob("*.json")))
    if not json_files:
        raise FileNotFoundError(f"Không có file JSON nào trong: {input_dir}")
        
    valid_chunks = []
    seen_chunk_ids = {}
    
    allowed_strategies = {"fixed-size", "semantic", "hierarchical"}
    if strategy not in allowed_strategies:
        raise ValueError(f"Strategy không hợp lệ: '{strategy}'")

    for file_path in json_files:
        stats["files_read"] += 1
        file_name = file_path.name
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"File {file_name} bị lỗi JSON: {e}")
            
        chunks_list = None
        if isinstance(data, list):
            chunks_list = data
        elif isinstance(data, dict):
            if "chunks" in data and isinstance(data["chunks"], list):
                chunks_list = data["chunks"]
            else:
                stats["skipped_files"] += 1
                continue
        else:
            raise ValueError(f"Sai cấu trúc JSON ở file {file_name}: Root phải là list hoặc object chứa field 'chunks'.")
            
        for i, record in enumerate(chunks_list):
            stats["total_records"] += 1
            if not isinstance(record, dict):
                raise ValueError(f"Record không phải JSON object ở file {file_name}, vị trí {i}.")
                
            rec_strategy = record.get("strategy")
            if rec_strategy != strategy:
                continue
                
            stats["selected_records"] += 1
            
            required_string_fields = ["chunk_id", "strategy", "source", "text"]
            for field in required_string_fields:
                if field not in record:
                    raise ValueError(f"Thiếu field '{field}' ở file {file_name}, vị trí {i}.")
                val = record[field]
                if not isinstance(val, str):
                    raise ValueError(f"Sai kiểu dữ liệu, '{field}' phải là string ở file {file_name}, vị trí {i}.")
                if field != "text" and not val.strip():
                    raise ValueError(f"Field '{field}' không được rỗng ở file {file_name}, vị trí {i}.")
                    
            text_stripped = record["text"].strip()
            if not text_stripped:
                stats["empty_text_skipped"] += 1
                continue
                
            page_start = record.get("page_start")
            page_end = record.get("page_end")
            
            for p_field, p_val in [("page_start", page_start), ("page_end", page_end)]:
                if p_val is None:
                    raise ValueError(f"Thiếu field '{p_field}' ở file {file_name}, vị trí {i}.")
                if isinstance(p_val, bool) or not isinstance(p_val, int):
                    raise ValueError(f"Sai kiểu dữ liệu, '{p_field}' phải là integer (không chấp nhận boolean) ở file {file_name}, vị trí {i}.")
                if p_val < 1:
                    raise ValueError(f"Trang không hợp lệ, '{p_field}' phải >= 1 ở file {file_name}, vị trí {i}.")
                    
            if page_start > page_end:
                raise ValueError(f"Trang không hợp lệ, page_start ({page_start}) > page_end ({page_end}) ở file {file_name}, vị trí {i}.")
                
            chunk_id = record["chunk_id"].strip()
            if chunk_id in seen_chunk_ids:
                prev_file, prev_idx = seen_chunk_ids[chunk_id]
                raise ValueError(f"Duplicate chunk_id: '{chunk_id}'\n- Lần 1: {prev_file}, vị trí {prev_idx}\n- Lần 2: {file_name}, vị trí {i}")
            
            seen_chunk_ids[chunk_id] = (file_name, i)
            
            new_chunk = {k: v for k, v in record.items()}
            new_chunk["text"] = text_stripped
            
            valid_chunks.append(new_chunk)
            stats["valid_chunks"] += 1
            
    return valid_chunks, stats

def generate_embeddings(chunks: list[dict], config: dict) -> list[list[float]]:
    client = genai.Client(api_key=config["GEMINI_API_KEY"])
    model_name = config["GEMINI_EMBEDDING_MODEL"]
    dim = config["GEMINI_EMBEDDING_DIM"]
    
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        title = chunk["source"]
        text = chunk["text"]
        input_text = f"title: {title} | text: {text}"
        
        res = client.models.embed_content(
            model=model_name,
            contents=input_text,
            config=types.EmbedContentConfig(
                output_dimensionality=dim,
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        
        vec = res.embeddings[0].values
        
        if not isinstance(vec, list):
            raise ValueError(f"Embedding không phải list ở chunk_id {chunk['chunk_id']}")
            
        if len(vec) != dim:
            raise ValueError(f"Dimension sai lệch: mong đợi {dim}, nhận {len(vec)} ở chunk_id {chunk['chunk_id']}")
            
        has_non_zero = False
        for val in vec:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Vector chứa giá trị không hợp lệ (không phải số thực) ở chunk_id {chunk['chunk_id']}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Vector chứa NaN hoặc Infinity ở chunk_id {chunk['chunk_id']}")
            if val != 0.0:
                has_non_zero = True
                
        if not has_non_zero:
            raise ValueError(f"Zero vector phát hiện ở chunk_id {chunk['chunk_id']}")
            
        embeddings.append(vec)
        
    if len(embeddings) != len(chunks):
        raise ValueError("Số lượng embeddings không khớp số lượng chunks.")
        
    return embeddings

def get_collection_name(strategy: str, config: dict) -> str:
    model_name = config["GEMINI_EMBEDDING_MODEL"]
    dim = config["GEMINI_EMBEDDING_DIM"]
    h = hashlib.md5(model_name.encode("utf-8")).hexdigest()[:6]
    return f"nhnn-{strategy}-{dim}-{h}"

def cmd_status(strategy: str):
    config = load_config()
    key_status = "Có" if config["GEMINI_API_KEY"] else "Thiếu"
    col_name = get_collection_name(strategy, config)
    
    print(f"--- TRẠNG THÁI HỆ THỐNG ---")
    print(f"- API Key: {key_status}")
    print(f"- Embedding Model: {config['GEMINI_EMBEDDING_MODEL']}")
    print(f"- Dimension: {config['GEMINI_EMBEDDING_DIM']}")
    print(f"- Strategy: {strategy}")
    print(f"- Collection Name: {col_name}")
    
    chroma_path = STORAGE_DIR / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    try:
        col = client.get_collection(name=col_name, embedding_function=None)
        print(f"- Collection Tồn tại: Có")
        print(f"- Số record hiện tại: {col.count()}")
        
        meta = col.metadata
        if meta:
            print("- Metadata Collection:")
            for k, v in meta.items():
                print(f"  + {k}: {v}")
    except Exception:
        print(f"- Collection Tồn tại: Không")

def cmd_index(input_dir: Path, strategy: str, reset: bool):
    config = load_config()
    if not config["GEMINI_API_KEY"]:
        raise ValueError("LỖI: Thiếu GEMINI_API_KEY trong .env. Không thể thực hiện index.")
        
    col_name = get_collection_name(strategy, config)
    
    chunks, stats = load_chunks(input_dir, strategy)
    if not chunks:
        print("Không có chunk hợp lệ để index.")
        return
        
    print("Đang gọi Gemini API để tạo embeddings...")
    embeddings = generate_embeddings(chunks, config)
    print("✓ Đã tạo và xác thực thành công tất cả embeddings.")
    
    chroma_path = STORAGE_DIR / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    if reset:
        try:
            client.delete_collection(name=col_name)
            print(f"✓ Đã xoá collection cũ: {col_name}")
        except Exception:
            pass
            
    expected_meta = {
        "strategy": strategy,
        "embedding_model": config["GEMINI_EMBEDDING_MODEL"],
        "embedding_dim": config["GEMINI_EMBEDDING_DIM"],
        "distance_metric": "cosine",
        "schema_version": "1.0"
    }
    
    try:
        col = client.get_collection(name=col_name, embedding_function=None)
        actual_meta = col.metadata or {}
        mismatch = False
        for k, v in expected_meta.items():
            if str(actual_meta.get(k)) != str(v):
                mismatch = True
                break
        if mismatch:
            raise ValueError(f"Mismatch metadata collection '{col_name}'. Vui lòng chạy lại với --reset.\nCũ: {actual_meta}\nMới: {expected_meta}")
    except Exception as e:
        if "Mismatch metadata" in str(e):
            raise e
        # Support multiple chromadb versions
        try:
            col = client.create_collection(
                name=col_name,
                embedding_function=None,
                metadata=expected_meta,
                configuration={"hnsw": {"space": "cosine"}}
            )
        except TypeError:
            # Fallback if configuration parameter is not supported in this version
            col = client.create_collection(
                name=col_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine", **expected_meta}
            )
            
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        
        meta = {
            "source": chunk["source"],
            "strategy": chunk["strategy"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_id": chunk["chunk_id"],
            "embedding_model": config["GEMINI_EMBEDDING_MODEL"],
            "embedding_dim": config["GEMINI_EMBEDDING_DIM"]
        }
        metadatas.append(meta)
        
    col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✓ Đã upsert thành công {len(ids)} records vào collection {col_name}.")

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Buổi 07 RAG Application")
    subparsers = parser.add_subparsers(dest="command")
    
    val_parser = subparsers.add_parser("validate", help="Load and validate JSON chunks.")
    val_parser.add_argument("--strategy", type=str, default="hierarchical", help="Strategy to load (default: hierarchical)")
    val_parser.add_argument("--input", type=str, help="Input directory (optional, overrides default chunks dir)")
    
    stat_parser = subparsers.add_parser("status", help="Check RAG status.")
    stat_parser.add_argument("--strategy", type=str, default="hierarchical", help="Strategy to check (default: hierarchical)")
    
    idx_parser = subparsers.add_parser("index", help="Create embeddings and index into Chroma.")
    idx_parser.add_argument("--strategy", type=str, default="hierarchical", help="Strategy to index (default: hierarchical)")
    idx_parser.add_argument("--input", type=str, help="Input directory (optional, overrides default chunks dir)")
    idx_parser.add_argument("--reset", action="store_true", help="Xóa collection cũ và index lại từ đầu.")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        try:
            target_dir = Path(args.input) if args.input else CHUNKS_DIR
            chunks, stats = load_chunks(target_dir, args.strategy)
            
            print(f"--- THỐNG KÊ VALIDATE ({args.strategy}) ---")
            for k, v in stats.items():
                print(f"- {k}: {v}")
                
            print(f"\n--- MẪU METADATA (Tối đa 3) ---")
            for i in range(min(3, len(chunks))):
                sample_meta = {k: v for k, v in chunks[i].items() if k != "text"}
                print(f"Mẫu {i+1}: {json.dumps(sample_meta, ensure_ascii=False)}")
                
        except Exception as e:
            print(f"LỖI: {e}")
            sys.exit(1)
            
    elif args.command == "status":
        try:
            cmd_status(args.strategy)
        except Exception as e:
            print(f"LỖI: {e}")
            sys.exit(1)
            
    elif args.command == "index":
        try:
            target_dir = Path(args.input) if args.input else CHUNKS_DIR
            cmd_index(target_dir, args.strategy, args.reset)
        except Exception as e:
            print(f"LỖI: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
