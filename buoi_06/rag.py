import os
import json
import glob
import sqlite3
import psycopg
import chromadb
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types

load_dotenv()

# --- DATABASE FALLBACK SETUP ---
DB_TYPE = "postgres"
db_conn = None

def get_db_connection():
    global DB_TYPE, db_conn
    if db_conn is not None:
        return db_conn

    # 1. Try PostgreSQL
    try:
        db_conn = psycopg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("POSTGRES_DB", "rag_db"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            autocommit=True
        )
        DB_TYPE = "postgres"
    except Exception:
        # 2. Fallback to SQLite local file
        os.makedirs("storage", exist_ok=True)
        db_conn = sqlite3.connect("storage/rag_fallback.db", check_same_thread=False)
        DB_TYPE = "sqlite"
    
    # Initialize Table
    if DB_TYPE == "postgres":
        with db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id VARCHAR PRIMARY KEY,
                    text TEXT
                );
            """)
    else:
        cursor = db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT
            );
        """)
        db_conn.commit()
        
    return db_conn

# --- CHROMA DB SETUP ---
def get_chroma_collection():
    os.makedirs("storage/chroma", exist_ok=True)
    client = chromadb.PersistentClient(path="storage/chroma")
    return client.get_or_create_collection(name="rag_chunks")

# --- GEMINI SETUP ---
HAS_LLM = False
gemini_client = None

api_key = os.getenv("GEMINI_API_KEY")
if api_key and api_key.strip():
    try:
        gemini_client = genai.Client(api_key=api_key)
        HAS_LLM = True
    except Exception:
        HAS_LLM = False

# --- LOGIC FUNCTIONS ---

def get_embedding(text: str) -> list[float]:
    if not HAS_LLM:
        return [0.0] * 384
        
    response = gemini_client.models.embed_content(
        model='gemini-embedding-2',
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=384)
    )
    return response.embeddings[0].values

def index():
    conn = get_db_connection()
    collection = get_chroma_collection()
    
    chunks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "buoi_05", "output", "chunks"))
    json_files = glob.glob(os.path.join(chunks_dir, "*.json"))
    
    for i, file_path in enumerate(json_files):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        chunk_id = data.get("chunk_id", f"chunk_{i}")
        text = data.get("text", "")
        
        # 1. Lưu text vào DB
        if DB_TYPE == "postgres":
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunks (chunk_id, text) VALUES (%s, %s) ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text",
                    (chunk_id, text)
                )
        else:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chunks (chunk_id, text) VALUES (?, ?) ON CONFLICT(chunk_id) DO UPDATE SET text=excluded.text",
                (chunk_id, text)
            )
            conn.commit()

        # 2. Embedding
        vector = get_embedding(text)
        
        # 3. Lưu vào ChromaDB
        collection.upsert(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[text]
        )
    return len(json_files)

def ask(question: str, k: int = 3):
    collection = get_chroma_collection()
    conn = get_db_connection()
    
    # 1. Embed câu hỏi
    query_vector = get_embedding(question)
    
    # 2. Tìm top-k trên Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )
    
    chunk_ids = results['ids'][0] if results['ids'] else []
    if not chunk_ids:
        return ([], "Không tìm thấy tài liệu phù hợp.")
        
    # 3. Lấy text tương ứng từ PostgreSQL hoặc SQLite
    texts = []
    if DB_TYPE == "postgres":
        with conn.cursor() as cur:
            for cid in chunk_ids:
                cur.execute("SELECT text FROM chunks WHERE chunk_id = %s", (cid,))
                row = cur.fetchone()
                if row:
                    texts.append(row[0])
    else:
        cursor = conn.cursor()
        for cid in chunk_ids:
            cursor.execute("SELECT text FROM chunks WHERE chunk_id = ?", (cid,))
            row = cursor.fetchone()
            if row:
                texts.append(row[0])

    # Trả về ngay nếu không có LLM
    if not HAS_LLM:
        return (texts, None)

    # 4. Gửi cho Gemini
    context_str = "\n\n".join(texts)
    prompt = f"""Bạn là một chuyên gia về văn bản pháp luật ngân hàng. 
Nhiệm vụ của bạn là trả lời câu hỏi CHỈ DỰA VÀO phần "Ngữ cảnh" được cung cấp.
Không được thêm thắt thông tin bên ngoài. Nếu Ngữ cảnh không chứa đủ thông tin để trả lời, hãy nói: "Văn bản hiện tại không đề cập đến vấn đề này".

Ngữ cảnh:
{context_str}

Câu hỏi: {question}
"""

    response = gemini_client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=prompt
    )
    
    return (texts, response.text)

def status():
    collection = get_chroma_collection()
    conn = get_db_connection()
    
    chroma_count = collection.count()
    db_count = 0
    if DB_TYPE == "postgres":
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            db_count = cur.fetchone()[0]
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        db_count = cursor.fetchone()[0]
        
    return {
        "db_type": DB_TYPE,
        "chroma_ok": chroma_count >= 0,
        "total_chunks_in_db": db_count,
        "total_vectors_in_chroma": chroma_count,
        "has_llm": HAS_LLM
    }
