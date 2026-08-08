import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def check_imports():
    print("--- PACKAGE IMPORT CHECK ---")
    packages = {
        "streamlit": "streamlit",
        "google-genai": "google.genai",
        "chromadb": "chromadb",
        "psycopg": "psycopg",
        "python-dotenv": "dotenv"
    }
    
    all_passed = True
    for pkg_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"[PASS] {pkg_name} (imported as {import_name})")
        except ImportError as e:
            print(f"[FAIL] {pkg_name} - Error: {e}")
            all_passed = False
    return all_passed

def check_chroma():
    print("\n--- CHROMADB SETUP ---")
    import chromadb
    try:
        # Try HTTP Client first
        client = chromadb.HttpClient()
        # Just a ping to see if server is alive
        client.heartbeat()
        print("[INFO] Found Chroma Server running. Using HttpClient.")
    except Exception as e:
        print("[INFO] Chroma Server not found or not reachable. Falling back to Embedded Persistent Client.")
        os.makedirs("storage/chroma", exist_ok=True)
        try:
            client = chromadb.PersistentClient(path="storage/chroma")
            print("[PASS] Successfully initialized Chroma PersistentClient at storage/chroma/")
        except Exception as e2:
            print(f"[FAIL] Failed to initialize Chroma PersistentClient: {e2}")

def setup_postgres():
    print("\n--- POSTGRESQL SETUP ---")
    import psycopg
    from psycopg.errors import DuplicateDatabase
    from dotenv import load_dotenv
    
    load_dotenv()
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    dbname = os.getenv("POSTGRES_DB", "rag_db")
    
    # 1. Test connection to default 'postgres' database
    conn_str_postgres = f"host={host} port={port} user={user} password={password} dbname=postgres"
    try:
        conn = psycopg.connect(conn_str_postgres, autocommit=True)
        print("[PASS] Connected to PostgreSQL server (database: postgres).")
    except psycopg.OperationalError as e:
        print("[FAIL] Could not connect to PostgreSQL server.")
        print(f"Error details: {e}")
        print("\n=== HƯỚNG DẪN CÀI ĐẶT POSTGRESQL ===")
        print("Nếu bạn chưa cài PostgreSQL:")
        print("1. Tải PostgreSQL từ trang chủ: https://www.postgresql.org/download/")
        print("2. Cài đặt và ghi nhớ mật khẩu bạn đặt cho user 'postgres'.")
        print("3. Điền mật khẩu đó vào POSTGRES_PASSWORD trong file .env")
        print("Nếu bạn đã cài, hãy kiểm tra lại PostgreSQL service đang chạy và password trong .env đã đúng.")
        print("====================================")
        return
        
    # 2. Check if rag_db exists and create if not
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cur.fetchone()
            if not exists:
                print(f"[INFO] Database '{dbname}' does not exist. Creating it...")
                cur.execute(f"CREATE DATABASE {dbname};")
                print(f"[PASS] Created database '{dbname}'.")
            else:
                print(f"[INFO] Database '{dbname}' already exists.")
    except Exception as e:
        print(f"[FAIL] Error checking/creating database: {e}")
    finally:
        conn.close()
        
    # 3. Test connection to rag_db
    conn_str_rag = f"host={host} port={port} user={user} password={password} dbname={dbname}"
    try:
        conn_rag = psycopg.connect(conn_str_rag)
        print(f"[PASS] Successfully connected to database '{dbname}'.")
        conn_rag.close()
    except psycopg.OperationalError as e:
        print(f"[FAIL] Could not connect to database '{dbname}': {e}")

if __name__ == "__main__":
    if check_imports():
        check_chroma()
        setup_postgres()
    else:
        print("\n[ERROR] Some packages failed to import. Please check your pip install.")
