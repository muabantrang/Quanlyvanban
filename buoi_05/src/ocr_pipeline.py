import os
import sys
import glob
import json
import argparse
import asyncio
import unicodedata
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv

# For LlamaParse
from llama_cloud import AsyncLlamaCloud

load_dotenv()

def is_text_garbled_or_empty(text):
    """
    Kiểm tra 3 tình huống lỗi:
    1. Lỗi rỗng: Văn bản quá ngắn hoặc rỗng.
    2. Lỗi ký tự lạ/encoding: Chứa ký tự thay thế () hoặc lỗi cid:.
    3. Lỗi font: Tỷ lệ ký tự alphanumeric quá thấp.
    """
    if not text or len(text.strip()) < 50:
        return True, "Văn bản rỗng hoặc quá ngắn (Tình huống lỗi 1)"
    
    if "" in text or "cid:" in text:
        return True, "Chứa ký tự lạ hoặc lỗi encoding (Tình huống lỗi 2)"
    
    alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
    if alnum_count / len(text) < 0.7:
        return True, "Tỷ lệ ký tự không xác định cao do lỗi font (Tình huống lỗi 3)"
        
    return False, ""

def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        try:
            full_text += page.get_text() + "\n"
        except Exception as e:
            print(f"    [CẢNH BÁO] Lỗi đọc trang {page.number}, bỏ qua trang này: {e}")
            continue
    return full_text

async def extract_text_llamaparse(pdf_path):
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key or api_key == "KEY CỦA BẠN":
        raise ValueError("Chưa cấu hình LLAMA_CLOUD_API_KEY hợp lệ trong .env")
    
    client = AsyncLlamaCloud(api_key=api_key)
    file_obj = await client.files.create(file=pdf_path, purpose="parse")
    
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        expand=["markdown_full"],
    )
    return result.markdown_full

def chunk_fixed_size(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_semantic(text):
    # Ưu tiên ranh giới đoạn văn (hết đoạn, cách dòng)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks

def chunk_hierarchical(text):
    # Tìm kiếm các mẫu Chương, Mục, Điều, Khoản, Điểm ở đầu dòng
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    
    pattern = re.compile(r'^(Chương|Mục|Điều|Khoản|Điểm)\s+[\dIVX]+', re.IGNORECASE)
    found_structure = False
    
    for line in lines:
        if pattern.match(line.strip()):
            found_structure = True
            if current_chunk:
                chunks.append('\n'.join(current_chunk).strip())
                current_chunk = []
        current_chunk.append(line)
        
    if current_chunk:
        chunks.append('\n'.join(current_chunk).strip())
        
    if not found_structure:
        pass # Cảnh báo đã được chuyển lên giao diện UI (app.py)
    
    return [c for c in chunks if c.strip()]

def get_stats(chunks):
    if not chunks:
        return 0, 0, 0, 0
    lengths = [len(c) for c in chunks]
    return len(chunks), min(lengths), max(lengths), sum(lengths)/len(lengths)

async def process_pdf(pdf_path, write_mode):
    print(f"\n--- Đang xử lý: {pdf_path} ---")
    source_name = os.path.basename(pdf_path)
    
    # Bước 1 & 2: Thử PyMuPDF và kiểm tra text layer
    print("  1. Thử trích xuất bằng PyMuPDF...")
    try:
        raw_text = extract_text_pymupdf(pdf_path)
        is_bad, reason = is_text_garbled_or_empty(raw_text)
        ocr_used = "PyMuPDF"
    except Exception as e:
        is_bad = True
        reason = f"Lỗi thư viện: {e}"
        raw_text = ""
        ocr_used = "PyMuPDF_Failed"

    # Bước 3: Fallback sang LlamaParse nếu cần OCR toàn bộ
    if is_bad:
        print(f"  PyMuPDF thất bại: {reason}")
        print("  => Chuyển sang LlamaParse (OCR toàn bộ)...")
        try:
            raw_text = await extract_text_llamaparse(pdf_path)
            ocr_used = "LlamaParse"
        except Exception as e:
            print(f"  [LỖI] LlamaParse thất bại: {e}")
            # Dừng xử lý file này, không làm crash toàn bộ job
            return
    else:
        print("  => PyMuPDF trích xuất thành công, bỏ qua LlamaParse (Tránh OCR không cần thiết).")

    # Bước 4: Chuẩn hóa Unicode NFC
    raw_text = unicodedata.normalize('NFC', raw_text)
    
    # Bước 6: Thử nghiệm chiến thuật Chunking
    chunks_fixed = chunk_fixed_size(raw_text)
    chunks_semantic = chunk_semantic(raw_text)
    chunks_hierarchical = chunk_hierarchical(raw_text)
    
    # Khởi tạo Metadata mẫu (Mô phỏng page=1 vì text đã bị gộp cho mức demo)
    demo_metadata = {
        "chunk_id": f"{source_name}_fixed_0",
        "strategy": "fixed-size",
        "source": source_name,
        "page_start": 1,
        "page_end": 1,
        "language": "vi",
        "ocr_used": ocr_used,
        "text": chunks_fixed[0][:150] + "..." if chunks_fixed else ""
    }

    # Bước 5: Lưu dữ liệu hoặc in ra terminal
    if write_mode:
        os.makedirs("output/raw", exist_ok=True)
        os.makedirs("output/chunks", exist_ok=True)
        
        with open(f"output/raw/{source_name}.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)
            
        with open(f"output/chunks/{source_name}_sample_meta.json", "w", encoding="utf-8") as f:
            json.dump(demo_metadata, f, ensure_ascii=False, indent=4)
            
        print("  [WRITE] Đã lưu file text gốc và metadata mẫu vào output/")
    else:
        print("  [DRY-RUN] Ví dụ metadata:", json.dumps(demo_metadata, ensure_ascii=False))

    # In thống kê
    print(f"  [THỐNG KÊ] Kết quả Chunking:")
    for name, chunks in [("Fixed-size", chunks_fixed), ("Semantic", chunks_semantic), ("Hierarchical", chunks_hierarchical)]:
        count, min_l, max_l, avg_l = get_stats(chunks)
        print(f"    - {name:<12}: Số chunk={count:<4}, Độ dài (min/max/avg)={min_l}/{max_l}/{avg_l:.1f}")

async def main():
    parser = argparse.ArgumentParser(description="OCR và Chunking PDF tiếng Việt")
    parser.add_argument("--write", action="store_true", help="Ghi kết quả ra thư mục output/")
    args = parser.parse_args()
    
    pdf_files = glob.glob("datademo/*.pdf")
    if not pdf_files:
        print("Không tìm thấy file PDF nào trong datademo/")
        return
        
    for pdf in pdf_files:
        await process_pdf(pdf, args.write)

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
