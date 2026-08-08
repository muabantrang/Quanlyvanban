import streamlit as st
import os
import glob
import sys

# Đảm bảo có thể import từ src
sys.path.append(os.path.dirname(__file__))
from src.ocr_pipeline import chunk_fixed_size, chunk_semantic, chunk_hierarchical

st.set_page_config(page_title="Trực quan hoá Chunking RAG", layout="wide")

st.title("🔍 Trực quan hoá Phân chia văn bản (Chunking)")
st.markdown("Dữ liệu đầu vào được lấy từ thư mục `output/raw` sau khi đã chạy OCR/PyMuPDF.")

raw_dir = "output/raw"
if not os.path.exists(raw_dir):
    st.warning("Thư mục `output/raw` chưa tồn tại. Vui lòng chạy lệnh: `python src/ocr_pipeline.py --write` trên Terminal trước.")
    st.stop()

txt_files = glob.glob(f"{raw_dir}/*.txt")
if not txt_files:
    st.warning("Không tìm thấy file text nào trong `output/raw`. Hãy chạy pipeline `--write` trước.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Cài đặt")
    selected_file = st.selectbox("1. Chọn tài liệu:", [os.path.basename(f) for f in txt_files])
    file_path = os.path.join(raw_dir, selected_file)
    
    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()
        
    strategy = st.radio(
        "2. Chọn chiến lược Chunking:", 
        [
            "Fixed-size (Kích thước cố định)", 
            "Semantic (Ranh giới đoạn văn)", 
            "Hierarchical (Cấu trúc văn bản)"
        ]
    )
    
    if strategy.startswith("Fixed-size"):
        chunk_size = st.number_input("Kích thước chunk (ký tự)", value=500, step=50)
        overlap_size = st.number_input("Kích thước overlap", value=50, step=10)
        chunks = chunk_fixed_size(text_content, chunk_size, overlap_size)
    elif strategy.startswith("Semantic"):
        chunks = chunk_semantic(text_content)
    else:
        chunks = chunk_hierarchical(text_content)
        if len(chunks) == 1:
            st.warning("CẢNH BÁO: Không tìm thấy cấu trúc phân tầng (Chương/Mục/Điều) trong văn bản. Đã tự động gộp thành 1 chunk lớn.")
        
    st.success(f"Đã chia thành **{len(chunks)} chunks**.")

with col2:
    st.subheader("Danh sách Chunks")
    
    if not chunks:
        st.info("Không có chunk nào được tạo ra.")
    else:
        # Giới hạn hiển thị 100 chunk đầu tiên nếu quá nhiều để tránh lag UI
        display_limit = min(len(chunks), 100)
        if len(chunks) > 100:
            st.warning(f"Văn bản quá dài. Đang hiển thị 100 chunk đầu tiên trên tổng số {len(chunks)} chunks.")
            
        for i in range(display_limit):
            with st.expander(f"Chunk {i+1} ({len(chunks[i])} ký tự)"):
                st.text(chunks[i])
