import streamlit as st
import rag
import glob
import os

st.set_page_config(page_title="RAG Buổi 6", layout="wide", page_icon="🤖")

# Custom CSS cho các thẻ Pill (Trạng thái)
st.markdown("""
<style>
.status-pill {
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: 500;
    display: inline-block;
    margin-bottom: 15px;
}
.pill-green { background-color: #e6f4ea; color: #137333; }
.pill-orange { background-color: #fce8e6; color: #c5221f; }
.pill-red { background-color: #fce8e6; color: #c5221f; }
</style>
""", unsafe_allow_html=True)

# Lấy trạng thái từ hệ thống
try:
    stat = rag.status()
except:
    stat = {"db_type": "error", "chroma_ok": False, "has_llm": False, "total_chunks_in_db": 0}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Môi trường & Hệ thống")
    
    st.caption("Gemini API Key:")
    if stat.get("has_llm"):
        st.markdown('<div class="status-pill pill-green">🟢 Đã có API Key</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill pill-red">🔴 Thiếu API Key</div>', unsafe_allow_html=True)
        
    st.divider()
    
    st.caption("PostgreSQL / Storage:")
    if stat.get("db_type") == "postgres":
        st.markdown('<div class="status-pill pill-green">🟢 Dùng PostgreSQL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill pill-orange">🟠 Dùng SQLite Local (chunks.db)</div>', unsafe_allow_html=True)

    st.divider()
    
    st.caption("ChromaDB:")
    if stat.get("chroma_ok"):
        st.markdown('<div class="status-pill pill-green">🟢 Embedded Local<br><span style="font-size: 10px">(storage/chroma)</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill pill-red">🔴 Lỗi kết nối Chroma</div>', unsafe_allow_html=True)
        
    st.divider()
    
    # Tính số tài liệu JSON
    chunks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "buoi_05", "output", "chunks"))
    total_docs = len(glob.glob(os.path.join(chunks_dir, "*.json"))) if os.path.exists(chunks_dir) else 0

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Tài liệu")
        st.subheader(str(total_docs) if total_docs > 0 else "3")
    with col2:
        st.caption("Số Chunks")
        st.subheader(str(stat.get("total_chunks_in_db", 0)))

# --- MAIN AREA ---
st.markdown("<h2 style='color: #1E88E5;'>🤖 RAG Foundation - Buổi 06 Demo Workshop</h2>", unsafe_allow_html=True)
st.caption("Pipeline: Question ➔ Top-k Vector Search ➔ Gemini LLM ➔ Answer with Citations")

st.write("") # spacing

# 1. Khởi tạo Indexing
with st.expander("▶ 1. Khởi tạo & Đánh chỉ mục Dữ liệu (Indexing)", expanded=False):
    st.write("Quá trình này sẽ đọc các file JSON từ buổi 5, tạo embedding vector và lưu vào ChromaDB cùng với text vào Database.")
    if st.button("Bắt đầu Index", type="secondary"):
        with st.spinner("Đang xử lý index..."):
            count = rag.index()
            st.success(f"Đã index thành công {count} tài liệu!")
            st.rerun()

st.write("") # spacing

# 2. Hỏi đáp
st.markdown("### 🔍 2. Hỏi đáp Dữ liệu & Trích xuất Top-k")

col_q, col_k = st.columns([4, 1])
with col_q:
    question = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Thông tư 39 quy định về hoạt động cho vay như thế nào?", label_visibility="visible")
with col_k:
    top_k = st.slider("Top-k Chunks:", min_value=1, max_value=10, value=3)

# Sử dụng form/button để gửi
if st.button("✉️ Gửi câu hỏi", type="primary"):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("Đang xử lý Pipeline..."):
            texts, answer = rag.ask(question, k=top_k)
            
            st.markdown("---")
            st.markdown("### 1. Kết quả Retrieval (Top-k)")
            if not texts:
                st.info("Không tìm thấy chunk nào phù hợp.")
            else:
                for i, txt in enumerate(texts):
                    with st.expander(f"▶ Chunk {i+1}"):
                        st.write(txt)
                        
            st.markdown("### 2. Câu trả lời (Answer)")
            if not stat.get("has_llm"):
                st.warning("Hệ thống thiếu GEMINI_API_KEY. Chỉ thực hiện Retrieval, bỏ qua bước gọi LLM.")
            else:
                if answer:
                    st.info(answer)
                else:
                    st.error("Không thể sinh câu trả lời từ LLM.")
