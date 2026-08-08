import streamlit as st
import rag

# Define constants
STRATEGIES = ["hierarchical", "semantic", "fixed-size"]
CHUNKS_DIR = rag.CHUNKS_DIR

# Initial configuration for page
st.set_page_config(page_title="RAG Foundation - Buổi 07", layout="wide")

st.title("RAG Foundation - Truy Vấn Dữ Liệu")
st.markdown("Hệ thống Retrieval-Augmented Generation với ChromaDB và Google Gemini.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Cấu hình & Trạng thái")
    
    strategy = st.selectbox("Chọn Strategy", STRATEGIES, index=0)
    top_k = st.slider("Top-K (Số lượng evidence)", min_value=1, max_value=10, value=5)
    
    # Fetch status for the selected strategy
    status_info = rag.get_status(strategy)
    
    st.subheader("Thông tin hệ thống")
    st.write(f"**API Key:** {'Có' if status_info['api_key_present'] else 'Thiếu (Vui lòng điền vào .env)'}")
    st.write(f"**Embedding Model:** {status_info['embedding_model']}")
    st.write(f"**Dimension:** {status_info['embedding_dim']}")
    st.write(f"**Generation Model:** {status_info['generation_model']}")
    st.write(f"**Ngưỡng RAG_MAX_DISTANCE:** {status_info['max_distance']}")
    
    st.subheader("Trạng thái Collection")
    st.write(f"**Tên:** {status_info['collection_name']}")
    st.write(f"**Tồn tại:** {'Có' if status_info['collection_exists'] else 'Chưa tạo'}")
    st.write(f"**Số lượng chunk (records):** {status_info['record_count']}")


# --- INDEX AREA ---
st.header("1. Cập nhật Dữ liệu (Indexing)")

reset_col = st.checkbox("Reset collection trước khi index", value=False)
if st.button("Index dữ liệu"):
    if not status_info['api_key_present']:
        st.error("LỖI: Thiếu API Key. Vui lòng cấu hình `GEMINI_API_KEY` trong file `.env`.")
    else:
        with st.spinner(f"Đang index dữ liệu vào bộ sưu tập `{status_info['collection_name']}`..."):
            try:
                res = rag.do_index(CHUNKS_DIR, strategy, reset_col)
                st.success(f"✓ Index thành công!")
                
                st.write("**Kết quả:**")
                st.write(f"- Strategy: `{res['strategy']}`")
                st.write(f"- Tên Collection: `{res['collection_name']}`")
                st.write(f"- Số chunk ban đầu: `{res['old_count']}`")
                st.write(f"- Số chunk sau khi index: `{res['new_count']}` (Đã upsert thêm {res['upserted']} records)")
                
                stats = res['stats']
                if stats['empty_text_skipped'] > 0:
                    st.warning(f"Đã bỏ qua {stats['empty_text_skipped']} records do text rỗng.")
                if stats['skipped_files'] > 0:
                    st.info(f"Đã bỏ qua {stats['skipped_files']} file vì không phải mảng chunks hợp lệ.")
                
                # Cập nhật lại thanh trạng thái
                status_info = rag.get_status(strategy)
            except Exception as e:
                st.error(f"LỖI INDEX: {str(e)}")


# --- QUESTION AREA ---
st.header("2. Truy vấn (Query)")

question = st.text_area("Nhập câu hỏi của bạn:", height=100)
can_query = True
why_cannot_query = []

if not status_info['api_key_present']:
    can_query = False
    why_cannot_query.append("Thiếu API Key.")
if not status_info['collection_exists']:
    can_query = False
    why_cannot_query.append("Collection chưa tồn tại. Hãy Index dữ liệu trước.")
elif status_info['record_count'] == 0:
    can_query = False
    why_cannot_query.append("Collection trống. Hãy Index dữ liệu trước.")
if not question.strip():
    can_query = False
    why_cannot_query.append("Vui lòng nhập câu hỏi.")

if st.button("Gửi câu hỏi", disabled=not can_query):
    with st.spinner("Đang tìm kiếm thông tin và tạo câu trả lời..."):
        try:
            result = rag.do_query(question, top_k, strategy)
            st.session_state['last_query_result'] = result
        except Exception as e:
            st.error(f"LỖI TRUY VẤN: {str(e)}")

# Display error reasons if cannot query and user tried to type
if not can_query and len(why_cannot_query) > 0 and question.strip():
    for reason in why_cannot_query:
        st.warning(reason)

# --- RENDER ANSWER & EVIDENCE ---
if 'last_query_result' in st.session_state:
    res = st.session_state['last_query_result']
    
    st.markdown("---")
    st.subheader("CÂU TRẢ LỜI")
    
    status = res['status']
    
    if status == "answered":
        st.success(res['answer'])
    elif status == "insufficient_evidence":
        st.warning(res['answer'])
    elif status == "retrieval_only":
        st.info(res['answer'])
        
    if res.get('citations'):
        st.write("**Trích dẫn đã sử dụng:**")
        for cit in res['citations']:
            st.markdown(f"- **{cit['evidence_id']}**: {cit['display']}")
            
    if res.get('warnings'):
        for w in res['warnings']:
            st.warning(w)
            
    st.markdown("---")
    st.subheader("Nguồn tham khảo")
    
    evidences = res.get('evidence', [])
    if not evidences:
        st.write("Chưa có evidence nào được tìm thấy.")
    else:
        for ev in evidences:
            ps = ev["page_start"]
            pe = ev["page_end"]
            page_str = f"tr. {ps}" if ps == pe else f"tr. {ps}-{pe}"
            
            title = f"{ev['source']} – {page_str} – {ev['chunk_id']}"
            
            with st.expander(title):
                st.write(f"**Evidence ID:** {ev['evidence_id']}")
                st.write(f"**Nguồn:** {ev['source']} | **Trang:** {page_str}")
                st.write(f"**Chunk ID:** {ev['chunk_id']}")
                
                st.write(f"**Distance:** {ev['distance']:.4f} *(khoảng cách thấp hơn thường liên quan chặt chẽ hơn)*")
                
                if ev['accepted']:
                    st.success("Trạng thái: **ĐẠT** ngưỡng độ tin cậy (Được đưa vào context)")
                else:
                    st.error("Trạng thái: **KHÔNG ĐẠT** ngưỡng độ tin cậy (Bị loại khỏi context)")
                    
                st.write("**Nội dung:**")
                st.info(ev['text'])
