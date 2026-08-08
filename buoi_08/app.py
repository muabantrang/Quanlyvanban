import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import sys

# Add parent to path for imports
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import advanced_rag
from rag import CHUNKS_DIR

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Advanced RAG Pipeline", layout="wide")

# CSS Injection for Agribank theme
st.markdown("""
<style>
    /* Primary Color for Buttons */
    div.stButton > button:first-child {
        background-color: #A31720;
        color: white;
        border: None;
        border-radius: 4px;
    }
    div.stButton > button:first-child:hover {
        background-color: #8a131a;
        color: white;
    }
    
    /* Headers and Tabs */
    h1, h2, h3 {
        color: #1A365D; /* Agribank often uses dark blue for headers too, but let's stick to standard */
    }
    h1 { color: #1A365D !important; }
    
    /* Tab color adjustments */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        color: #A31720 !important;
        border-bottom-color: #A31720 !important;
    }
</style>
""", unsafe_allow_html=True)

# State
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_compare" not in st.session_state:
    st.session_state.last_compare = None

@st.cache_data
def get_config():
    return advanced_rag.load_config()

@st.cache_data
def get_chunks(strategy: str):
    return advanced_rag.load_chunks(CHUNKS_DIR, strategy)

config = get_config()

# Sidebar
st.sidebar.image("https://cdn.haitrieu.com/wp-content/uploads/2022/01/Logo-Agribank-V.png", use_container_width=True)
st.sidebar.markdown("<h4 style='text-align: center; color: #A31720; margin-top: -10px;'>Ban Kiểm tra, giám sát nội bộ</h4>", unsafe_allow_html=True)
st.sidebar.title("⚙️ Cấu hình & Trạng thái")

strategy = st.sidebar.selectbox("Chiến lược Chunking (Strategy)", ["hierarchical", "recursive", "semantic"], index=0)
mode = st.sidebar.selectbox("Retrieval Mode Mặc định", ["hybrid_rerank", "bm25", "semantic", "hybrid"], index=0)

final_top_k = st.sidebar.number_input("Final Top-K Evidences", min_value=1, max_value=20, value=int(config.get('FINAL_TOP_K', 5)))

with st.sidebar.expander("📌 Trạng thái Hệ thống (Read-only)", expanded=True):
    status = advanced_rag.advanced_status(strategy, config)
    st.markdown(f"**Corpus Size:** {status['corpus_size']} chunks")
    
    bm25_icon = "✅ Sẵn sàng" if status['corpus_size'] > 0 else "❌ Chưa có"
    st.markdown(f"**BM25 Ready:** {bm25_icon}")
    
    st.markdown(f"**Semantic Index:** ✅ {status['collection_count']} recs")
    
    st.markdown(f"**Reranker Model:** {status['reranker_model']}")
    cache_icon = "✅ Đã tải" if status['reranker_cache_exists'] else "⚠️ Chưa cache (Tải khi dùng)"
    st.markdown(f"**Reranker Cache:** {cache_icon}")
    
    api_icon = "✅ Đã cấu hình" if config.get('GEMINI_API_KEY') else "❌ Thiếu"
    st.markdown(f"**Gemini API Key:** {api_icon}")


try:
    chunks, _ = get_chunks(strategy)
except Exception as e:
    st.error(f"Lỗi tải chunks: {e}")
    chunks = []

# Main Area
st.title("🔍 Advanced Hybrid RAG Engine")
st.markdown("Hệ thống RAG nâng cao kết hợp BM25 Keyword Search, Gemini Vector Retrieval, RRF Fusion & Cross-Encoder Reranking")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Hỏi đáp Advanced RAG", 
    "📊 So sánh Retrieval (4 Modes)", 
    "🔀 Pipeline Trace", 
    "📈 Báo cáo Đánh giá"
])

with tab1:
    st.header("💬 Hỏi đáp & Trích dẫn Nguồn dữ liệu (Grounding)")
    question = st.text_input("Nhập câu hỏi cần tra cứu:", placeholder="Điều kiện để tổ chức tín dụng cơ cấu lại thời hạn trả nợ gốc và lãi vay theo Thông tư 02 là gì?")
    
    if st.button("🔍 Truy vấn RAG", type="primary"):
        if not question:
            st.warning("Vui lòng nhập câu hỏi.")
        elif not chunks:
            st.error("Chưa tải được chunks.")
        else:
            with st.spinner("⏳ Đang thực hiện quy trình Advanced RAG (Retrieval + Rerank + Grounding)..."):
                try:
                    current_config = config.copy()
                    current_config['FINAL_TOP_K'] = final_top_k
                    
                    ans = advanced_rag.generate_advanced_answer(question, mode, strategy, chunks, current_config)
                    st.session_state.last_answer = ans
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    
    ans = st.session_state.last_answer
    if ans:
        status_text = ans["status"]
        if status_text == "answered":
            st.success("Tạo câu trả lời thành công.")
        elif status_text == "insufficient_evidence":
            st.warning("Không có văn bản nào thỏa mãn điều kiện chất lượng (ngưỡng cosine/rerank).")
        elif status_text == "retrieval_only":
            st.warning("Có lỗi trong quá trình tạo câu trả lời (retrieval_only).")
        elif status_text == "reranker_unavailable":
            st.error("Mô hình Reranker không khả dụng. Hãy chắc chắn máy tính có internet hoặc đã tải cache model, và cấu hình đúng (RERANK_DEVICE).")
            
        st.markdown(f"**Câu hỏi:** {ans['question']}")
        if ans["answer"]:
            st.markdown("### Câu trả lời")
            st.info(ans["answer"])
            
        if ans["citations"]:
            st.markdown("### Trích dẫn")
            for c in ans["citations"]:
                st.markdown(f"- **{c['label']}**: `{c['chunk_id']}` (Nguồn: {c['source']}, Trang: {c['page_start']}-{c['page_end']})")
                
        if ans["warnings"]:
            st.markdown("### Cảnh báo")
            for w in ans["warnings"]:
                st.error(w)
                
        st.markdown("### Bằng chứng (Evidence)")
        for c in ans["evidence"]:
            with st.expander(f"Chunk: {c['chunk_id']} | Accepted: {'✅' if c['accepted'] else '❌'}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("BM25 Score", f"{c['bm25_score']:.3f}" if c['bm25_score'] is not None else "N/A")
                c2.metric("Sem Distance", f"{c['semantic_distance']:.3f}" if c['semantic_distance'] is not None else "N/A")
                c3.metric("RRF Score", f"{c['rrf_score']:.4f}" if c['rrf_score'] is not None else "N/A")
                
                rr_score_str = f"{c['rerank_score']:.3f}" if c['rerank_score'] is not None else "N/A"
                chg_str = f"{c['rank_change']:+d}" if c['rank_change'] else ""
                c4.metric("Rerank Score", rr_score_str, chg_str)
                
                st.markdown("**Văn bản:**")
                st.write(c["text"])

with tab2:
    st.header("So sánh Retrieval")
    cmp_question = st.text_input("Nhập câu hỏi so sánh:", key="cmp_q")
    
    if st.button("Compare", type="primary"):
        if not cmp_question:
            st.warning("Vui lòng nhập câu hỏi.")
        elif not chunks:
            st.error("Chưa tải được chunks.")
        else:
            with st.spinner("Đang chạy 4 chế độ retrieval..."):
                try:
                    res = advanced_rag.compare_retrieval(cmp_question, chunks, strategy, config)
                    st.session_state.last_compare = res
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    
    res = st.session_state.last_compare
    if res:
        # Build DataFrame
        all_cids = set()
        for k in res.keys():
            if "evidence" in res[k]:
                for c in res[k]["evidence"]:
                    all_cids.add(c["chunk_id"])
                
        data = []
        for cid in all_cids:
            row = {"chunk_id": cid}
            def get_c(mode_res):
                if "evidence" not in mode_res:
                    return None, None
                for idx, c in enumerate(mode_res["evidence"]):
                    if c["chunk_id"] == cid:
                        return c, idx + 1
                return None, None
            
            c_bm25, r_bm25 = get_c(res.get("bm25", {}))
            c_sem, r_sem = get_c(res.get("semantic", {}))
            c_hyb, r_hyb = get_c(res.get("hybrid", {}))
            c_rr, r_rr = get_c(res.get("hybrid_rerank", {}))
            
            row["bm25_rank"] = r_bm25
            row["semantic_rank"] = r_sem
            row["fused_rank"] = r_hyb
            row["rerank_rank"] = r_rr
            
            row["rank_change"] = c_rr["rank_change"] if c_rr and c_rr.get("rank_change") is not None else None
            
            modes = []
            if c_bm25: modes.append("bm25")
            if c_sem: modes.append("sem")
            if c_hyb: modes.append("hyb")
            if c_rr: modes.append("rr")
            row["final modes"] = ", ".join(modes)
            
            data.append(row)
            
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("### Top 5 Preview")
        cols = st.columns(4)
        for i, k in enumerate(["bm25", "semantic", "hybrid", "hybrid_rerank"]):
            with cols[i]:
                st.subheader(k)
                if k in res and "evidence" in res[k]:
                    for c in res[k]["evidence"][:5]:
                        acc_str = "✅" if c['accepted'] else "❌"
                        st.markdown(f"**{c['chunk_id']}** {acc_str}")
                        st.caption(c["text"][:100] + "...")

with tab3:
    st.header("Pipeline Trace")
    ans = st.session_state.last_answer
    if ans and "trace" in ans:
        trace = ans["trace"]
        st.markdown("### Lực lượng Ứng viên (Candidate Flow)")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("BM25", trace.get("bm25_candidates", 0))
        c2.metric("Semantic", trace.get("semantic_candidates", 0))
        c3.metric("Union/Overlap", f"{trace.get('union', 0)} / {trace.get('overlap', 0)}")
        c4.metric("Reranked", trace.get("reranked", 0))
        c5.metric("Accepted", trace.get("accepted", 0))
        
        st.markdown("### Độ trễ (Latency - ms)")
        lat = trace.get("latency_ms", {})
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        l1.metric("BM25", f"{lat.get('bm25', 0):.1f}")
        l2.metric("Semantic", f"{lat.get('semantic', 0):.1f}")
        l3.metric("Fusion", f"{lat.get('fusion', 0):.1f}")
        l4.metric("Rerank", f"{lat.get('rerank', 0):.1f}")
        l5.metric("Generation", f"{lat.get('generation', 0):.1f}")
        l6.metric("Total", f"{lat.get('total', 0):.1f}")
        
        st.info(
            "**Chú thích Đo lường:**\n"
            "- **BM25 Score:** Càng cao càng tốt.\n"
            "- **Cosine Distance:** Càng thấp càng tốt.\n"
            "- **RRF/Rerank Score:** Càng cao càng tốt.\n"
            "- **Rerank Score:** Điểm đã chuẩn hoá Sigmoid, không phải xác suất đúng tuyệt đối."
        )
        
    else:
        st.info("Hãy thực hiện một truy vấn ở Tab 1 để xem Trace.")

with tab4:
    st.header("Đánh giá (Evaluation)")
    report_file = BASE_DIR / "reports" / "report.json"
    eval_file = BASE_DIR / "eval" / "report.json"
    
    target_file = None
    if report_file.exists():
        target_file = report_file
    elif eval_file.exists():
        target_file = eval_file
        
    if target_file:
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                rep = json.load(f)
                
            st.success("Đã tìm thấy báo cáo đánh giá.")
            
            if "metrics" in rep:
                st.dataframe(pd.DataFrame(rep["metrics"]).T)
            else:
                st.json(rep)
                
            if rep.get("needs_human_review", False):
                st.warning("Cảnh báo: Dữ liệu Ground Truth cần con người kiểm tra (needs_human_review = true)")
                
        except Exception as e:
            st.error(f"Lỗi đọc report: {e}")
    else:
        st.info("Chưa có report hợp lệ. Vui lòng chạy công cụ Đánh giá (`evaluate.py`) trước.")
