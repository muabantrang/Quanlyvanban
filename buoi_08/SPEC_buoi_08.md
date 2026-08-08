# Đặc tả Kỹ thuật: Buổi 08 - Advanced RAG

## 1. Workspace và Security
- **Workspace**: Mọi file thực thi của Buổi 08 nằm trong thư mục `rag_foundation/buoi_08/`.
- **Security**: Không lưu trực tiếp API keys vào mã nguồn. Phải sử dụng file `.env` (không commit lên Git) và biến môi trường. Chặn các chuỗi Prompt Injection độc hại trước khi đưa vào pipeline.

## 2. Quan hệ với Buổi 05 và Buổi 07
- **Buổi 05**: Tái sử dụng metadata và strategy chunking từ dữ liệu Buổi 05 (`chunks`).
- **Buổi 07**: `rag.py` của Buổi 08 kế thừa hoàn toàn từ `rag.py` của Buổi 07 đóng vai trò là Baseline (Semantic Search đơn thuần). Các cải tiến Advanced RAG sẽ được code trong `advanced_rag.py`. 

## 3. Data contract
- Dữ liệu đầu vào cho indexer phải là mảng JSON, chứa các trường: `chunk_id`, `strategy`, `source`, `page_start`, `page_end`, `text`.
- Vector Database sử dụng là ChromaDB (chế độ PersistentClient lưu tại `storage/chroma`).

## 4. BM25 tokenizer/retrieval contract
- Sử dụng mô hình/ngôn ngữ hỗ trợ phân tách từ (tokenizer) phù hợp với tiếng Việt để index các terms.
- API BM25 phải nhận query string, trả về top-K văn bản theo điểm lexical. Trả về format: danh sách chunk_ids kèm score.

## 5. Semantic candidate contract
- Dùng Gemini Embedding Model để nhúng query.
- Truy vấn vector database bằng khoảng cách Cosine, lấy top-K candidates.
- Trả về format: danh sách chunk_ids kèm distance.

## 6. RRF fusion contract
- Reciprocal Rank Fusion (RRF) kết hợp kết quả từ BM25 và Semantic.
- Công thức: `RRF_Score = 1 / (k + rank_bm25) + 1 / (k + rank_semantic)`, với `k` thường bằng 60.
- Output: Một danh sách ranking duy nhất chứa top-N chunk_ids.

## 7. Cross-encoder reranker contract
- Đưa top-N kết quả từ RRF qua mô hình Cross-encoder để chấm điểm (score) lại mối tương quan giữa (Query, Document).
- Output: Top-M kết quả cuối cùng có score cao nhất (M <= N), thỏa mãn ngưỡng tối thiểu.

## 8. Final evidence và citation contract
- Danh sách evidence cuối cùng gửi vào prompt LLM (Gemini).
- Tích hợp kỹ thuật trích dẫn (ví dụ: `[E1], [E2]`).
- Hệ thống map các cờ evidence về metadata nguồn (`source`, `page_start`, `page_end`, `chunk_id`).

## 9. Pipeline trace contract
- Lưu log (hoặc cấu trúc trace) của cả quá trình: 
  1. Câu hỏi gốc
  2. Số lượng/Rank của BM25
  3. Số lượng/Rank của Semantic
  4. Top-N sau RRF
  5. Top-M sau Reranker.
- Phục vụ cho việc debug và hiển thị so sánh.

## 10. Evaluation metrics contract
- Offline đánh giá bằng các metrics chuẩn: `Precision@K`, `Recall@K`, `MRR` (Mean Reciprocal Rank), và tỉ lệ trích dẫn ảo (Hallucination rate).
- Đánh giá trên bộ câu hỏi có sẵn `eval/questions.json`.

## 11. Offline testing contract
- Chạy đánh giá tự động không cần UI.
- Input là mảng `eval/questions.json`, Output là file report `reports/eval_results.json` tổng hợp điểm.

## 12. UI comparison contract
- Ứng dụng Streamlit hiển thị 2 luồng: Baseline (Buổi 07) và Advanced RAG (Buổi 08).
- Hiển thị trực quan sự khác biệt ở danh sách retrieval và chất lượng câu trả lời cuối cùng.
