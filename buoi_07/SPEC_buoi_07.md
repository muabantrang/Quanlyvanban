# Agent Specification - Buổi 07

## Workspace
- Vùng được đọc: `rag_foundation/buoi_05/output/chunks/`, `rag_foundation/buoi_05/.venv/`, `rag_foundation/buoi_06/`, `rag_foundation/buoi_07/`
- Vùng được ghi: `rag_foundation/buoi_07/`
- Không sửa Buổi 05 và Buổi 06

## Python
- Dùng `.venv` Buổi 05
- Không tạo venv mới

## Input
- JSON trong `buoi_05/output/chunks/`
- Buổi 05 là nguồn dữ liệu đã chuẩn bị
- Không OCR, parse PDF hoặc chunk lại

## Packages
- Chỉ dùng package được quy định

## Pipeline
- validate
- embedding
- Chroma persistent
- retrieval
- confidence gate
- generation
- citation
- Streamlit
- unittest offline

## Data Contract
Các field bắt buộc:
- chunk_id
- strategy
- source
- page_start
- page_end
- text

## Index Contract
- một strategy trong một collection
- model và dimension của index/query phải khớp
- dùng embedding thật
- không dùng vector giả
- chặn NaN, Infinity, boolean và zero vector
- Chroma cosine, `embedding_function=None`
- idempotent
- status read-only
- validate embedding xong trước khi reset/upsert

## Retrieval Contract
- trả evidence thật
- có distance
- chỉ evidence đạt threshold được đưa vào generation
- evidence yếu thì không gọi generation

## Citation Contract
- citation lấy từ metadata thật
- không tin source/page/chunk_id do LLM tự tạo
- result có `citations` và `warnings`; code thay label hợp lệ bằng citation thật

## Security
- không lộ secret

## Testing
- unittest
- mock API
- temporary storage
- không Internet/key thật

## Coding Style
- ít file
- ít class
- ít function
- không kiến trúc phức tạp
