# Bo chunk chuan cho Buoi 05

Du lieu dau vao da kiem chung, dung chung cho ca lop de Buoi 06 va Buoi 07
khong phu thuoc vao ket qua chunking khac nhau tren tung may.

## Noi dung

| Chi so | Gia tri |
|---|---|
| So file | 9 |
| Tong chunk | 581 |
| hierarchical | 346 |
| semantic | 116 |
| fixed-size | 119 |
| So trang | TT_02: 1-11 · TT_06: 1-9 · TT_39: 2-18 |
| Text rong | 0 |
| chunk_id trung | 0 |
| Tinh don dieu trang | 100% |

Nguon: 3 Thong tu cong khai cua Ngan hang Nha nuoc Viet Nam
(TT 02/2023, TT 06/2023, TT 39/2016). Text da OCR bang LlamaParse,
chuan hoa Unicode NFC.

## Truong du lieu

Moi file la mot JSON list. Moi phan tu co:

    chunk_id, strategy, source, page_start, page_end, text, structure

`strategy` chi nhan: fixed-size, semantic, hierarchical
`structure` chi co gia tri o chien luoc hierarchical, con lai la null.

## Cach dung

1. Chuyen toan bo file cu trong `buoi_05/output/chunks/` sang mot thu muc khac
   de sao luu. KHONG xoa.
2. Chep 9 file .json trong goi nay vao `buoi_05/output/chunks/`.
3. Kiem tra lai:

       <PYTHON> rag_foundation/buoi_07/rag.py validate --strategy hierarchical
       <PYTHON> rag_foundation/buoi_07/rag.py validate --strategy semantic
       <PYTHON> rag_foundation/buoi_07/rag.py validate --strategy fixed-size

   Ket qua mong doi: 346 / 116 / 119 chunk hop le, 0 text rong.

Khong chep file MANIFEST.md va README.md nay vao thu muc chunks.
