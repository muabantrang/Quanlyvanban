import json
import sys
import glob
import re
from pathlib import Path

chunks_dir = Path(r"c:\Rag_Agribank_Thuchanh\RAG\rag_foundation\buoi_05\output\chunks")
files = glob.glob(str(chunks_dir / "*.hierarchical.json"))

total_records = 0
sources = set()

structure_keys_combo = {}
records_with_chapter_article_clause_point = 0
records_no_structure = 0

heading_dieu_pattern = re.compile(r'(?i)(?:^|\n)\s*(Điều\s+\d+)', re.UNICODE)
heading_chuong_pattern = re.compile(r'(?i)(?:^|\n)\s*(Chương\s+[IVXLCDM\d]+)', re.UNICODE)
chuong_dieu_headings = 0

lengths = []
chunk_id_sequence_check = {}

metadata_article_every_child = True
metadata_missing_but_text_has_heading = []
modified_article_quoted = []

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        total_records += len(data)
        
        last_chunk_num = -1
        
        for record in data:
            source = record.get("source")
            if source:
                sources.add(source)
                
            text = record.get("text", "")
            lengths.append(len(text))
            
            # Check structure keys
            struct = record.get("structure", {})
            keys = tuple(sorted(struct.keys()))
            structure_keys_combo[keys] = structure_keys_combo.get(keys, 0) + 1
            
            has_chap = "chapter" in struct
            has_art = "article" in struct
            has_cl = "clause" in struct
            has_pt = "point" in struct
            
            if has_chap or has_art or has_cl or has_pt:
                records_with_chapter_article_clause_point += 1
            if not keys:
                records_no_structure += 1
                
            # Headings in text
            if heading_dieu_pattern.search(text) or heading_chuong_pattern.search(text):
                chuong_dieu_headings += 1
                if not keys and len(metadata_missing_but_text_has_heading) < 3:
                    metadata_missing_but_text_has_heading.append(record.get("chunk_id"))
                    
            # Check chunk_id sequence
            chunk_id = record.get("chunk_id", "")
            if chunk_id:
                parts = chunk_id.split("::")
                if len(parts) >= 3:
                    num = int(parts[2])
                    if num <= last_chunk_num:
                        chunk_id_sequence_check[source] = False
                    if source not in chunk_id_sequence_check:
                        chunk_id_sequence_check[source] = True
                    last_chunk_num = num
                    
            # metadata article every child? 
            # (In buoi 05 we didn't inject parent metadata to children explicitly if they are standalone, or maybe we did)
            # Actually, buoi 05 hierarchical chunks should have "article" in structure if it's under an article.
            if not has_art and has_cl:
                metadata_article_every_child = False
                
            # Check for modified article quoted
            if "sửa đổi" in text.lower() or "bổ sung" in text.lower():
                if "điều" in text.lower() and len(modified_article_quoted) < 3:
                    modified_article_quoted.append(record.get("chunk_id"))

lengths.sort()
min_len = lengths[0] if lengths else 0
max_len = lengths[-1] if lengths else 0
median_len = lengths[len(lengths)//2] if lengths else 0
p95_len = lengths[int(len(lengths)*0.95)] if lengths else 0

print("== BÁO CÁO HIERARCHY ==")
print(f"Số file: {len(files)}")
print(f"Số record: {total_records}")
print(f"Số source: {len(sources)}")
print(f"Số record theo tổ hợp structure keys: {structure_keys_combo}")
print(f"Số record có chapter/article/clause/point: {records_with_chapter_article_clause_point}")
print(f"Số record không có structure: {records_no_structure}")
print(f"Số heading Chương/Điều nhận diện trong text: {chuong_dieu_headings}")
print(f"Độ dài text min: {min_len}, median: {median_len}, p95: {p95_len}, max: {max_len}")
print(f"chunk_id có thứ tự số ổn định: {all(chunk_id_sequence_check.values())}")
print(f"metadata article được lặp ở mọi child: {metadata_article_every_child}")
print(f"Ví dụ metadata thiếu nhưng text có heading: {metadata_missing_but_text_has_heading}")
print(f"Ví dụ văn bản sửa đổi có Điều được trích dẫn: {modified_article_quoted}")
