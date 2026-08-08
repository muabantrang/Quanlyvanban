import unittest
import json
import tempfile
from pathlib import Path
import rag

class TestLoaderAndValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.input_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def write_json(self, data, filename="test.json"):
        with open(self.input_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    # 1. Loader đọc JSON list.
    def test_loader_reads_json_list(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "src1", "text": "abc", "page_start": 1, "page_end": 1}
        ])
        chunks, stats = rag.load_chunks(self.input_dir, "hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "1")

    # 2. Loader đọc object có field chunks.
    def test_loader_reads_object_with_chunks_field(self):
        self.write_json({
            "meta": "data",
            "chunks": [
                {"chunk_id": "2", "strategy": "hierarchical", "source": "src1", "text": "abc", "page_start": 1, "page_end": 1}
            ]
        })
        chunks, stats = rag.load_chunks(self.input_dir, "hierarchical")
        self.assertEqual(len(chunks), 1)

    # 3. Chỉ lấy đúng strategy.
    def test_loader_filters_strategy(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "src1", "text": "abc", "page_start": 1, "page_end": 1},
            {"chunk_id": "2", "strategy": "semantic", "source": "src1", "text": "def", "page_start": 1, "page_end": 1}
        ])
        chunks, stats = rag.load_chunks(self.input_dir, "hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "1")

    # 4. Thiếu field bắt buộc phải fail.
    def test_missing_required_field(self):
        self.write_json([
            {"strategy": "hierarchical", "source": "src1", "text": "abc", "page_start": 1, "page_end": 1}
        ])
        with self.assertRaises(ValueError):
            rag.load_chunks(self.input_dir, "hierarchical")

    # 5. Field sai kiểu phải fail.
    def test_wrong_type_field(self):
        self.write_json([
            {"chunk_id": 123, "strategy": "hierarchical", "source": "src1", "text": "abc", "page_start": 1, "page_end": 1}
        ])
        with self.assertRaises(ValueError):
            rag.load_chunks(self.input_dir, "hierarchical")

    # 6. Boolean không được chấp nhận làm page number.
    def test_boolean_not_allowed_for_page(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "src1", "text": "abc", "page_start": True, "page_end": 1}
        ])
        with self.assertRaises(ValueError):
            rag.load_chunks(self.input_dir, "hierarchical")

    # 7. page_start > page_end phải fail.
    def test_page_start_greater_than_page_end(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "src1", "text": "abc", "page_start": 2, "page_end": 1}
        ])
        with self.assertRaises(ValueError):
            rag.load_chunks(self.input_dir, "hierarchical")

    # 8. Text rỗng bị bỏ qua và thống kê đúng.
    def test_empty_text_skipped(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "src1", "text": "   ", "page_start": 1, "page_end": 1},
            {"chunk_id": "2", "strategy": "hierarchical", "source": "src1", "text": "ok", "page_start": 1, "page_end": 1}
        ])
        chunks, stats = rag.load_chunks(self.input_dir, "hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(stats["empty_text_skipped"], 1)

    # 9. Duplicate chunk_id phải fail.
    def test_duplicate_chunk_id(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "src1", "text": "abc", "page_start": 1, "page_end": 1},
            {"chunk_id": "1", "strategy": "hierarchical", "source": "src1", "text": "def", "page_start": 1, "page_end": 1}
        ])
        with self.assertRaises(ValueError):
            rag.load_chunks(self.input_dir, "hierarchical")

    # 38. Loader chặn record không phải JSON object.
    def test_record_not_object(self):
        self.write_json(["this is not an object"])
        with self.assertRaises(ValueError):
            rag.load_chunks(self.input_dir, "hierarchical")

if __name__ == '__main__':
    unittest.main()
