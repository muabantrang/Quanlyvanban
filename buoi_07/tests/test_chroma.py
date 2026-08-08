import unittest
import json
import tempfile
from pathlib import Path
import math
import rag

class TestEmbeddingAndChroma(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.input_dir = Path(self.temp_dir.name) / "chunks"
        self.input_dir.mkdir()
        self.storage_dir = Path(self.temp_dir.name) / "storage"
        self.storage_dir.mkdir()
        
        self.config_override = {
            "GEMINI_API_KEY": "test_key",
            "GEMINI_EMBEDDING_MODEL": "test-model",
            "GEMINI_EMBEDDING_DIM": "128",
            "RAG_MAX_DISTANCE": "0.5",
            "GEMINI_GENERATION_MODEL": "test-gen-model"
        }
        
    def tearDown(self):
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.temp_dir.cleanup()

    def write_json(self, data, filename="test.json"):
        with open(self.input_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    def mock_embed_valid(self, chunks, config):
        return [[0.1] * 128 for _ in chunks]

    # 10. Index hai lần không tăng record count.
    def test_idempotency_index_twice(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, self.mock_embed_valid)
        res = rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, self.mock_embed_valid)
        self.assertEqual(res["new_count"], 1)
        self.assertEqual(res["old_count"], 1)

    # 11. Metadata citation được lưu đầy đủ.
    def test_metadata_saved_correctly(self):
        self.write_json([
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "src.pdf", "text": "t", "page_start": 1, "page_end": 2}
        ])
        res = rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, self.mock_embed_valid)
        
        # Verify directly in Chroma
        import chromadb
        client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma"))
        col = client.get_collection(name=res["collection_name"], embedding_function=None)
        doc = col.get(ids=["c1"])
        meta = doc["metadatas"][0]
        self.assertEqual(meta["source"], "src.pdf")
        self.assertEqual(meta["page_start"], 1)
        self.assertEqual(meta["page_end"], 2)
        self.assertEqual(meta["chunk_id"], "c1")
        self.assertEqual(meta["strategy"], "hierarchical")

    # 12. Collection identity thay đổi khi strategy thay đổi.
    def test_collection_changes_with_strategy(self):
        name1 = rag.get_collection_name("hierarchical", self.config_override)
        name2 = rag.get_collection_name("semantic", self.config_override)
        self.assertNotEqual(name1, name2)

    # 13. Collection identity thay đổi khi model hoặc dimension thay đổi.
    def test_collection_changes_with_model_or_dim(self):
        name1 = rag.get_collection_name("hierarchical", self.config_override)
        
        config2 = self.config_override.copy()
        config2["GEMINI_EMBEDDING_MODEL"] = "new-model"
        name2 = rag.get_collection_name("hierarchical", config2)
        
        config3 = self.config_override.copy()
        config3["GEMINI_EMBEDDING_DIM"] = "256"
        name3 = rag.get_collection_name("hierarchical", config3)
        
        self.assertNotEqual(name1, name2)
        self.assertNotEqual(name1, name3)

    # 14. Query chặn collection có metadata không khớp.
    def test_query_blocks_mismatched_metadata(self):
        # Create a collection with some metadata
        import chromadb
        client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma"))
        col_name = rag.get_collection_name("hierarchical", self.config_override)
        col = client.create_collection(
            name=col_name,
            embedding_function=None,
            metadata={"strategy": "semantic"} # intentionally mismatched
        )
        col.upsert(ids=["1"], embeddings=[[0.1]*128], documents=["d"])
        
        with self.assertRaisesRegex(ValueError, "Mismatch metadata"):
            rag.do_query("hello", 5, "hierarchical", self.config_override, self.storage_dir, None, None)

    # 42. Existing collection có metadata/configuration mismatch bị chặn trước upsert.
    def test_index_blocks_mismatched_metadata(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma"))
        col_name = rag.get_collection_name("hierarchical", self.config_override)
        client.create_collection(
            name=col_name,
            embedding_function=None,
            metadata={"strategy": "semantic"} # intentionally mismatched
        )
        
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        with self.assertRaisesRegex(ValueError, "Mismatch metadata"):
            rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, self.mock_embed_valid)

    # 15. Embedding trả sai số vector phải fail.
    def test_embedding_wrong_number_of_vectors(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        def mock_embed_wrong_count(chunks, config):
            return [] # Returns 0 vectors instead of 1
        
        with self.assertRaisesRegex(ValueError, "Số lượng embeddings không khớp"):
            rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, mock_embed_wrong_count)

    # 16. Embedding trả vector rỗng phải fail (already covered by length validation but let's test specific logic).
    def test_embedding_empty_vector(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        def mock_embed_empty(chunks, config):
            return [[]]
            
        with self.assertRaisesRegex(ValueError, "Dimension sai lệch"):
            rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, mock_embed_empty)

    # 17. Embedding trả sai dimension phải fail.
    def test_embedding_wrong_dimension(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        def mock_embed_dim(chunks, config):
            return [[0.1] * 100] # dim 100 instead of 128
            
        with self.assertRaisesRegex(ValueError, "Dimension sai lệch"):
            rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, mock_embed_dim)

    # 18. Embedding có NaN hoặc Infinity phải fail.
    def test_embedding_nan_inf(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        def mock_embed_nan(chunks, config):
            vec = [0.1] * 128
            vec[0] = math.nan
            return [vec]
            
        with self.assertRaisesRegex(ValueError, "chứa NaN hoặc Infinity"):
            rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, mock_embed_nan)

    # 39. Embedding chặn boolean và zero vector.
    def test_embedding_boolean_zero(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        def mock_embed_bool(chunks, config):
            vec = [0.1] * 128
            vec[0] = True
            return [vec]
            
        with self.assertRaisesRegex(ValueError, "giá trị không hợp lệ"):
            rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, mock_embed_bool)
            
        def mock_embed_zero(chunks, config):
            return [[0.0] * 128]
            
        with self.assertRaisesRegex(ValueError, "Zero vector"):
            rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, mock_embed_zero)

    # 19. Embedding lỗi trước upsert không thêm record mới.
    def test_embedding_error_does_not_upsert(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        # First index succeeds
        rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, self.mock_embed_valid)
        
        self.write_json([
            {"chunk_id": "2", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        # Second index fails during embedding
        def mock_embed_fail(chunks, config):
            raise ValueError("Simulated failure")
            
        with self.assertRaisesRegex(ValueError, "Simulated failure"):
            rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, mock_embed_fail)
            
        status = rag.get_status("hierarchical", self.config_override, self.storage_dir)
        self.assertEqual(status["record_count"], 1) # Only chunk 1 exists

    # 20. Thiếu API key phải fail rõ và không upsert vector giả.
    def test_missing_api_key_blocks_index(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        cfg = self.config_override.copy()
        cfg["GEMINI_API_KEY"] = ""
        
        with self.assertRaisesRegex(ValueError, "Thiếu GEMINI_API_KEY"):
            rag.do_index(self.input_dir, "hierarchical", False, cfg, self.storage_dir, self.mock_embed_valid)

    # 40. status trên storage trống không tạo collection.
    def test_status_does_not_create_collection(self):
        status = rag.get_status("hierarchical", self.config_override, self.storage_dir)
        self.assertFalse(status["collection_exists"])
        
        import chromadb
        client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma"))
        col_name = rag.get_collection_name("hierarchical", self.config_override)
        with self.assertRaises(Exception):
            client.get_collection(name=col_name) # Ensure it doesn't exist

    # 41. --reset gặp embedding lỗi vẫn giữ nguyên collection hợp lệ cũ.
    def test_reset_with_embedding_error_keeps_old_collection(self):
        self.write_json([
            {"chunk_id": "1", "strategy": "hierarchical", "source": "s", "text": "t", "page_start": 1, "page_end": 1}
        ])
        # Index successfully
        rag.do_index(self.input_dir, "hierarchical", False, self.config_override, self.storage_dir, self.mock_embed_valid)
        
        # Reset and fail embedding
        def mock_embed_fail(chunks, config):
            raise ValueError("Simulated failure")
            
        with self.assertRaises(ValueError):
            rag.do_index(self.input_dir, "hierarchical", True, self.config_override, self.storage_dir, mock_embed_fail)
            
        status = rag.get_status("hierarchical", self.config_override, self.storage_dir)
        self.assertTrue(status["collection_exists"])
        self.assertEqual(status["record_count"], 1) # Still intact

if __name__ == '__main__':
    unittest.main()
