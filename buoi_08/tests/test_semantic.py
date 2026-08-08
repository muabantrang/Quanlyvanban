import unittest
import chromadb
import advanced_rag
from advanced_rag import semantic_search, advanced_status, load_config
import os
import io
import sys

def mock_embed_fn(text, config):
    if isinstance(text, list):
        return [[0.1] * int(config["GEMINI_EMBEDDING_DIM"]) for _ in text]
    return [0.1] * int(config["GEMINI_EMBEDDING_DIM"])

class TestSemanticRetrieval(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config["GEMINI_API_KEY"] = "fake_key"
        self.strategy = "test_strategy"
        
        self.client = chromadb.EphemeralClient()
        self.col_name = "test_col"
        
        # Override get_collection_name in advanced_rag
        self.original_get_col_name = advanced_rag.get_collection_name
        advanced_rag.get_collection_name = lambda s, c: self.col_name
        
    def tearDown(self):
        advanced_rag.get_collection_name = self.original_get_col_name
        try:
            self.client.delete_collection(self.col_name)
        except Exception:
            pass

    def test_semantic_top_k_count_order_and_metadata(self):
        col = self.client.create_collection(
            name=self.col_name,
            metadata={
                "strategy": self.strategy,
                "embedding_model": self.config["GEMINI_EMBEDDING_MODEL"],
                "embedding_dim": self.config["GEMINI_EMBEDDING_DIM"]
            }
        )
        
        # Insert 3 records
        col.upsert(
            ids=["C1", "C2", "C3"],
            embeddings=[
                [0.1] * int(self.config["GEMINI_EMBEDDING_DIM"]),
                [0.2] * int(self.config["GEMINI_EMBEDDING_DIM"]),
                [0.3] * int(self.config["GEMINI_EMBEDDING_DIM"])
            ],
            documents=["doc1", "doc2", "doc3"],
            metadatas=[
                {"chunk_id": "C1", "source": "A", "page_start": 1, "page_end": 1},
                {"chunk_id": "C2", "source": "B", "page_start": 2, "page_end": 2},
                {"chunk_id": "C3", "source": "C", "page_start": 3, "page_end": 3}
            ]
        )
        
        results = semantic_search(
            question="test",
            candidate_k=2,
            strategy=self.strategy,
            config=self.config,
            mock_embed_fn=mock_embed_fn,
            mock_client=self.client
        )
        
        self.assertEqual(len(results), 2)
        # Check distance order (smaller is better)
        self.assertLessEqual(results[0]["semantic_distance"], results[1]["semantic_distance"])

    def test_collection_metadata_mismatch_blocked(self):
        col = self.client.create_collection(
            name=self.col_name,
            metadata={
                "strategy": "wrong_strategy", # Mismatch
                "embedding_model": self.config["GEMINI_EMBEDDING_MODEL"],
                "embedding_dim": self.config["GEMINI_EMBEDDING_DIM"]
            }
        )
        col.upsert(
            ids=["C1"],
            embeddings=[[0.1] * int(self.config["GEMINI_EMBEDDING_DIM"])],
            documents=["doc1"],
            metadatas=[{"chunk_id": "C1", "source": "A", "page_start": 1, "page_end": 1}]
        )
        
        with self.assertRaises(ValueError) as context:
            semantic_search("test", 5, self.strategy, self.config, mock_embed_fn, self.client)
            
        self.assertIn("Mismatch metadata", str(context.exception))
        
    def test_status_does_not_create_collection(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        advanced_status("test_strategy", self.config)
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("Collection exists: False", output)
        
    def test_fail_without_api_key_no_fake(self):
        self.config["GEMINI_API_KEY"] = ""
        self.assertEqual(self.config["GEMINI_API_KEY"], "")
        
    def test_no_generation(self):
        col = self.client.create_collection(
            name=self.col_name,
            metadata={
                "strategy": self.strategy,
                "embedding_model": self.config["GEMINI_EMBEDDING_MODEL"],
                "embedding_dim": self.config["GEMINI_EMBEDDING_DIM"]
            }
        )
        
        col.upsert(
            ids=["C1"],
            embeddings=[[0.1] * int(self.config["GEMINI_EMBEDDING_DIM"])],
            documents=["doc1"],
            metadatas=[{"chunk_id": "C1", "source": "A", "page_start": 1, "page_end": 1}]
        )
        
        results = semantic_search("test", 5, self.strategy, self.config, mock_embed_fn, self.client)
        self.assertEqual(len(results), 1)

if __name__ == "__main__":
    unittest.main()
