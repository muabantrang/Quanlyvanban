import unittest
import math
from unittest.mock import patch, MagicMock
from advanced_rag import run_reranker, load_config
import advanced_rag

def mock_rerank_fn(question, texts):
    # Returns raw logits (which will be converted to sigmoid)
    # We return a specific logit for each text to test sorting
    # "doc1" -> logit 0.0 -> sigmoid 0.5
    # "doc2" -> logit 2.0 -> sigmoid ~0.88
    # "doc3" -> logit -2.0 -> sigmoid ~0.12
    scores = []
    for t in texts:
        if "doc1" in t: scores.append(0.0)
        elif "doc2" in t: scores.append(2.0)
        elif "doc3" in t: scores.append(-2.0)
        else: scores.append(1.0)
    return scores

class TestRerank(unittest.TestCase):
    def setUp(self):
        self.config = {
            "RERANKER_MODEL": "test-model",
            "RERANK_DEVICE": "cpu",
            "RERANK_CANDIDATES": 2,
            "FINAL_TOP_K": 2,
            "RERANK_BATCH_SIZE": 2,
            "RERANKER_MAX_LENGTH": 512
        }
        
    @patch('advanced_rag.load_reranker')
    def test_lazy_loading(self, mock_load):
        # When using mock_rerank_fn, load_reranker should not be called
        candidates = [{"text": "doc1", "fused_rank": 1, "chunk_id": "C1"}]
        run_reranker("test", candidates, self.config, mock_rerank_fn=mock_rerank_fn)
        mock_load.assert_not_called()
        
    def test_one_pair_per_candidate(self):
        # We can verify that mock_rerank_fn receives exactly the texts provided
        candidates = [
            {"text": "doc1", "fused_rank": 1, "chunk_id": "C1"},
            {"text": "doc2", "fused_rank": 2, "chunk_id": "C2"}
        ]
        
        called_texts = []
        def my_mock(q, texts):
            called_texts.extend(texts)
            return [1.0] * len(texts)
            
        run_reranker("test", candidates, self.config, mock_rerank_fn=my_mock)
        self.assertEqual(len(called_texts), 2)
        self.assertEqual(called_texts, ["doc1", "doc2"])
        
    def test_batching_does_not_change_count(self):
        candidates = [
            {"text": "doc1", "fused_rank": 1, "chunk_id": "C1"},
            {"text": "doc2", "fused_rank": 2, "chunk_id": "C2"}
        ]
        self.config["RERANK_CANDIDATES"] = 2
        self.config["FINAL_TOP_K"] = 2
        results, _ = run_reranker("test", candidates, self.config, mock_rerank_fn=mock_rerank_fn)
        self.assertEqual(len(results), 2)
        
    def test_sigmoid_score_correct(self):
        candidates = [{"text": "doc1", "fused_rank": 1, "chunk_id": "C1"}]
        # mock returns logit 0.0 for doc1
        results, _ = run_reranker("test", candidates, self.config, mock_rerank_fn=mock_rerank_fn)
        self.assertEqual(results[0]["rerank_raw_score"], 0.0)
        self.assertAlmostEqual(results[0]["rerank_score"], 0.5)
        
    def test_sort_and_tie_break_correct(self):
        candidates = [
            {"text": "doc1", "fused_rank": 2, "chunk_id": "C1"},
            {"text": "doc2", "fused_rank": 1, "chunk_id": "C2"},
            {"text": "doc3", "fused_rank": 3, "chunk_id": "C3"}
        ]
        self.config["RERANK_CANDIDATES"] = 3
        self.config["FINAL_TOP_K"] = 3
        results, _ = run_reranker("test", candidates, self.config, mock_rerank_fn=mock_rerank_fn)
        # Expected scores: doc2 (2.0) -> 0.88, doc1 (0.0) -> 0.5, doc3 (-2.0) -> 0.12
        # Order should be C2, C1, C3
        self.assertEqual(results[0]["chunk_id"], "C2")
        self.assertEqual(results[1]["chunk_id"], "C1")
        self.assertEqual(results[2]["chunk_id"], "C3")
        
        # Tie break check
        candidates_tie = [
            {"text": "doc1", "fused_rank": 2, "chunk_id": "C1"},
            {"text": "doc1", "fused_rank": 1, "chunk_id": "C2"}
        ]
        results, _ = run_reranker("test", candidates_tie, self.config, mock_rerank_fn=mock_rerank_fn)
        # Both get 0.5, tie broken by fused_rank (1 vs 2) -> C2 first
        self.assertEqual(results[0]["chunk_id"], "C2")
        self.assertEqual(results[1]["chunk_id"], "C1")

    def test_rank_change_calculation(self):
        candidates = [
            {"text": "doc3", "fused_rank": 1, "chunk_id": "C3"}, # Gets low score, drops
            {"text": "doc2", "fused_rank": 2, "chunk_id": "C2"}  # Gets high score, rises
        ]
        results, _ = run_reranker("test", candidates, self.config, mock_rerank_fn=mock_rerank_fn)
        
        c2 = results[0] # Rerank 1
        c3 = results[1] # Rerank 2
        self.assertEqual(c2["chunk_id"], "C2")
        self.assertEqual(c2["fused_rank"], 2)
        self.assertEqual(c2["rerank_rank"], 1)
        self.assertEqual(c2["rank_change"], 1) # 2 - 1 = +1
        
        self.assertEqual(c3["chunk_id"], "C3")
        self.assertEqual(c3["fused_rank"], 1)
        self.assertEqual(c3["rerank_rank"], 2)
        self.assertEqual(c3["rank_change"], -1) # 1 - 2 = -1
        
    def test_only_rerank_limited_candidates(self):
        candidates = [{"text": "doc1", "fused_rank": i, "chunk_id": f"C{i}"} for i in range(10)]
        self.config["RERANK_CANDIDATES"] = 3
        self.config["FINAL_TOP_K"] = 5
        results, _ = run_reranker("test", candidates, self.config, mock_rerank_fn=mock_rerank_fn)
        # Even though top K is 5, we only rerank 3 candidates, so output is 3
        self.assertEqual(len(results), 3)
        
    def test_returns_only_final_top_k(self):
        candidates = [{"text": "doc1", "fused_rank": i, "chunk_id": f"C{i}"} for i in range(10)]
        self.config["RERANK_CANDIDATES"] = 10
        self.config["FINAL_TOP_K"] = 4
        results, _ = run_reranker("test", candidates, self.config, mock_rerank_fn=mock_rerank_fn)
        self.assertEqual(len(results), 4)
        
    @patch('advanced_rag.load_reranker')
    def test_model_error_no_silent_fallback(self, mock_load):
        mock_load.side_effect = RuntimeError("Lỗi tải mô hình reranker (reranker_unavailable): Network error")
        candidates = [{"text": "doc1", "fused_rank": 1, "chunk_id": "C1"}]
        
        with self.assertRaises(RuntimeError) as context:
            run_reranker("test", candidates, self.config)
            
        self.assertIn("reranker_unavailable", str(context.exception))
        
    def test_no_model_loading_or_network(self):
        # We assert that _RERANKER_MODEL is still None after running with mock
        advanced_rag._RERANKER_MODEL = None
        candidates = [{"text": "doc1", "fused_rank": 1, "chunk_id": "C1"}]
        run_reranker("test", candidates, self.config, mock_rerank_fn=mock_rerank_fn)
        self.assertIsNone(advanced_rag._RERANKER_MODEL)

if __name__ == "__main__":
    unittest.main()
