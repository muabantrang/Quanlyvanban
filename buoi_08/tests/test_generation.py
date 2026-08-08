import unittest
from unittest.mock import patch, MagicMock
from advanced_rag import generate_advanced_answer, compare_retrieval

class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.config = {
            "RAG_MAX_DISTANCE": 0.45,
            "RERANK_MIN_SCORE": 0.50,
            "FINAL_TOP_K": 2,
            "BM25_CANDIDATES": 2,
            "SEMANTIC_CANDIDATES": 2,
            "RERANK_CANDIDATES": 2
        }
        self.chunks = [{"chunk_id": "C1", "text": "doc1", "source": "A", "page_start": 1, "page_end": 1}]
        
        self.mock_deps = {
            "embed": MagicMock(return_value=[0.1]*768),
            "client": MagicMock(),
            "rerank": MagicMock(),
            "generate": MagicMock(return_value="Mocked answer [E1]")
        }
        
    @patch("advanced_rag.hybrid_search")
    @patch("advanced_rag.run_reranker")
    def test_gating_logic_per_mode(self, mock_rr, mock_hyb):
        # hybrid_rerank mode gating
        fused = [{"chunk_id": "C1", "semantic_distance": 0.2}]
        mock_hyb.return_value = (fused, {"latency_ms": {"bm25": 0, "semantic": 0, "fusion": 0}, "bm25_candidate_count": 1, "semantic_candidate_count": 1, "union_count": 1, "overlap_count": 1})
        # Mock rerank output: rerank_score < min_score
        mock_rr.return_value = ([{"chunk_id": "C1", "rerank_score": 0.4}], {"rerank_latency_ms": 0})
        
        ans = generate_advanced_answer("test", "hybrid_rerank", "strategy", [], self.config, self.mock_deps)
        self.assertEqual(ans["status"], "insufficient_evidence")
        self.assertFalse(ans["evidence"][0]["accepted"])
        
        # Test semantic distance gate for semantic mode
        with patch("advanced_rag.semantic_search") as mock_sem:
            mock_sem.return_value = [{"chunk_id": "C1", "semantic_distance": 0.6}]
            ans2 = generate_advanced_answer("test", "semantic", "strategy", [], self.config, self.mock_deps)
            self.assertEqual(ans2["status"], "insufficient_evidence")
            self.assertFalse(ans2["evidence"][0]["accepted"])
            
    @patch("advanced_rag.semantic_search")
    def test_rejected_evidence_excluded_from_prompt(self, mock_sem):
        mock_sem.return_value = [
            {"chunk_id": "C1", "semantic_distance": 0.2, "text": "GOOD", "source": "A", "page_start": 1, "page_end": 1},
            {"chunk_id": "C2", "semantic_distance": 0.6, "text": "BAD", "source": "A", "page_start": 1, "page_end": 1}
        ]
        
        mock_gen = MagicMock(return_value="Answer [E1]")
        deps = {"generate": mock_gen}
        ans = generate_advanced_answer("test", "semantic", "strategy", [], self.config, deps)
        
        self.assertEqual(ans["status"], "answered")
        self.assertTrue(mock_gen.called)
        
        # Check that prompt contains "GOOD" but not "BAD"
        system_prompt, user_prompt = mock_gen.call_args[0]
        self.assertIn("GOOD", user_prompt)
        self.assertNotIn("BAD", user_prompt)
        
    @patch("advanced_rag.bm25_search")
    def test_trace_counts_timing_keys(self, mock_bm25):
        mock_bm25.return_value = [{"chunk_id": "C1", "semantic_distance": 0.2, "text": "GOOD", "source": "A", "page_start": 1, "page_end": 1}]
        ans = generate_advanced_answer("test", "bm25", "strategy", [], self.config, self.mock_deps)
        trace = ans["trace"]
        self.assertIn("bm25_candidates", trace)
        self.assertIn("semantic_candidates", trace)
        self.assertIn("overlap", trace)
        self.assertIn("union", trace)
        self.assertIn("reranked", trace)
        self.assertIn("accepted", trace)
        self.assertIn("generation_called", trace)
        
        latency = trace["latency_ms"]
        self.assertIn("bm25", latency)
        self.assertIn("semantic", latency)
        self.assertIn("fusion", latency)
        self.assertIn("rerank", latency)
        self.assertIn("generation", latency)
        self.assertIn("total", latency)
        
    @patch("advanced_rag.semantic_search")
    def test_citation_mapping_and_fake_label(self, mock_sem):
        mock_sem.return_value = [
            {"chunk_id": "C1", "semantic_distance": 0.2, "text": "GOOD", "source": "A", "page_start": 1, "page_end": 1}
        ]
        # LLM returns E1 (valid) and E99 (fake)
        mock_gen = MagicMock(return_value="Answer [E1] and fake [E99]")
        deps = {"generate": mock_gen}
        
        ans = generate_advanced_answer("test", "semantic", "strategy", [], self.config, deps)
        
        self.assertEqual(len(ans["citations"]), 1)
        self.assertEqual(ans["citations"][0]["label"], "[E1]")
        self.assertEqual(ans["citations"][0]["chunk_id"], "C1")
        
        self.assertEqual(len(ans["warnings"]), 1)
        self.assertIn("E99", ans["warnings"][0])
        
    @patch("advanced_rag.semantic_search")
    def test_generation_max_1_call(self, mock_sem):
        mock_sem.return_value = [{"chunk_id": "C1", "semantic_distance": 0.2, "text": "GOOD", "source": "A", "page_start": 1, "page_end": 1}]
        mock_gen = MagicMock(return_value="Answer")
        deps = {"generate": mock_gen}
        
        generate_advanced_answer("test", "semantic", "strategy", [], self.config, deps)
        mock_gen.assert_called_once()
        
    @patch("advanced_rag.generate_advanced_answer")
    def test_compare_does_not_call_generation(self, mock_gen_answer):
        mock_gen_answer.return_value = {"trace": {"latency_ms": {"total": 0}}, "evidence": []}
        compare_retrieval("test", [], "strategy", self.config, self.mock_deps)
        
        # compare calls generate_advanced_answer 4 times
        self.assertEqual(mock_gen_answer.call_count, 4)
        # But wait, compare_retrieval actually does call generation inside if not patched?
        # Let's test by patching generation mock directly instead of generate_advanced_answer
        pass
        
    def test_compare_no_generation_leak(self):
        mock_gen = MagicMock()
        deps = {"generate": mock_gen}
        
        # We need mock_bm25, mock_sem, etc to not crash.
        with patch("advanced_rag.bm25_search", return_value=[]), \
             patch("advanced_rag.semantic_search", return_value=[]), \
             patch("advanced_rag.hybrid_search", return_value=([], {"latency_ms": {"bm25": 0, "semantic": 0, "fusion": 0}, "bm25_candidate_count": 0, "semantic_candidate_count": 0, "union_count": 0, "overlap_count": 0})), \
             patch("advanced_rag.run_reranker", return_value=([], {"rerank_latency_ms": 0})):
             
            compare_retrieval("test", [], "strategy", self.config, deps)
            
            # Since all searches return [], generation is skipped due to insufficient_evidence
            mock_gen.assert_not_called()
            
    @patch("advanced_rag.hybrid_search")
    @patch("advanced_rag.run_reranker")
    def test_reranker_unavailable_status(self, mock_rr, mock_hyb):
        mock_hyb.return_value = ([{"chunk_id": "C1"}], {"latency_ms": {"bm25": 0, "semantic": 0, "fusion": 0}, "bm25_candidate_count": 1, "semantic_candidate_count": 1, "union_count": 1, "overlap_count": 1})
        mock_rr.side_effect = RuntimeError("reranker_unavailable")
        
        ans = generate_advanced_answer("test", "hybrid_rerank", "strategy", [], self.config, self.mock_deps)
        self.assertEqual(ans["status"], "reranker_unavailable")
        
    @patch("advanced_rag.bm25_search")
    def test_complete_schema(self, mock_bm25):
        mock_bm25.return_value = []
        ans = generate_advanced_answer("test", "bm25", "strategy", [], self.config, self.mock_deps)
        
        keys = ["status", "mode", "question", "answer", "evidence", "citations", "warnings", "trace"]
        for k in keys:
            self.assertIn(k, ans)

if __name__ == "__main__":
    unittest.main()
