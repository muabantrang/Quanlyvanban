import unittest
from advanced_rag import tokenize_vi_legal, bm25_search

class TestBM25Lexical(unittest.TestCase):
    def test_tokenizer_keeps_vietnamese_accents(self):
        text = "ngân hàng nhà nước Việt Nam"
        tokens = tokenize_vi_legal(text)
        self.assertIn("ngân", tokens)
        self.assertIn("hàng", tokens)
        self.assertIn("nhà", tokens)
        self.assertIn("nước", tokens)
        self.assertIn("việt", tokens)
        self.assertIn("nam", tokens)

    def test_tokenizer_keeps_article_and_clause_numbers(self):
        text = "Điều 7, Khoản 2"
        tokens = tokenize_vi_legal(text)
        self.assertEqual(tokens, ["điều", "7", "khoản", "2"])

    def test_preprocessing_is_identical(self):
        query = "Thế chấp tài sản"
        corpus_text = "thế chấp tài sản"
        
        q_tokens = tokenize_vi_legal(query)
        c_tokens = tokenize_vi_legal(corpus_text)
        self.assertEqual(q_tokens, c_tokens)

    def test_exact_term_ranked_higher(self):
        question = "cơ cấu lại thời hạn trả nợ"
        chunks = [
            {"chunk_id": "C1", "text": "Khoản vay này không quy định cơ cấu lại thời hạn trả nợ.", "source": "A", "page_start": 1, "page_end": 1},
            {"chunk_id": "C2", "text": "Việc giải ngân phải được thực hiện đúng quy trình.", "source": "B", "page_start": 1, "page_end": 1},
            {"chunk_id": "C3", "text": "Một văn bản khác không liên quan.", "source": "C", "page_start": 1, "page_end": 1}
        ]
        
        results = bm25_search(question, chunks, candidate_k=5)
        self.assertEqual(len(results), 3)
        # C1 has the exact keywords, C2 has none.
        self.assertEqual(results[0]["chunk_id"], "C1")
        self.assertEqual(results[1]["chunk_id"], "C2")
        self.assertGreater(results[0]["bm25_score"], results[1]["bm25_score"])

    def test_candidate_k_larger_than_corpus(self):
        question = "điều khoản"
        chunks = [
            {"chunk_id": "C1", "text": "điều khoản 1", "source": "A", "page_start": 1, "page_end": 1},
        ]
        # candidate_k = 10 while corpus has 1 chunk
        results = bm25_search(question, chunks, candidate_k=10)
        self.assertEqual(len(results), 1)

    def test_empty_question_fails(self):
        chunks = [{"chunk_id": "C1", "text": "test", "source": "A", "page_start": 1, "page_end": 1}]
        with self.assertRaises(ValueError):
            bm25_search("", chunks, 5)
            
        with self.assertRaises(ValueError):
            bm25_search("   ", chunks, 5)
            
        with self.assertRaises(ValueError):
            bm25_search(", , ,", chunks, 5) # tokenizer will yield empty tokens

    def test_tie_break_deterministic(self):
        question = "vay vốn"
        # C1 and C2 have exact same text, so BM25 score will be identical
        # Need 3 chunks so score is not 0
        chunks = [
            {"chunk_id": "C2", "text": "quy định vay vốn", "source": "B", "page_start": 1, "page_end": 1},
            {"chunk_id": "C1", "text": "quy định vay vốn", "source": "A", "page_start": 1, "page_end": 1},
            {"chunk_id": "C3", "text": "không liên quan", "source": "C", "page_start": 1, "page_end": 1},
            {"chunk_id": "C4", "text": "không liên quan", "source": "D", "page_start": 1, "page_end": 1},
            {"chunk_id": "C5", "text": "không liên quan", "source": "E", "page_start": 1, "page_end": 1}
        ]
        results = bm25_search(question, chunks, candidate_k=5)
        # Verify C1 and C2 are top 2 and their scores are identical
        self.assertEqual(results[0]["bm25_score"], results[1]["bm25_score"])
        self.assertEqual(results[0]["chunk_id"], "C1")
        self.assertEqual(results[1]["chunk_id"], "C2")

    def test_no_external_calls(self):
        # We can just check that a simple search runs without failing
        # If it called gemini or reranker, it would fail without API keys / model loaded
        question = "test no external"
        chunks = [{"chunk_id": "1", "text": "test no external calls", "source": "", "page_start": 1, "page_end": 1}]
        results = bm25_search(question, chunks, candidate_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "1")

if __name__ == '__main__':
    unittest.main()
