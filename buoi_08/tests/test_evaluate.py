import unittest
import math
from evaluate import recall_at_k, mrr_at_k, ndcg_at_k

class TestEvaluate(unittest.TestCase):
    def test_recall_at_k(self):
        retrieved = ["A", "B", "C", "D"]
        relevant = ["B", "C", "E"]
        
        # k = 2: retrieved = ["A", "B"], intersection = {"B"} (1)
        # relevant total = 3. recall = 1/3 = 0.333
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 2), 1.0 / 3.0)
        
        # k = 4: retrieved = ["A", "B", "C", "D"], intersection = {"B", "C"} (2)
        # recall = 2/3 = 0.666
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 4), 2.0 / 3.0)
        
        # no relevant
        self.assertEqual(recall_at_k(retrieved, [], 2), 0.0)

    def test_mrr_at_k(self):
        retrieved = ["A", "B", "C", "D"]
        relevant = ["C", "E"]
        
        # k = 2: ["A", "B"] -> no relevant found -> 0
        self.assertEqual(mrr_at_k(retrieved, relevant, 2), 0.0)
        
        # k = 4: ["A", "B", "C", "D"] -> first relevant is "C" at index 2 (rank 3) -> 1/3
        self.assertAlmostEqual(mrr_at_k(retrieved, relevant, 4), 1.0 / 3.0)
        
        # first rank hit
        self.assertAlmostEqual(mrr_at_k(["C"], relevant, 1), 1.0)
        
    def test_ndcg_at_k(self):
        retrieved = ["A", "B", "C", "D"]
        relevant = ["B", "D", "E"]
        
        # k = 4
        # ranks: A=1, B=2, C=3, D=4
        # DCG = rel(B)/log2(2+1) + rel(D)/log2(4+1) wait, indexing starts at 0 for me
        # index i: B is i=1 -> log2(1+2)=log2(3). D is i=3 -> log2(3+2)=log2(5)
        # DCG = 1/1.585 + 1/2.321 = 0.6309 + 0.4306 = 1.0615
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
        
        # IDCG (ideal): rel elements ranked at top: B, D, E at i=0, 1, 2
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5 = 2.1309
        idcg = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        
        expected_ndcg = dcg / idcg
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, 4), expected_ndcg)

if __name__ == "__main__":
    unittest.main()
