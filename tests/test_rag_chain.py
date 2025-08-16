import unittest
from unittest.mock import MagicMock
from rag.rag_chain import build_rag_chain

class TestRAGChain(unittest.TestCase):
    def test_build_rag_chain(self):
        llm = MagicMock()
        retriever = MagicMock()
        system_prompt = "You are a helpful assistant."
        contextualize_q_system_prompt = "Contextualize the question."
        rag_chain = build_rag_chain(llm, retriever, system_prompt, contextualize_q_system_prompt)
        self.assertIsNotNone(rag_chain)

if __name__ == "__main__":
    unittest.main()