import unittest
from rag.chat_history import get_session_history
import streamlit as st

class TestChatHistory(unittest.TestCase):
    def test_get_session_history(self):
        session_id = "test_session"
        history = get_session_history(session_id)
        self.assertIsNotNone(history)
        history2 = get_session_history(session_id)
        self.assertIs(history, history2)

if __name__ == "__main__":
    unittest.main()