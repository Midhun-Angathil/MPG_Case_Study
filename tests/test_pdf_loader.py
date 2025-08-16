import unittest
from unittest.mock import patch, MagicMock
from rag.pdf_loader import load_and_split_pdfs

class DummyFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content
    def getvalue(self):
        return self._content

class TestPDFLoader(unittest.TestCase):
    @patch("rag.pdf_loader.PyPDFLoader")
    @patch("rag.pdf_loader.RecursiveCharacterTextSplitter")
    def test_load_and_split_pdfs(self, mock_splitter, mock_loader):
        dummy_pdf = DummyFile("test.pdf", b"dummy content")
        mock_loader.return_value.load.return_value = ["doc1", "doc2"]
        mock_splitter.return_value.split_documents.return_value = ["chunk1", "chunk2"]
        chunks = load_and_split_pdfs([dummy_pdf])
        self.assertEqual(chunks, ["chunk1", "chunk2"])
        mock_loader.assert_called()
        mock_splitter.assert_called()

if __name__ == "__main__":
    unittest.main()