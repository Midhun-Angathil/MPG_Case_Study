import unittest
import os
from rag.faiss_utils import get_dir_size_mb

class TestFAISSUtils(unittest.TestCase):
    def test_get_dir_size_mb(self):
        os.makedirs("test_dir", exist_ok=True)
        with open("test_dir/test.txt", "w") as f:
            f.write("a" * 1024 * 10)  # 10 KB
        size = get_dir_size_mb("test_dir")
        self.assertTrue(size > 0)
        os.remove("test_dir/test.txt")
        os.rmdir("test_dir")

if __name__ == "__main__":
    unittest.main()