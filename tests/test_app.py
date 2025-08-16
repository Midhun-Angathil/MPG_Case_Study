import unittest

class TestApp(unittest.TestCase):
    def test_import_app(self):
        try:
            import app
        except Exception as e:
            self.fail(f"Importing app.py failed: {e}")

if __name__ == "__main__":
    unittest.main()