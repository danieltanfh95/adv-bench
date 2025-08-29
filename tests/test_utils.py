import unittest
from base_provider import load_prompt, compute_prompt_hash

class UtilsTest(unittest.TestCase):
    def test_load_prompt(self):
        p = load_prompt()
        self.assertIsInstance(p, str)
        self.assertGreater(len(p), 0)

    def test_compute_prompt_hash(self):
        s = "hello"
        h = compute_prompt_hash(s)
        self.assertEqual(len(h), 64)

if __name__ == '__main__':
    unittest.main()
