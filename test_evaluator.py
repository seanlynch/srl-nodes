import unittest

from evaluator import evaluate


class TestEvaluator(unittest.TestCase):
    def test_format_access(self):
        with self.assertRaises(NotImplementedError):
            evaluate("(lambda x: f'{x.__class__}')(1)")

    def test_fstrings(self):
        s = {"x": 5}
        r = evaluate("f'{x}'", s)
        self.assertEqual(r, "5")
        r = evaluate("f'{x:05}'", s)
        self.assertEqual(r, "00005")

    def test_lambda(self):
        r = evaluate("(lambda x, y: x + y)(2, 3)")
        self.assertEqual(r, 5)


if __name__ == "__main__":
    unittest.main()
