import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from repair_sft_answers import extract_balanced_boxed


class ExtractBalancedBoxedTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(extract_balanced_boxed(r"answer \boxed{A}"), ["A"])

    def test_nested_fraction(self):
        self.assertEqual(
            extract_balanced_boxed(r"\boxed{-\dfrac{3}{4}}"),
            [r"-\dfrac{3}{4}"],
        )

    def test_multiple_uses_last_at_call_site(self):
        self.assertEqual(extract_balanced_boxed(r"\boxed{1}, then \boxed{2}"), ["1", "2"])

    def test_unbalanced_is_ignored(self):
        self.assertEqual(extract_balanced_boxed(r"\boxed{\frac{1}{2}"), [])

    def test_no_box(self):
        self.assertEqual(extract_balanced_boxed("The answer is 42"), [])


if __name__ == "__main__":
    unittest.main()
