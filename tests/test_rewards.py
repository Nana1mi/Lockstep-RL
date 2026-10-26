import unittest

from lockstep_rl.data import validate_splits
from lockstep_rl.rewards import extract_answer, numeric_reward, simple_tasks


class RewardTest(unittest.TestCase):
    def test_numeric_formats_missing_answers_and_truncation(self):
        for text in ("<answer>1,234</answer>", "reason\n#### 1234", r"\boxed{1234}"):
            self.assertEqual(numeric_reward(text, "1234"), 1.)
        self.assertEqual(numeric_reward("I guess 1234", "1234"), 0.)
        self.assertEqual(numeric_reward("<answer>123", "1234"), 0.)
        self.assertEqual(numeric_reward("<answer>1.25</answer>", "1.250"), 1.)
        self.assertEqual(numeric_reward("<answer>2</answer> then <answer>3</answer>", "3"), 1.)
        self.assertIsNone(extract_answer("#### no answer"))

    def test_disjoint_validation_and_leakage_rejection(self):
        train, validation = simple_tasks(), simple_tasks("validation")
        validate_splits(train, validation)
        with self.assertRaises(ValueError):
            validate_splits(train, train)
        with self.assertRaises(ValueError):
            validate_splits(train, [{**train[0], "id": "another"}])


if __name__ == "__main__":
    unittest.main()
