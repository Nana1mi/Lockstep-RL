import unittest

from lockstep_rl.data import validate_splits
from lockstep_rl.rewards import extract_answer, numeric_reward, simple_tasks
from lockstep_rl.rewards import reward_details


class RewardTest(unittest.TestCase):
    def test_lenient_formats_and_explicit_failure_reasons(self):
        for text, answer in ((r"\boxed{\$72}", "72"), ("#### 18.", "18"),
                             (r"\boxed{1{,}200}", "1200"), (r"\boxed{72\text{ dollars}}", "72")):
            self.assertEqual(numeric_reward(text, answer, mode="strict"), 0.)
            parsed = reward_details(text, answer, mode="lenient")
            self.assertEqual(parsed["reward"], 1.)
            self.assertEqual(parsed["parse_status"], "ok")
        self.assertEqual(reward_details("reason only", "1")["parse_status"], "no_answer")
        self.assertEqual(reward_details("<answer>wrong</answer>", "1")["parse_status"], "unparseable")
        self.assertEqual(reward_details("<answer>1", "1")["parse_status"], "unparseable")
        self.assertEqual(reward_details("<answer>2</answer>", "1")["parse_status"], "ok")
        self.assertEqual(numeric_reward(r"\boxed{1,,200}", "1200", "lenient"), 0.)
        validate_splits(simple_tasks(count=512), simple_tasks("validation", 128))

    def test_numeric_formats_missing_answers_and_truncation(self):
        for text in ("<answer>1,234</answer>", "reason\n#### 1234", r"\boxed{1234}"):
            self.assertEqual(numeric_reward(text, "1234"), 1.)
        self.assertEqual(numeric_reward("I guess 1234", "1234"), 0.)
        self.assertEqual(numeric_reward("<answer>123", "1234"), 0.)
        self.assertEqual(numeric_reward("<answer>1.25</answer>", "1.250"), 1.)
        self.assertEqual(numeric_reward("<answer>2</answer> then <answer>3</answer>", "3"), 1.)
        self.assertIsNone(extract_answer("#### no answer"))
        self.assertEqual(numeric_reward("<answer>1,,234</answer>", "1234"), 0.)
        self.assertEqual(numeric_reward("<answer>1234</answer><answer>unknown</answer>", "1234"), 0.)

    def test_disjoint_validation_and_leakage_rejection(self):
        train, validation = simple_tasks(), simple_tasks("validation")
        validate_splits(train, validation)
        with self.assertRaises(ValueError):
            validate_splits(train, train)
        with self.assertRaises(ValueError):
            validate_splits(train, [{**train[0], "id": "another"}])


if __name__ == "__main__":
    unittest.main()
