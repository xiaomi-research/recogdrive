import unittest

from internvl_chat.tools.reasoning_data_pipeline.utils.accuracy_reward import (
    math_score,
)


class MathScoreTest(unittest.TestCase):
    def test_equivalent_zero_values_are_equal(self):
        self.assertTrue(math_score("0.0", "0"))

    def test_different_value_is_not_equal_to_zero(self):
        self.assertFalse(math_score("1", "0"))

    def test_equivalent_non_zero_values_are_still_equal(self):
        self.assertTrue(math_score("1.0", "1"))


if __name__ == "__main__":
    unittest.main()
