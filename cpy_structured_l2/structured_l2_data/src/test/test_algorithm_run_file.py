import unittest
from ..models.algorithm_run_file import main_run_file


class TestAlgorithmRunFile(unittest.TestCase):
    def test_main_run_file(self):
        flag_value = main_run_file(choose_flag=1)
        self.assertEqual(flag_value, True)