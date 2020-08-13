import numpy as np
import unittest
from ..models.func_cond import func_compute_cond


class TestFuncCond(unittest.TestCase):

    def test_func_compute_cond(self):
        n = 10
        my_matrix = np.random.rand(n, 3)
        m = func_compute_cond(my_matrix)
        self.assertEqual(m, n)
