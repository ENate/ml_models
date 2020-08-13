import unittest
import numpy as np
from ..features.build_features import BuildFeatures


class TestBuildFeatures(unittest.TestCase):

    def test_build_features_function(self):
        x_value = np.random.randint(1)
        x_pred = BuildFeatures().build_features_function(x_value)
        self.assertEqual(x_pred, x_value * 2)
