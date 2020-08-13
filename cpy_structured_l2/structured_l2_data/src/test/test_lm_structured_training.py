import unittest
import numpy as np
from ..models.lm_structured_training import func_classifier_l2l1


class TestLMAdaptedTraining(unittest.TestCase):

    def test_func_classifier_l2l1(self):
        rnd_num_x = np.random.rand()
        rnd_num_y = np.random.rand()
        k_dict = {'mlp_hid_structure': [5, 3], 'n_classes': 3, 'num_features': 4}
        classify = func_classifier_l2l1(rnd_num_x, rnd_num_y, k_dict)
        self.assertEqual(classify, rnd_num_x * 4)