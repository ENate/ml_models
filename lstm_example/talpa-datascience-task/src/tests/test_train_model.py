import unittest
import sys
sys.path.append('../data/')  # relative import of files to make_data and features
sys.path.append('../features/')  # relative import of files to make_data and features
sys.path.append('../model')
import numpy as np
import pandas as pd
from ..models.train_model import ModelTraining


class ModelTrainingTest(unittest.TestCase):

    def test_batch_size(self):
        self.batch_num = 10
        self.step_sizes = 20
        self._train_data = np.random.rand(100, 3)
        obj_model_train_bs = ModelTraining()
        self.batch_num = obj_model_train_bs.function_batch_size(self._train_data, self.step_sizes, self.batch_num)
        self.assertEqual(self.batch_num.shape[0], 10)
        self.assertEqual(self.batch_num.shape[1], 3)

    def test_training_params(self):
        x_steps = 5
        sequnce_of_steps = 50
        # check if it returns a float, integer and other training parameters
        in_random_features = np.random.rand(1000, 5)
        in_test_features = np.random.rand(1000, 5)
        df_feats_in = pd.DataFrame({'a': in_random_features[:, 0], 'b': in_random_features[:, 1],
                                    'c': in_random_features[:, 2], 'd': in_random_features[:, 3],
                                    'e': in_random_features[:, 4]})
        segment_input_mat = []
        for input_idx in range(0, len(df_feats_in) - sequnce_of_steps, x_steps):
            input_matrix = [df_feats_in[cols].values[input_idx: input_idx + sequnce_of_steps]
                            for cols in df_feats_in.columns]
            segment_input_mat.append(input_matrix)
        count_feats, count_test_feats, new_steps, dim_input = ModelTraining().training_params(segment_input_mat,
                                                                                              in_test_features)

        self.assertEqual(new_steps, x_steps)
        self.assertEqual(dim_input, sequnce_of_steps)
        self.assertEqual(count_feats, np.array(segment_input_mat).shape[0])


if __name__ == '__main__':
    unittest.main()
