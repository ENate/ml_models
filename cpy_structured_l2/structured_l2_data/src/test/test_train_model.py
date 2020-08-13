import unittest
import numpy as np
import tensorflow as tf
from ..models.train_model import TrainingModel
from ..models.algorithm_run_file import main_run_file

 
class TestTrainingModel(unittest.TestCase):

    def test_training_model_function(self):
        x_input = np.random.randint(1)
        x_pred = TrainingModel().training_model_function(x_input)
        self.assertEqual(x_pred, 0)

    def test_batch_size(self):
        x_batch = 128
        x_input = np.random.randint(1)
        batch_value = TrainingModel().batch_sizes(x_batch)
        self.assertEqual(batch_value, 128)

    def test_build_mlp_structure(self):
        kwarg_dict = {'mlp_hid_structure': [5, 3], 'n_classes': 3, 'num_features': 4}
        m = TrainingModel()
        neurons_count, sizes_wb, shapes_wb = m.build_mlp_structure(kwarg_dict)
        print(shapes_wb)
        self.assertEqual(neurons_count, 55)
        self.assertEqual(len(sizes_wb), 6)
        self.assertEqual(len(shapes_wb), 6)

    def test_func_structured_l2pen_classifier(self):
        wb_matrix, b_matrix = tf.truncated_normal([4, 3]), tf.truncated_normal([1, 3])
        obj_func = TrainingModel()
        gen_sum_params = obj_func.func_structured_l2pen_classifier(ws_matrix=wb_matrix, bs_matrix=b_matrix)
        # self.assertEqual(gen_sum_params, tf.tensor)

    def test_train_tf_classifier(self):
        train_class = TrainingModel()
        val_try = np.random.rand()
        # correct_predictions = train_class.train_tf_classifier(val_try)
        # self.assertEqual(correct_predictions, val_try + 4)

    def test_main_run_file(self):
        flag_value = main_run_file(choose_flag=1)
        self.assertEqual(flag_value, True)

    #def test_func_cross_entropy_loss(self):
    #    t_class = TrainingModel()
    #    wb_sizes_classes, p_values, kwarg_dict = np.random.rand(), np.random.rand(), np.random.rand()
    #    loss = t_class.func_cross_entropy_loss(wb_sizes_classes, p_values, kwarg_dict)
