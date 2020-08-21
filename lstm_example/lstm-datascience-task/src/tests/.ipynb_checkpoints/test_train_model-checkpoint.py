import unittest
import numpy as np
import tensorflow as tf
from ..models.train_model import ModelTraining


class ModelTrainingTest(unittest.TestCase):

    def test_batch_size(self):
        self.batch_num = 10
        obj_model_train_bs = ModelTraining()
        self.batch_num = obj_model_train_bs.batch_size(self.batch_num)
        self.assertEqual(self.batch_num, 20)

    def test_number_epochs(self):
        steps_to_run = 100
        # create class object for number_epochs in ModelTraining class:
        obj_model_train = ModelTraining()
        num_of_epochs = obj_model_train.number_epochs(steps_to_run)
        self.num_steps = 50
        self.assertEqual(self.num_steps, num_of_epochs)

    def test_training_params(self):
        # check if it returns a float, integer and other useful training parameters
        input_x = 9
        self.assertEqual(ModelTraining().training_params(input_x), 0)

    def test_create_LSTM_model(self):
        in_features, out_labels, n_sizes, nsteps, munits = 4, 2, 1, 9, 8
        net_model = ModelTraining().create_lstm_model(in_features, out_labels, n_sizes, nsteps, munits)
        self.assertEqual(net_model, 3)

    def test_loss_function(self):
        num_feats, num_hidden, num_classes, num_time = 9, 3, 2, 10
        self.in_tensor = tf.placeholder(tf.float32, (None, 3))
        labels = tf.placeholder(tf.int32, (None, 1))
        self.model = ModelTraining()  # self.in_tensor, labels)
        sess = tf.Session()
        loss = sess.run(self.model.loss, feed_dict={self.in_tensor: np.ones(1, 3), labels: [[1]]})
        prediction = self.model.create_lstm_model(input, num_feats, num_hidden, num_classes, num_time)
        # Prediction size is (None, 1).
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction, labels=labels)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        assert loss != 0

    def test_training_function(self):
        self.model = ModelTraining()
        sess = tf.Session()
        gen_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='gen')
        des_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='des')
        before_gen = sess.run(gen_vars)
        before_des = sess.run(des_vars)
        # Train the generator.
        sess.run(self.model.train_gen)
        after_gen = sess.run(gen_vars)
        after_des = sess.run(des_vars)
        # Make sure the generator variables changed.
        for b, a in zip(before_gen, after_gen):
            assert (a != b).any()
        # Make sure discriminator did NOT change.
        for b, a in zip(before_des, after_des):
            assert (a == b).all()


if __name__ == '__main__':
    unittest.main()
