# Add this line to the beginning of relative.py file
import os
import sys
sys.path.append('../data/')  # relative import of files to prepare data using the 'make_dataset.py' file and features
sys.path.append('../features/')
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from make_dataset import make_data_set
from build_features import generate_feats_and_labels
from .predict_model import ModelPredict


class ModelTraining(object):

    def __init__(self):
        self.b_size, self.epochs = None, None
        self.sum_values, self.cost = 0., None
        self.x_y_layer_values, self.batch_s = None, None
        self.n_hidden, self.n_classes = None, None
        self._x_feats, self.y_raw, self.random_seed = None, None, 0
        self.train_accuracies, self.training_data_count = None, None

    def function_batch_size(self, _train, step, batch_size):
        # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.
        shape = list(_train.shape)
        shape[0] = batch_size
        self.batch_s = np.empty(shape)

        for i in range(batch_size):
            # Loop index
            index = ((step - 1) * batch_size + i) % len(_train)
            self.batch_s[i] = _train[index]

        return self.batch_s

    def training_params(self, in_train_data, in_test_data):
        """
        Aim: To compute the input sequence, the training and test shapes from the
        formatted dataset. It returns the corresponding shapes for the training and test sets.
        :param in_train_data:
        :param in_test_data:
        :return: training_data_count, num_steps, n_input
        """
        # self.n_hidden = 32  # Hidden layer num of features
        # self.n_classes  # Total classes (should go up, or should go down)
        self.training_data_count = len(in_train_data)  # total training series (with 50% overlap between each series)
        test_data_count = len(in_test_data)  # total length of testing series
        n_steps = len(in_train_data[0])  # timesteps per series
        n_input = len(in_train_data[0][0])  # 16 input parameters per timestep
        # assert n_input != 0
        return self.training_data_count, test_data_count, n_steps, n_input

    def training_testing_data(self, reshaped_segments, en_labels):
        """
        This function is used to format the training set (with segments containing the pre-defined input sequences)
        and output labels.
        :param reshaped_segments:
        :param en_labels:
        :return: train_features, test_features, out_train_labels, out_test_labels, number_classes
        """
        number_classes = en_labels.shape[1]
        self.random_seed = 42
        in_train_features, in_test_features, out_train_labels, out_test_labels = \
            train_test_split(reshaped_segments, en_labels, test_size=0.2)  # , random_state=self.random_seed)
        return in_train_features, in_test_features, out_train_labels, out_test_labels, number_classes

    def create_lstm_rnn(self, _x, n_input, n_steps, nodes_hidden, _weights, _biases):
        """
        :param _x: input placeholder defined as a tensor
        :param n_input: number of input features
        :param n_steps: timesteps per series
        :param nodes_hidden: number of hidden units in network
        :param _weights: initial training weights
        :param _biases: initial training biases
        :return: initial_model
        """
        # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
        # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
        # Note, some code of this notebook is inspired from an slightly different
        # RNN architecture used on another dataset, some of the credits goes to
        # "aymericdamien" under the MIT license.
        self._x_feats = _x
        # (NOTE: This step could be greatly optimised by shaping the dataset once
        # input shape: (batch_size, n_steps, n_input)
        self._x_feats = tf.transpose(self._x_feats, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        self._x_feats = tf.reshape(self._x_feats, [-1, n_input])
        # new shape: (n_steps*batch_size, n_input)
        # ReLU activation, thanks to Yu Zhao for adding this improvement here:
        self._x_feats = tf.nn.relu(tf.matmul(self._x_feats, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        self._x_feats = tf.split(self._x_feats, n_steps, 0)
        # new shape: n_steps * (batch_size, n_hidden)

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(nodes_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(nodes_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        # Get LSTM cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, self._x_feats, dtype=tf.float32)

        # Get last time step's output feature for a "many-to-one" style classifier,
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]
        initial_model = tf.matmul(lstm_last_output, _weights['out']) + _biases['out']
        # Linear activation
        return initial_model

    def loss_function(self, num_inputs, num_steps, kwargs):
        """
        :param num_inputs: Number of features
        :param num_steps:
        :param kwargs: A dictionary of parameters needed to build the LSTM model
        :return: cost, optimizer, pred, accuracy, placeholders: input_holder, output_holder
        """
        # Define Learning rate and hyper-parameters
        learn_rate, lambda_loss_amount = kwargs['learning-rate'], kwargs['tuning-param']
        n_of_classes, num_of_hidden = kwargs['num-of-classes'], kwargs['nos_of_hidden']
        # Graph input/output
        input_holder = tf.placeholder(tf.float32, [None, num_steps, num_inputs])
        output_holder = tf.placeholder(tf.float32, [None, n_of_classes])

        # Graph weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([num_inputs, num_of_hidden])),  # Hidden layer weights
            'out': tf.Variable(tf.random_normal([num_of_hidden, n_of_classes], mean=1.0))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([num_of_hidden])),
            'out': tf.Variable(tf.random_normal([n_of_classes]))
        }

        pred = ModelTraining().create_lstm_rnn(input_holder, num_inputs, num_steps, num_of_hidden, weights, biases)
        # Loss, optimizer and evaluation
        l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        # L2 loss prevents this overkill neural network to over-fit the data
        # Soft max loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_holder, logits=pred)) + l2
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learn_rate).minimize(self.cost)  # Adam Optimizer
        pred_softmax = tf.nn.softmax(pred, name="y_")
        correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(output_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return self.cost, optimizer, pred_softmax, accuracy, input_holder, output_holder

    def training_function(self, training_data_count, dict_params):
        """
        :param training_data_count: The number of features after pre-processing data
        :param dict_params: A dictionary containing user defined training, testing data and other parameters
        :return: test_accuracies, test_losses, one_hot_predictions
        """
        # Training features and labels
        feats_train, train_labels = dict_params['training-features'], dict_params['labels-output']
        # testing features and labels
        input_test, output_test = dict_params['testing_features'], dict_params['labels-test']
        # Other parameters
        feats_test, n_of_classes = dict_params['testing_features'], dict_params['num-of-classes']
        # batch_size, display_iter = dict_params['batch_size'], dict_params['display_iter']
        loop_training_iteration = training_data_count * 100  # Loop 100 times on the dataset
        batch_size = 1000
        display_iter = 3000
        # To keep track of training's performance
        test_losses = []
        test_accuracies = []
        train_losses = []
        self.train_accuracies = []
        # Return the number of inputs and other essential parameters for training:
        training_data_count, test_data_count, n_of_steps, n_in_feats =\
            ModelTraining().training_params(feats_train, feats_test)
        # Compute accuracy and optimizer object
        cost, optimizer, pred, accuracy, in_holder, out_holder \
            = ModelTraining().loss_function(n_in_feats, n_of_steps, dict_params)

        # Launch the graph
        sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)

        # Perform Training steps with "batch_size" amount of example data at each loop
        step = 1
        while step * batch_size <= loop_training_iteration:
            batch_xs = ModelTraining().function_batch_size(feats_train, step, batch_size)
            batch_ys = ModelTraining().function_batch_size(train_labels, step, batch_size)
            # batch_ys = ModelTraining().one_hot(ModelTraining().batch_size(train_labels, step, batch_size), y_data)
            # Fit training using batch data
            _, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={in_holder: batch_xs, out_holder: batch_ys})
            train_losses.append(loss)
            self.train_accuracies.append(acc)

            # Choose steps to evaluate network for faster training:
            if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > loop_training_iteration):
                # Show training accuracy/loss in this "if"
                print("Training iteration number:" + str(step * batch_size) + ":   Batch Loss = " + "{:.4f}".format(loss) +
                      ", Accuracy = {}".format(acc))
                # Evaluation on the test set for diagnosis
                loss, acc = sess.run([cost, accuracy], feed_dict={in_holder: input_test, out_holder: output_test})
                # out_holder: ModelTraining().one_hot(output_test, n_of_classes)}
                test_losses.append(loss)
                test_accuracies.append(acc)
                print("ACCURACY ON TEST SET: " + "Batch Loss = {}".format(loss) + ", Accuracy = {}".format(acc))

            step += 1

        print("End of the training / Optimization")

        # Accuracy for test data

        one_hot_predictions, accuracy, final_loss = \
            sess.run([pred, accuracy, cost], feed_dict={in_holder: input_test, out_holder: output_test})

        test_losses.append(final_loss)
        test_accuracies.append(accuracy)
        ModelPredict(batch_size, loop_training_iteration, display_iter).plot_function(
            train_losses, self.train_accuracies, test_losses, test_accuracies)
        return accuracy, one_hot_predictions, loop_training_iteration


if __name__ == '__main__':
    # Try to let tensorflow keep quiet
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('INFO')
    # call the make_dataset file to create the training and labels sets
    original_data_file = '~/tasks/talpa-datascience-task/data/raw/data_case_study.csv'
    dat_with_activity, dat_without_activity, y_label_col = make_data_set(original_data_file)
    feature_num, segment_inputs, encoded_labels = generate_feats_and_labels(dat_without_activity, dat_with_activity)

    # Data files are passed
    training_features, testing_features, training_labels, testing_labels, n_classes = \
        ModelTraining().training_testing_data(segment_inputs, encoded_labels)
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=1,
        type=int
    )
    PARSER.add_argument(
        '--learning-rate',
        help='The learning rate for training',
        default=0.0025,
        type=float)
    PARSER.add_argument(
        '--reg_param',
        help='The tuning value to reduce over-fitting',
        default=0.002,
        type=float
    )
    PARSER.add_argument(
        '--n_hidden',
        help='The number of hidden nodes',
        default=32,
        type=int
    )
    PARSER.add_argument(
        '--batch_size',
        help='initial batch sizes during training and testing',
        default=1500,
        type=int
    )
    PARSER.add_argument(
        '--display_iter',
        help='Time lapse to display results during iteration',
        default=300,
        type=int
    )

    # Total classes (should go up, or should go down)
    ARGS = PARSER.parse_args()
    n_hidden = ARGS.n_hidden
    learning_rate = ARGS.learning_rate
    eval_steps = ARGS.eval_steps
    reg_param = ARGS.reg_param
    ds_display_iter = ARGS.display_iter
    ds_batch_size = ARGS.batch_size

    kwargs_dict = {'nos_of_hidden': n_hidden, 'eval-steps': eval_steps, 'learning-rate': learning_rate,
                   'training-features': training_features, 'testing_features': testing_features,
                   'tuning-param': reg_param, 'labels-output': training_labels, 'labels-test': testing_labels,
                   'num-of-classes': n_classes, 'display_iter': ds_display_iter, 'batch_size': ds_batch_size}
    training_features_count, _, _, _ = ModelTraining().training_params(training_features, testing_features)
    accuracy_val, one_hot_predict, train_iter = ModelTraining().training_function(training_features_count, kwargs_dict)
    tf.compat.v1.logging.set_verbosity(10)
    # Plot error loss, analyze accuracies and other metrics
    ModelPredict(ds_batch_size, train_iter, ds_display_iter).metrics_confusion_matrices(one_hot_predict, accuracy_val, testing_labels, n_classes)


