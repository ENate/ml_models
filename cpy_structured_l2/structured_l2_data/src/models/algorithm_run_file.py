import os
import sys
sys.path.append("../data/")
sys.path.append('../features/')
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from .results_classifier import func_prediction_analysis
from .lm_structured_training import model_l1_l2_func, func_classifier_l2l1, train_classifier_sgd, train_tf_classifier
from .train_model import TrainingModel
from make_dataset import process_dataset_func, wcds_preprocess
from build_features import BuildFeatures
from .LMAlgorithmImpl import func_pred


SEED = 42
# variants of initializers
INITIALIZERS = {'xavier': tf.contrib.layers.xavier_initializer(seed=SEED),
                'rand_uniform': tf.random_uniform_initializer(seed=SEED),
                'rand_normal': tf.random_normal_initializer(seed=SEED)}

# how frequently log is written and checkpoint saved
LOG_INTERVAL_IN_SEC = 0.05
# variants of tensor flow built-in optimizers
TF_OPTIMIZERS = {'sgd': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer}
step_delta, time_delta = 0, 0


class RunAllAlgorithms(object):
    def __init__(self):
        self.h_params, self.initializer = None, None

    def create_args_function(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--eval_steps', help='Number of steps to run for at each checkpoint', default=1, type=int)
        parser.add_argument('-a', '--activation', help='nonlinear activation function', type=str,
                            choices=['relu', 'sigmoid', 'tanh'], default='tanh')
        parser.add_argument('--num_epochs', help='Number of epochs for iteration', default=100, type=int)
        parser.add_argument('--batch_size', help='Number of batch sizes', default=40, type=int)
        parser.add_argument('--mlp_hid_structure', help='Number of hidden layers for MLP', default=[5, 3], type=object)
        parser.add_argument('--n_classes', help='Number of classes for output labels', default=3, type=int)
        parser.add_argument('--num_features', help='Number of input features', default=4, type=int)
        parser.add_argument('--optimizer', help='optimization algorithms', type=str, choices=['sgd', 'adam', 'lm'],
                            default='sgd')
        parser.add_argument('--initializer', help='trainable variables initializer', type=str,
                            choices=['rand_normal', 'rand_uniform', 'xavier'], default='xavier')
        parser.add_argument('--step_delta_n', help='Delta time change', default=0, type=float)
        parser.add_argument('--time_delta_n', help='Delta time change', default=0, type=float)
        parser.add_argument('--choose_flag', help='Choice of algorithm to run', default=1, type=int)
        parser.add_argument('--LOG_INTERVAL_IN_SEC', help='Time interval to print results', default=0.05, type=float)
        arguments = parser.parse_args()
        self.initializer = INITIALIZERS[arguments.initializer]
        return arguments, self.initializer


def main_run_file(choose_flag, wcds_data):
    opt_obj = tf.train.GradientDescentOptimizer(learning_rate=1)
    # Compute the number of neurons, shapes and sizes of each parameter set
    # x_input_size, x_in_features, y_labels_output = process_dataset_func(wcds_data)
    # Compute the number of neurons in entire network, shapes, sizes of the underlying network in each layers
    # xtrain, xtest, ytrain, ytest = BuildFeatures().func_cancer_data_features(x_in_features, y_labels_output)
    xtrain, xtest, ytrain, ytest = wcds_preprocess(wcds_data)
    x_points, n_feats, nclasses = xtrain.shape[0], xtrain.shape[1], ytrain.shape[1]
    hyper_params, initializer = RunAllAlgorithms().create_args_function()
    num_neurons, num_shapes, num_sizes, nn_hidden = TrainingModel().build_mlp_structure(n_feats, hyper_params, nclasses)

    second_kwarg = {'xtest': xtest, 'ytest': ytest, 'wb_sizes': num_sizes, 'xydata': [xtest, ytest],
                    'wb_shapes': num_shapes, 'num_neurons': num_neurons, 'xtr': xtrain, 'ytr': ytrain,
                    'sess': tf.Session(), 'lambda1_vec': np.array([0.01]), 'lambda2_vec': np.array([0.004]),
                    'initializer': initializer, 'n': n_feats, 'nhidden': nn_hidden, 'y_n': ytest.shape[1],
                    'nclasses': nclasses, 'opt_obj': opt_obj, 'choose_flag': choose_flag, 'reg_param': 1}
    print(second_kwarg['num_neurons'])

    lst_nonzero = []
    lst_nonzero2 = []
    lst_err_l1 = []
    lst_err_l2 = []
    if hyper_params.optimizer == 'lm':
        if choose_flag == 1:
            restore_param, correct_predict = func_classifier_l2l1(second_kwarg, x_points, choose_flag, hyper_params)
            p_theta = np.asarray(restore_param)
            func_prediction_analysis(correct_predict, ytest)
        else:
            restore_param, opt_new_p, y_model, n_loss, no_zero = model_l1_l2_func(x_points, n_feats,
                                                                                  second_kwarg, hyper_params)
            lst_err_l1.append(n_loss)
            lst_nonzero.append(no_zero)
            lst_err_l2.append(lst_err_l1)
            lst_nonzero2.append(lst_nonzero)
            nm_train = xtrain.shape[0]
            y_hat_flat, x = func_pred(n_feats, nn_hidden, restore_param, second_kwarg)
            y_labeled = second_kwarg['sess'].run(y_hat_flat, feed_dict={x: xtrain})
            f1 = plt.figure(1)
            colors = np.random.rand(nm_train)
            area = (10 * np.random.rand(nm_train)) ** 2
            plt.scatter(y_labeled, ytrain, s=area, c=colors, alpha=0.5)
            f1.suptitle('Predicting Artificial Data from Model', fontsize=14, fontweight='bold')
            plt.xlabel('Model', fontsize=14, fontweight='bold')
            plt.ylabel('Data', fontsize=14, fontweight='bold')
            f1.savefig('/home/nath/finalResults/classifiers/scatter_testData.pdf')
            # test set ##########################################################
            # ####################################################################
            nm_test = xtest.shape[0]
            colors_t = np.random.rand(nm_test)
            area_t = (10 * np.random.rand(nm_test)) ** 2
            g1 = plt.figure(2)
            y_hf1, x = func_pred(n_feats, nn_hidden, opt_new_p, second_kwarg)
    else:
        # restore_param = train_classifier_sgd(TF_OPTIMIZERS, second_kwarg, hyper_params)
        restore_param = train_tf_classifier(TF_OPTIMIZERS, second_kwarg, hyper_params)

    return restore_param


if __name__ == '__main__':
    """ Enter file Name and call functions to start training """
    run_flag = 1
    # wcds_data_file = "/home/nath/Desktop/dec2019/0412_folder/saved_latest/data/data.csv"
    wcds_data_file = '/pathtodata/data.csv'
    opt_theta = main_run_file(run_flag, wcds_data_file)
    # Pass data to model building function to compute the loss and other values
    # Pass the loss to training function and start training
    # Pass results and compute metrics and accuracy
