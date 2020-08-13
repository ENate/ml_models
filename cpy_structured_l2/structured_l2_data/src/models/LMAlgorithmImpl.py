import os
import math
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
__path__ = [os.path.dirname(os.path.abspath(__file__))]


def load_solar_data(mdatfile, datmsg):
    """ This is the mat lab data function """
    mat_contents = sio.loadmat(mdatfile, struct_as_record=False)
    oct_struct = mat_contents[datmsg]
    if datmsg == 'testData':
        valdata = oct_struct[0, 0].xyvalues
        # valdata = (valdata - valdata.min(0)) / valdata.ptp(0)
        x_data = valdata[:, 0:-1]
        y_data = valdata[:, -1]
    
    else:
        # datmsg == 'solardatanorm':
        valdata = oct_struct[0, 0].values
        x_data = valdata[:, 0:-1]
        y_data = valdata[:, -1]
    x_train, x_test, y_train_set, y_test_set = train_test_split(x_data, y_data, test_size=0.20, shuffle=False)
    return x_train, x_test, y_train_set, y_test_set


def data_to_train0(xtrain0, ytr0, xt0):
    """ Format the data from the mat lab function """
    xtrain00 = np.asanyarray(xtrain0)
    biases = np.ones((xtrain00.shape[0], 1))
    xtr = np.c_[xtrain00, biases]
    # For the sake of testing
    xt00 = np.asanyarray(xt0)
    biases = np.ones((xt00.shape[0], 1))
    xt1 = np.c_[xt00, biases]
    ytr = ytr0[:, None]
    n = xtrain00.shape[1]
    Nm = xtr.shape[0]
    return Nm, n, xtrain00, xt00, ytr


def build_network_model(n, nn_hidden):
    # nn = [15]
    st = [n] + nn_hidden + [1]
    shapes = []
    for i in range(len(nn_hidden) + 1):
        shapes.append((st[i], st[i + 1]))
        shapes.append((1, st[i + 1]))
    sizes = [h * w for h, w in shapes]
    neurons_cnt = sum(sizes)
    return shapes, sizes, neurons_cnt, nn_hidden


def activation_func():
    activation = tf.nn.sigmoid
    return activation


def func_pred(n_inputs, nn_0, p, kwargs):
    shapes_vec, sizes_vec = kwargs['shapes'], kwargs['sizes']
    activation = kwargs['activation']
    # placeholder variables (we have m data points)
    x_inputs = tf.placeholder(tf.float64, shape=[None, n_inputs])
    parameter_vector = tf.split(p, sizes_vec, 0)
    for i in range(len(parameter_vector)):
        parameter_vector[i] = tf.reshape(parameter_vector[i], shapes_vec[i])
    ws = parameter_vector[0:][::2]
    bs = parameter_vector[1:][::2]
    y_hat = x_inputs
    for i in range(len(nn_0)):
        y_hat = activation(tf.matmul(y_hat, ws[i]) + bs[i])
    y_hat = tf.matmul(y_hat, ws[-1]) + bs[-1]
    y_hat_flat_out = tf.squeeze(y_hat)
    return y_hat_flat_out, x_inputs


def train_classifier_sgd(x, y, loss, train_step, **kwargs3):
    """ classifier loss?"""
    step = 0
    batch_size = 5
    x_dat = kwargs3['xtrain']
    y_dat = kwargs3['ytrain']
    tf_train_labels = kwargs3['tf_train_labels']
    tf_train_dataset = kwargs3['tf_train_dataset']
    train_prediction = kwargs3['train_prediction']
    optimizer = kwargs3['optimizer']
    train_labels = y_dat
    train_dataset = x_dat
    feed_dict = {x: x_dat, y: y_dat}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # calc initial loss
    current_loss = session.run(loss, feed_dict)
    while current_loss > 1e-10 and step < 400:
        step += 1
        # log(step, current_loss, session.run(params))
        session.run(train_step, feed_dict)
        current_loss = session.run(loss, feed_dict)
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a mini-batch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the
        # session where to feed the mini-batch.
        # The key of the dictionary is the
        # placeholder node of the graph to be fed
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    return current_loss
