#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from audioop import cross
NUM_CLASSES = 4


def get_training_steps():
    """Returns the number of batches that will be used to train your solution.
    It is recommended to change this value."""
    return 400


def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value."""
    return 50  # 50 # 80


def solution(features, labels, mode):
    _NUM_CLASSES = 4
    _LEARNING_RATE = 0.0007
    global_step = tf.train.get_global_step()
    """Returns an EstimatorSpec that is constructed using the solution that you have to write below."""
    # Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])
    net = tf.layers.conv2d(input_layer, 128, [10, 10], activation=tf.nn.relu, name='conv1')
    net = tf.layers.max_pooling2d(net, [10, 10], 5, name='pool1')
    net = tf.layers.conv2d(net, 64, [5, 5], activation=tf.nn.relu, name='conv2')
    dropout1 = tf.layers.dropout(net, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))
    net2 = tf.layers.max_pooling2d(dropout1, [5, 5], 5, name='pool2')
    net3 = tf.layers.flatten(net2)
    dropoutx = tf.layers.dropout(net3, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(dropoutx, _NUM_CLASSES, activation=None, name='fc1')
    # TODO: Code of your solution
    predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    probabilities = tf.nn.softmax(logits)
    predictions = {"predicted_logit": predicted_logit, "probabilities": probabilities}
    # with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope='loss')
    tf.summary.scalar('loss', cross_entropy)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_logit, name='acc')
    tf.summary.scalar('accuracy', accuracy[1])
    optimizer = tf.train.AdamOptimizer(learning_rate=_LEARNING_RATE)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: return tf.estimator.EstimatorSpec with prediction values of all classes
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO: Let the model train here
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # The classes variable below exists of an tensor that contains all the predicted classes in a batch
        # TODO: eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)}
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops={'accuracy/accuracy': accuracy}, evaluation_hooks=None)


def solution0(features, labels, mode):
    #  This is the function which describes the structure of the neural network

    # input layer
    # reshaping x to 4-D tensor: [batch_size, width, height, channels]
    # features['x'] - the dictionary we passed for x in the input functions
    layer_1 = tf.reshape(features['x'], [-1, 64, 64, 3])

    # convolution layer 1
    # computes 32 features using 10x10 filter with ReLU activation.
    # input tensor: [batch_size, 28, 28, 1]
    # output tensor: [batch_size, 28, 28, 32]
    layer_2 = tf.layers.conv2d(inputs=layer_1, filters=32, kernel_size=[10, 10], padding="same", activation=tf.nn.relu)

    # convolution layer 2
    # computes 32 features using 5x5 filter with ReLU activation.
    # input tensor: [batch_size, 28, 28, 32]
    # output tensor: [batch_size, 28, 28, 64]
    layer_3 = tf.layers.conv2d(inputs=layer_2, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 28, 28, 64]
    # Output Tensor Shape: [batch_size, 28 * 28 * 64]
    # layer_4 = tf.reshape(layer_3, [-1, 28 * 28 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 28 * 28 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    layer_5 = tf.layers.dense(inputs=layer_3, units=4, activation=tf.nn.relu)

    # Dropout operation; 0.6 probability that element will be kept
    # notice that this layer will perform droupout only during training!
    layer_6 = tf.layers.dropout(inputs=layer_5, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=layer_6, units=4)

    # define the values which our neural network will output
    # classes - which number the NN 'thinks' is on the image
    # probabilities - how certain our NN is about its prediction
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # here we define what happens if we call the predict method of our estimator
    # with the current settings it will return the dictionary defined above
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # here we define the loss for our training (the thing we minimize)
    # I do not need to perform one-hot-encoding to my training labels because the method
    # sparse_softmax_cross_entropy will do that for me and I don't need to think about that
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # here we define how we calculate our accuracy
    # if you want to monitor your training accuracy you need these two lines
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'], name='acc_op')
    tf.summary.scalar('accuracy', accuracy[1])

    # here we define what happens if we call the train method of our estimator
    # with its current settings it will adjust the weights and biases of our neurons
    # using the Adam Optimization Algorithm based on the loss function we defined earlier
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # what evaluation metric we want to show
    eval_metric_ops = {"accuracy": accuracy}

    # here we define what happens if we call the evaluate method of our estimator
    # with its current settings it will display the loss and the accuracy which we defined earlier
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
