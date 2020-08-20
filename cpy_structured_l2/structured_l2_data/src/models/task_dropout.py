#!/usr/bin/env python
"""This file trains the model upon the training data and evaluates it with the test data.
It uses the arguments it got via the gcloud command."""
import argparse
import os
import sys
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from . import model_dropout
sys.path.append("../data/")
import make_dataset


def training_model(params, data_location):
    """The function gets the training data from the training folder,
    the evaluation data from the test folder and trains your solution from the model.py file with it."""
    train_data, _, train_labels, _ = make_dataset.wcds_preprocess(data_location)
    print(train_data.shape)
    print(train_labels.shape)

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels,
                                                                  batch_size=model_dropout.get_batch_size(),
                                                                  num_epochs=None, shuffle=True)

    _, eval_data, _, eval_labels = make_dataset.wcds_preprocess(data_location)

    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1,
                                                                 shuffle=False)

    estimator = tf.compat.v1.estimator.Estimator(model_fn=model_dropout.solution)

    steps_per_eval = int(model_dropout.get_training_steps() / params.eval_steps)

    for _ in range(params.eval_steps):
        estimator.train(train_input_fn, steps=steps_per_eval)
        estimator.evaluate(eval_input_fn)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=1,
        type=int
    )

    ARGS = PARSER.parse_args()
    tf.compat.v1.logging.set_verbosity('INFO')
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__['INFO'] / 10)

    HPARAMS = hparam.HParams(**ARGS.__dict__)
    data_wcds = '/home/to-data/data.csv'
    training_model(HPARAMS, data_wcds)
