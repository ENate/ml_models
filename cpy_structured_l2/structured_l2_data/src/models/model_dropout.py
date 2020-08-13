import os
__path__ = [os.path.dirname(os.path.abspath(__file__))]
import tensorflow as tf
from .train_model import TrainingModel
from .algorithm_run_file import RunAllAlgorithms


def get_batch_size():
    return 20


def get_training_steps():
    return 20


def solution(features, labels, mode):
    _NUM_CLASSES, choose_type = 2, 1
    print(tf.shape(features))
    _LEARNING_RATE, n = 0.0007, 31 # features.shape[1]
    global_step = tf.train.get_global_step()
    k_parsers = RunAllAlgorithms().create_args_function()
    neurons_cnt, wb_shapes, wb_sizes, nhidden = TrainingModel().build_mlp_structure(n, k_parsers, _NUM_CLASSES)
    build_kwargs = {'n': n, 'nhidden': nhidden, 'wb_sizes': wb_sizes, 'wb_shapes': wb_shapes}
    params0 = tf.Variable(k_parsers.initializer([neurons_cnt], dtype=tf.float64))
    logits = TrainingModel().func_mse_loss(build_kwargs, params0)
    """Returns an EstimatorSpec that is constructed using the solution that you have to write below."""
    # TODO: Code of your solution
    predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    probabilities = tf.nn.softmax(logits)
    predictions = {"predicted_logit": predicted_logit, "probabilities": probabilities}
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



