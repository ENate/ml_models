import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense


N = 500  # Input size
H = 100  # Hidden layer size
O = 10   # Output size

w1 = np.random.randn(N, H)
b1 = np.random.randn(H)

w2 = np.random.randn(H, O)
b2 = np.random.randn(O)

""" Keras implementation
"""


def ffpass_np(x):
    """ 
    Implementing the keras
    @param: x
    @return
    """
    a1 = np.dot(x, w1) + b1  # affine
    r = np.maximum(0, a1)  # ReLU
    a2 = np.dot(r, w2) + b2  # affine

  exps = np.exp(a2 - np.max(a2))  # softmax
  out = exps / exps.sum()
  return out


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

model = tf.keras.Sequential()
model.add(Dense(H, activation='relu', use_bias=True, input_dim=N))
model.add(Dense(O, activation='softmax', use_bias=True, input_dim=O))
model.get_layer(index=0).set_weights([w1, b1])
model.get_layer(index=1).set_weights([w2, b2])


def ffpass_tf(x):
  xr = x.reshape((1, x.size))
  return model.predict(xr)[0]


def jacobian_tensorflow(x, verbose=False):
  jacobian_matrix = []
  it = tqdm(range(O)) if verbose else range(O)
  for o in it:
    grad_func = tf.gradients(model.output[:, o], model.input)
    gradients = sess.run(grad_func, feed_dict={model.input: x.reshape((1, x.size))})
    jacobian_matrix.append(gradients[0][0, :])

  return np.array(jacobian_matrix)



