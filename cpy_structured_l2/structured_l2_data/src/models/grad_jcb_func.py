import autograd.numpy as np


def sig_activation(y_hat_vec_mat):
    sig_vector = 1 / (1 + np.exp(- y_hat_vec_mat))
    return sig_vector

# function compute bs and ws


def bs_ws_function(all_p, wb_shapes, wb_sizes):
    # print(np.size(inputs), np.size(y))
    # wb_shapes, wb_sizes = kwargs1['wb_shapes'], kwargs1['wb_sizes']
    wb_size_new = []
    sum_sizes = 0
    wb_sizes_array = [int(i) for i in wb_sizes] 
    # Format the size entries using number python package
    for k in range(len(wb_sizes_array)):
        sum_sizes += wb_sizes_array[k]
        wb_size_new.append(sum_sizes)
    # ##### calculate length of input and bias params
    # ##### Another way of doing things
    # print(type(all_p))
    l_p_values = np.split(all_p, wb_size_new) 
    lp_values = l_p_values[0:-1]
    lp_value = []
    for i in range(len(lp_values)):
        # lp_values[i] = npr.reshape(lp_values[i], wb_shapes[i])
        lp_value.append(np.reshape(l_p_values[i], wb_shapes[i]))
    ws_classify = lp_value[0:][::2]
    # print(ws_classify)
    bs_classify = lp_value[1:][::2]
    # print(bs_classify)
    return ws_classify, bs_classify


def neural_net_loss(p, kwargs1):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    inputs, y = kwargs1['x_inputs'], kwargs1['y']
    wb_sizes_x, wb_shapes = kwargs1['wb_sizes'], kwargs1['wb_shapes']
    x_wb_size_new = []
    sum_wb_sizes = 0
    wb_sizes_array = np.asarray(wb_sizes_x)

    # Format the size entries using number python package
    for k in range(len(wb_sizes_array)):
        sum_wb_sizes += wb_sizes_array[k]
        x_wb_size_new.append(sum_wb_sizes)

    # print(x_wb_size_new)
    # print(wb_shapes)
    split_params_lst = []
    split_params = np.split(p, x_wb_size_new)
    for i in range(len(split_params) - 1):
        split_params_lst.append(np.reshape(split_params[i], wb_shapes[i]))
    ws_classify = split_params_lst[0:][::2]
    bs_classify = split_params_lst[1:][::2]

    hidden = kwargs1['hidden']
    y_hat = inputs
    for k_idx in range(len(hidden)):
        y_hat = sig_activation(np.dot(y_hat, ws_classify[k_idx]) + bs_classify[k_idx]) 
    y_hat = np.dot(y_hat, ws_classify[-1]) + bs_classify[-1]
    r = y - y_hat
    residue = np.square(r)
    loss = np.sum(np.sum(np.square(r), 1))
    return loss, r


def neural_net_loss_jcb(p_values, xinputs, y, kwargs1):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    hidden = kwargs1['hidden']
    sizes_wb, shapes_wb = kwargs1['wb_sizes'], kwargs1['wb_shapes']
    new_sizes_sum = []
    sum_size_psns = 0
    wb_sizes_array = np.asarray(sizes_wb)
    for new_idx in range(len(wb_sizes_array)):
        sum_size_psns += wb_sizes_array[new_idx]
        new_sizes_sum.append(sum_size_psns)
    print(new_sizes_sum)
    print(shapes_wb)
    # construct loss function
    splitted_list_params = []
    split_all_params = np.split(p_values, new_sizes_sum)
    # print(split_all_params)
    for ijex in range(len(split_all_params) - 1):
        splitted_list_params.append(np.reshape(split_all_params[ijex], shapes_wb[ijex]))
    ws_parameters = splitted_list_params[0:][::2]
    bs_parameters = splitted_list_params[1:][::2]

    y_hat = xinputs
    for k_idx in range(len(hidden)):
        y_hat = sig_activation(np.dot(y_hat, ws_parameters[k_idx]) + bs_parameters[k_idx])
    y_hat = np.dot(y_hat, ws_parameters[-1]) + bs_parameters[-1]
    r = y - y_hat
    y_loss_vec = np.sum(np.sum(np.square(r), 1))
    return y_loss_vec


def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sig_activation(np.dot(inputs, weights))


def func_loss_entropy(weights, inputs, targets):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))


