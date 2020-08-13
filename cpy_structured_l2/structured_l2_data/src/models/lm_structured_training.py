import os
import time
import pickle
import math
import numpy as np
import tensorflow as tf
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from .func_cond import func_compute_cond
from .classifpredAnalysis import predclassif, func_pred_new
from .py_func_loss import func_collect_allparams
from .train_model import TrainingModel


def new_structured_l2(ps, all_sizes, all_shapes):
    ws0 = tf.reshape(ps[0:all_sizes[0]], shape=all_shapes[0])
    bs0 = tf.reshape(ps[all_sizes[0]:], all_shapes[1])
    concat_wb = tf.concat([ws0, bs0], axis=0)
    p_square = tf.square(concat_wb)
    x_sum_cols = tf.reduce_sum(p_square, axis=1)
    x_sum_sqrt = tf.sqrt(x_sum_cols)
    x_sum_row = tf.reduce_sum(x_sum_sqrt, axis=0)
    return x_sum_row


def jacobian_mse(y, x_var, n_m):
    """compute the Jacbian matrix """
    loop_vars = [tf.constant(0, tf.int32), tf.TensorArray(tf.float64, size=n_m), ]
    _, jacobian = tf.while_loop(lambda i, _: i < n_m,
                                lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x_var)[0])), loop_vars)
    return jacobian.stack()


def model_l1_l2_func(nm_set_points, n_in, kwargs_vals, hyper_kwarg):
    hess_approx_flag = False
    neurons_cnt_x1, initializer = kwargs_vals['neurons_cnt'], kwargs_vals['initializer']
    all_sizes_vec, all_shapes_vec = kwargs_vals['sizes'], kwargs_vals['shapes']
    x_trained = kwargs_vals['xtr']
    y_trained = kwargs_vals['ytr']
    sess_values = kwargs_vals['sess']
    neurons_cnt = kwargs_vals['neurons_cnt']
    p = tf.Variable(initializer([neurons_cnt], dtype=tf.float64))
    # print(p)
    y_hat_model, y_hat_model_flat_x, y_labeled, x_in = TrainingModel().func_mse_l2(n_in, p, hyper_kwarg['mlp_hid_structure'], kwargs_vals)
    p_store = tf.Variable(tf.zeros([neurons_cnt_x1], dtype=tf.float64))
    save_params_p = tf.assign(p_store, p)
    restore_params_p = tf.assign(p, p_store)
    I_mat = tf.eye(neurons_cnt_x1, dtype=tf.float64)

    shaped_new = np.int(all_sizes_vec[0]) + np.int(all_sizes_vec[1])
    lasso_p = p[shaped_new:]
    l2_ps = p[0:shaped_new]
    l2_norm_val = new_structured_l2(l2_ps, all_sizes_vec, all_shapes_vec)
    # l2vals = func_structured_l2pen(p[0:shaped_new])
    all_reg_0 = tf.reduce_sum(tf.abs(lasso_p))
    r = y_labeled - y_hat_model  # regularization parameters lambda_param
    lambda_param = kwargs_vals['lambda_param']
    lambda_param2 = kwargs_vals['lambda_param2']
    # l2 structured norm loss function
    loss_val = tf.reduce_sum(tf.square(r)) + lambda_param * all_reg_0 + lambda_param2 * l2_norm_val
    mu = tf.placeholder(tf.float64, shape=[1])  # LM parameter
    # initialized store for all params, grad and hessian to be trained

    if hess_approx_flag:
        j1 = jacobian_mse(y_hat_model, p, nm_set_points)
        jt = tf.transpose(j1)
        jtj = tf.matmul(jt, j1)
        jtr = tf.matmul(jt, r)
        # compute gradient of l2 params
        l2_p_grads = tf.gradients(l2_norm_val, l2_ps)[0]
        hess_l2 = tf.hessians(l2_norm_val, l2_ps)[0]
        reshaped_gradl2 = jtr[0:shaped_new]
        l2_pen_hess = reshaped_gradl2 + lambda_param2 * tf.expand_dims(l2_p_grads, 1)
        # calculate gradient for lasso params group
        dxdt = tf.gradients(all_reg_0, lasso_p)[0]
        reshaped_gradl1 = jtr[shaped_new:]
        l1_pen_laapgrad = reshaped_gradl1 + lambda_param * tf.expand_dims(dxdt, 1)
        # creating block matrices for f(x, theta)
        reshaped_hessl2 = jtj[0:shaped_new, 0:shaped_new]
        # print(sess_values.run(tf.shape(l2_pen_hess)))
        # print(sess_values.run(tf.shape(l1_pen_laapgrad)))
        mat_hess_cols = jtj[0:shaped_new:, shaped_new:]
        mat_hess_rows = jtj[shaped_new:, :]
        # hess of in -> hid 1
        l2p_hess = reshaped_hessl2 + lambda_param2 * hess_l2
        # print(tf.shape(l2p_hess))
        print(tf.shape(l1_pen_laapgrad))
        jtr = tf.concat([l2_pen_hess, l1_pen_laapgrad], axis=0)
        l2p_concatcols = tf.concat([l2p_hess, mat_hess_cols], axis=1)
        jtj = tf.concat([l2p_concatcols, mat_hess_rows], axis=0)
    else:
        # remove it
        # stop_grads = tf.where(tf.math.equal(p, 0))
        jtj = tf.hessians(loss_val, p)[0]
        jtr = -tf.gradients(loss_val, p)[0]  # , stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(neurons_cnt_x1, 1))

    jtj_store = tf.Variable(tf.zeros((neurons_cnt_x1, neurons_cnt_x1), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((neurons_cnt_x1, 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, I_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except:
        c = tf.constant(0.1, dtype=tf.float64)
        input_mat += np.identity(input_mat.shape) * c
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    dx = tf.squeeze(dx)
    lm = hyper_kwarg['sgd'].apply_gradients([(-dx, p)])
    # p2 = p.assign(p + dx)
    sess_values = kwargs_vals['sess']
    feed_dict = {x_in: x_trained, y_labeled: y_trained}
    feed_dict[mu] = np.array([0.1], dtype=np.float64)
    i_cnt = 0
    step = 0
    mat_values = []
    sess_values.run(tf.global_variables_initializer())
    current_loss = sess_values.run(loss_val, feed_dict)

    while feed_dict[mu] > 1e-10 and step < 250:
        p0 = sess_values.run(p)
        p_0_indices = np.where(p == 0)
        p0[p_0_indices] = 0.0
        step += 1
        sess_values.run(save_params_p)
        if math.log(step, 2).is_integer():
            print('step', 'mu: ', 'current loss: ')
            print(step, feed_dict[mu][0], current_loss)
        success = False
        sess_values.run(jtj_store, feed_dict)
        sess_values.run(p_store)
        for _ in range(200):

            sess_values.run(save_jtj_jtr, feed_dict)
            sess_values.run(jtj_store, feed_dict)
            # p0 equals  session object with run of p2 and feed dict
            sess_values.run(lm, feed_dict)
            p0 = sess_values.run(p)
            # p0 = tf.where(p == 0, tf.zeros_like(p), p)
            values_vec = np.where(p0 == 0.0)
            p0[values_vec] = 0.0
            # print(sess.run(p0))
            new_loss = sess_values.run(loss_val, feed_dict)
            if new_loss < current_loss:
                # divide parameters to 2 groups: 1 for l1 and the other for structured l2
                shaped_new = np.int(all_sizes_vec[0]) + np.int(all_sizes_vec[1])
                lasso_p0 = p0[shaped_new:]
                in2_hidden_params = p0[0:shaped_new]
                mat_values.append(lasso_p0)
                # mat_values.append(p0)
                i_cnt += 1
                if len(mat_values) == 3:
                    sgn1 = mat_values[0] * mat_values[1]
                    sgn2 = mat_values[1] * mat_values[2]
                    # send the parameters to compute the values of structured penalty after
                    # checking if parameters are locally close to zero
                    px = mat_values[2]
                    osc_vec0 = np.where((sgn1 < 0) & (sgn2 < 0))
                    px[osc_vec0] = 0.0
                    # print(len(mat_values))
                    # join both sets of parameter lists here
                    # joined_params = np.concatenate(l2_params_set, new_p0)
                    # p_values_send = func_collect_all_params(px, all_sizes_vec, all_shapes_vec)
                    # p = my_full_matrix(nm, n, nn, sizes, shapes, p0)
                    px0 = tf.concat([in2_hidden_params, px], 0)

                    if lambda_param2 > 0.0 and np.mod(step, 10) == 0:
                        px0 = sess_values.run(px0)
                        new_all_params, ws_bs_in1_hid1 = func_compute_cond(px0, lambda_param2, kwargs_vals)
                        # cv = open("results_condvecs.csv", "w")
                        # write_cvs = csv.writer(cv, delimiter = '\t', lineterminator = '\n')
                        # write_cvs.writerow(cond_vec)
                    else:
                        new_all_params = np.array(sess_values.run(px0))
                    p_values_send = func_collect_allparams(new_all_params, all_sizes_vec, all_shapes_vec)
                    p.assign(p_values_send)
                    # p.assign(p_values_send)
                    i_cnt = 0
                    mat_values = []
                    # mat_values = [px]
                    # join both sets of params here
                    # joined_params = np.concatenate(l2_params, new_p0)
                else:
                    p.assign(p0)
                feed_dict[mu] /= 10
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu] *= 10
                p.assign(p0)
                # sess_values.run(save_params_p)
                sess_values.run(restore_params_p)
        if not success:
            print('Failed to improve')
            break

    p_new = sess_values.run(restore_params_p)
    abs_p = np.abs(p_new)
    idx_absp = np.where(abs_p < 0.1)
    p_new[idx_absp] = 0.0
    new_all_params, ws_bs_in1_hid1 = func_compute_cond(p_new, lambda_param2, kwargs_vals)
    p_new = func_collect_allparams(p_new, all_sizes_vec, all_shapes_vec)
    # p_new[osc_vec0]=0.0
    non_zero = np.count_nonzero(p_new)
    y_predict, x_inputs = func_pred_new(n_in, hyper_kwarg['mlp_hid_structure'], p_new, **kwargs_vals)
    inw_hid1 = tf.reshape(p_new[0:shaped_new],
                          shape=(all_shapes_vec[0][0] + all_shapes_vec[1][0], all_shapes_vec[0][1]))
    feed_dict2 = {x_inputs: x_trained}
    print('ENDED ON STEP: ', ' FINAL LOSS:')
    print(step, current_loss)
    # print(sess_values.run(inw_hid1))
    # print(sess_values.run(inw_hid1))
    y_model = sess_values.run(y_predict, feed_dict2)
    # cv.close()
    return restore_params_p, p_new, y_model, current_loss, non_zero


def model_main_func_l1(nm_set_points, n_in, nn_1, opt_obj, **kwargs_values):
    hess_approx = False
    neurons_cnt_x, sess = kwargs_values['neurons_cnt'], kwargs_values['sess']
    all_sizes, all_shapes = kwargs_values['sizes'], kwargs_values['shapes']
    y_hat, y_hat_flat_x, y, p, x_in = TrainingModel().func_mse_loss(n_in, nn_1)
    x_tr = kwargs_values['xtr']
    y_tr = kwargs_values['ytr']
    all_reg_0 = tf.reduce_sum(tf.abs(p))
    r = y - y_hat
    lambda1 = 0.001
    loss = tf.reduce_sum(tf.square(r)) + lambda1 * all_reg_0
    mu = tf.placeholder(tf.float64, shape=[1])
    p_store = tf.Variable(tf.zeros([neurons_cnt_x], dtype=tf.float64))
    save_params_p = tf.assign(p_store, p)
    restore_params_p = tf.assign(p, p_store)
    i_mat = tf.eye(neurons_cnt_x, dtype=tf.float64)
    I_mat_diag = tf.eye(neurons_cnt_x, dtype=tf.float64)

    if hess_approx:
        j1, dx_dt = jacobian_mse(y_hat, p, nm_set_points)
        jt = tf.transpose(j1)
        jtj = tf.matmul(jt, j1)
        jtr = 2 * (tf.matmul(jt, r)) + lambda1 * tf.reshape(dx_dt, shape=(neurons_cnt_x, 1))
    else:
        # remove it
        # stop_grads = tf.where(tf.math.equal(p, 0))
        jtj = tf.hessians(loss, p)[0]
        jtr = -tf.gradients(loss, p)[0]  # , stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(neurons_cnt_x, 1))

    jtj_store = tf.Variable(tf.zeros((neurons_cnt_x, neurons_cnt_x), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((neurons_cnt_x, 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, i_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except:
        c = tf.constant(0.1, dtype=tf.float64)
        input_mat += np.identity(input_mat.shape) * c
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)

    dx = tf.squeeze(dx)
    lm = opt_obj.apply_gradients([(-dx, p)])
    # p2 = p.assign(p + dx)
    sess_values = kwargs_values['sess']
    feed_dict = {x_in: x_tr, y: y_tr}
    feed_dict[mu] = np.array([0.1], dtype=np.float64)
    i_cnt = 0
    step = 0
    mat_values = []
    sess_values.run(tf.global_variables_initializer())
    current_loss = sess_values.run(loss, feed_dict)
    zero0 = tf.constant(0., dtype=tf.float64)
    while feed_dict[mu] > 1e-10 and step < 300:
        p0 = sess_values.run(p)
        p_0_indices = np.where(p == 0)
        p0[p_0_indices] = 0.0
        step += 1
        sess.run(save_params_p)
        if math.log(step, 2).is_integer():
            print('step', 'mu: ', 'current loss: ')
            print(step, feed_dict[mu][0], current_loss)
        success = False
        sess_values.run(jtj_store, feed_dict)
        sess_values.run(p_store)
        for _ in range(400):

            sess_values.run(save_jtj_jtr, feed_dict)
            sess_values.run(jtj_store, feed_dict)
            # p0 equals  session object with run of p2 and feed dict
            sess_values.run(lm, feed_dict)
            p0 = sess_values.run(p)
            # p0 = tf.where(p == 0, tf.zeros_like(p), p)
            values_vec = np.where(p0 == 0.0)
            p0[values_vec] = 0.0
            # print(sess.run(p0))
            new_loss = sess_values.run(loss, feed_dict)
            if new_loss < current_loss:
                mat_values.append(p0)
                i_cnt += 1
                if len(mat_values) == 3:
                    sgn1 = mat_values[0] * mat_values[1]
                    sgn2 = mat_values[1] * mat_values[2]
                    # checking if parameters are locally close to zero
                    px = mat_values[2]
                    p_0_indices = np.where((sgn1 < 0) & (sgn2 < 0))
                    print(len(mat_values))
                    # print(len(mat_values))
                    print(p_0_indices)
                    # print(p_0_indices)
                    i_cnt = 0
                    mat_values = []
                    # mat_values = [px]
                else:
                    p.assign(p0)
                feed_dict[mu] /= 10
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu] *= 10
                p.assign(p0)
                # sess_values.run(save_params_p)
                sess_values.run(restore_params_p)
        if not success:
            print('Failed to improve')
            break
    p_new = sess_values.run(restore_params_p)
    abs_p = np.abs(p_new)
    ixd_p = np.where(abs_p < 1e-02)
    p_new[ixd_p] = 0.0
    p20 = p_new
    # p20[p_0_indices] = 0.0
    p20 = func_collect_allparams(p20, all_sizes, all_shapes)
    ymodel, x_inputs = func_pred_new(n_in, nn_1, p20, **kwargs_values)
    feed_dict_vals = {x_inputs: x_tr}
    print('ENDED ON STEP: ', ' FINAL LOSS:')
    print(step, current_loss)
    print('Parameters: ')
    print(sess_values.run(restore_params_p))
    print('Parameters: ')
    print(p_new)
    print('p2:')
    y_model = sess_values.run(ymodel, feed_dict_vals)
    print(p20)
    return restore_params_p, p20, y_model


def jacobian_classif(y, x, m):
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float64, size=m),
    ]
    _, jac_classif = tf.while_loop(lambda i, _: i < m, lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x)[0])),
                                   loop_vars)
    print(jac_classif.stack())
    return jac_classif.stack()


# saves checkpoint and outputs current step/loss/mu to files
def log(step, loss, params, out_file, d_kwargs, mu=None):
    global log_prev_time, log_first_time

    now = time.time()
    if log_prev_time and now - log_prev_time < d_kwargs['LOG_INTERVAL_IN_SEC']:
        return
    if not log_prev_time:
        log_prev_time, log_first_time = now, now
    secs_from_start = int(now - log_first_time) + d_kwargs['time_delta']
    step += d_kwargs['step_delta']
    message = step + secs_from_start + loss
    message += mu if mu else ''
    print(message)
    with open(out_file, 'a') as file:
        file.write('message \n')
    pickle.dump((step, secs_from_start, params), open(out_file + '.ckpt', "wb"))
    log_prev_time = now


def func_classifier_l2l1(sec_dict, n_points, choose_flag, d_kwargs):
    hess_approx_flag = False
    neurons_cnt, initializer = sec_dict['num_neurons'], sec_dict['initializer']
    activation = d_kwargs.activation
    params0 = tf.Variable(initializer([neurons_cnt], dtype=tf.float64))
    loss, x, y, y_hat_model, struct_l2pen = TrainingModel().func_cross_entropy_loss(sec_dict, params0, choose_flag)
    feed_dict = {x: sec_dict['xtr'], y: sec_dict['ytr']}
    # feed_dict2 = {x: sec_kwarg['xtest'], y: sec_kwarg['ytest']}

    # regularization parameters
    lambda_param = 0.00005
    lambda_param2 = 0.00006
    # l2 structured norm loss function

    mu = tf.placeholder(tf.float64, shape=[1])
    # initialized store for all params, grad and hessian to be trained # LM parameter
    p_store = tf.Variable(tf.zeros([sec_dict['num_neurons']], dtype=tf.float64))
    save_params_p = tf.assign(p_store, params0)
    restore_params_p = tf.assign(params0, p_store)
    i_mat = tf.eye(sec_dict['num_neurons'], dtype=tf.float64)

    shaped_new = np.int(sec_dict['wb_sizes'][0]) + np.int(sec_dict['wb_sizes'][1])
    lasso_p = params0[shaped_new:]
    print(lasso_p)
    all_reg_0 = tf.reduce_sum(tf.abs(lasso_p))
    loss_val = loss + lambda_param * all_reg_0 + lambda_param2 * struct_l2pen

    if hess_approx_flag:
        j1 = jacobian_classif(y_hat_model, p_store, n_points)
        jt = tf.transpose(j1)
        jtj = tf.matmul(jt, j1)
        # check the correct value of r
        jtr = tf.matmul(jt, loss)
    else:
        # remove it
        jtj = tf.hessians(loss_val, params0)[0]
        jtr = -tf.gradients(loss_val, params0)[0]  # , stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(sec_dict['num_neurons'], 1))

    jtj_store = tf.Variable(tf.zeros((sec_dict['num_neurons'], sec_dict['num_neurons']), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((sec_dict['num_neurons'], 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, i_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            c = tf.constant(0.1, dtype=tf.float64)
            input_mat += np.identity(input_mat.shape) * c
            dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
        else:
            raise
    dx = tf.squeeze(dx)
    lm = sec_dict['opt_obj'].apply_gradients([(-dx, params0)])
    # p2 = p.assign(p + dx)
    # sess_values = d_kwargs['sess']
    # print(sess_values.run(lasso_p)) 
    feed_dict[mu] = np.array([10], dtype=np.float64)
    i_cnt = 0
    step = 0
    mat_values = []
    sec_dict['sess'].run(tf.global_variables_initializer())
    current_loss = sec_dict['sess'].run(loss_val, feed_dict)

    while feed_dict[mu] > 1e-10 and step < 200:
        p0 = sec_dict['sess'].run(params0)
        values_vec = np.where(params0 == 0)
        p0[values_vec] = 0.0
        step += 1
        sec_dict['sess'].run(save_params_p)
        # sess.run(restore_params_p)
        if math.log(step, 2).is_integer():
            print('step', 'mu: ', 'current loss: ')
            print(step, feed_dict[mu][0], current_loss)
        success = False
        sec_dict['sess'].run(jtj_store, feed_dict)
        sec_dict['sess'].run(p_store)
        for _ in range(300):

            sec_dict['sess'].run(save_jtj_jtr, feed_dict)
            sec_dict['sess'].run(jtj_store, feed_dict)
            # p0 equals  session object with run of p2 and feed dict
            sec_dict['sess'].run(lm, feed_dict)
            p0 = sec_dict['sess'].run(params0)
            # p0 = tf.where(p == 0, tf.zeros_like(p), p)
            values_vec = np.where(p0 == 0.0)
            p0[values_vec] = 0.0
            new_loss = sec_dict['sess'].run(loss_val, feed_dict)
            if new_loss < current_loss:
                # divide parameters to 2 groups: 1 for l1 and the other for structured l2
                shaped_new = np.int(sec_dict['wb_sizes'][0]) + np.int(sec_dict['wb_sizes'][1])
                lasso_p0 = p0[shaped_new:]
                in2_hidden_params = p0[0:shaped_new]
                mat_values.append(lasso_p0)
                i_cnt += 1
                if len(mat_values) == 3:
                    sgn1 = mat_values[0] * mat_values[1]
                    sgn2 = mat_values[1] * mat_values[2]  # store parameters
                    # checking if parameters are locally close to zero
                    px = mat_values[2]
                    values_vec = np.where((sgn1 < 0) & (sgn2 < 0))
                    px[values_vec] = 0.0
                    print(len(mat_values))
                    # join both sets of parameter lists here joined_params = np.concatenate(l2_params_set, new_p0)
                    px0 = tf.concat([in2_hidden_params, px], 0)
                    if lambda_param2 > 0.0 and np.mod(step, 5):
                        px0 = sec_dict['sess'].run(px0)
                        new_all_params, ws_bs_in1_hid1 = func_compute_cond(px0, lambda_param2, sec_dict)
                    else:
                        new_all_params = np.array(sec_dict['sess'].run(px0))
                    # sess_px0 = np.array(sess_values.run(px0))
                    p_values_send = func_collect_allparams(new_all_params, sec_dict['wb_sizes'], sec_dict['wb_shapes'])
                    print(p_values_send.shape)
                    params0.assign(p_values_send)
                    i_cnt = 0
                    mat_values = []
                    # mat_values = [px]
                else:
                    params0.assign(p0)
                feed_dict[mu] /= 10
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu] *= 10
                params0.assign(p0)
                # sess.run(save_params_p)
                sec_dict['sess'].run(restore_params_p)
        if not success:
            print('Failed to improve')
            break

    p_new = sec_dict['sess'].run(restore_params_p)
    abs_p = np.abs(p_new)
    idx_absp = np.where(abs_p < 0.1)
    p_new[idx_absp] = 0.0
    # p_new[values_vec] = 0.0
    correct_prediction, feed_dict2, y_hat_classif_logits = predclassif(sec_dict['wb_sizes'], sec_dict['xydata'],
                                                                       d_kwargs.mlp_hid_structure, p_new,
                                                                       activation, sec_dict['wb_shapes'])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print('ENDED ON STEP: ')
    print(step)
    print(' FINAL LOSS:')
    print(current_loss)
    print('Parameters: ')
    print(sec_dict['sess'].run(restore_params_p))
    print('Parameters: ')
    print(p_new)
    print("Accuracy:", sec_dict['sess'].run(accuracy, feed_dict2))
    correct_predictions = sec_dict['sess'].run(y_hat_classif_logits, feed_dict2)
    return p_new, correct_predictions


def func_classifier_l1(nm_set_points, choose_flag_0, kwargspred, **kwargs):
    hess_approx_flag = False
    initializer = kwargspred['initializer']
    mu1, _, mu_dec, max_inc = kwargs['mu'], kwargs['mu_inc'], kwargs['mu_dec'], kwargs['mu_inc']
    wb_shapes, wb_sizes_classif, hidden = kwargspred['wb_shapes'], kwargspred['wb_sizes'], kwargspred['hidden']
    activation, xydat = kwargspred['activation'], kwargspred['xydat']
    x_in = kwargspred['xtr']
    y_labeled = kwargspred['ytr']
    sess, neurons_cnt_x1 = kwargspred['sess'], kwargspred['neurons_cnt']
    opt_obj = kwargspred['opt_obj']
    params0 = tf.Variable(initializer([neurons_cnt_x1], dtype=tf.float64))
    loss, x, y, y_hat_model = TrainingModel().func_cross_entropy_loss(wb_sizes_classif,
                                                                      params0, choose_flag_0)
    feed_dict = {x: x_in, y: y_labeled}
    # feed_dict2 = {x: kwargspred['ytest1'], y: kwargspred['ytest']}
    # regularization parameters
    lambda_param = 0.005
    # l2 structured norm loss function
    mu = tf.placeholder(tf.float64, shape=[1])
    # initialized store for all params, grad and hessian to be trained # LM parameter
    p_store = tf.Variable(tf.zeros([neurons_cnt_x1], dtype=tf.float64))
    save_params_p = tf.assign(p_store, params0)
    restore_params_p = tf.assign(params0, p_store)
    ix_mat = tf.eye(neurons_cnt_x1, dtype=tf.float64)
    all_reg_0 = tf.reduce_sum(tf.abs(params0))
    loss_val = loss + lambda_param * all_reg_0

    if hess_approx_flag:
        j1 = jacobian_classif(y_hat_model, p_store, nm_set_points)
        jt = tf.transpose(j1)
        jtj = tf.matmul(jt, j1)
        # check the correct value of r
        jtr = tf.matmul(jt, loss)
    else:
        # remove it
        # stop_grads = tf.where(tf.math.equal(p, 0))
        jtj = tf.hessians(loss_val, params0)[0]
        jtr = -tf.gradients(loss_val, params0)[0]  # , stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(neurons_cnt_x1, 1))

    jtj_store = tf.Variable(tf.zeros((neurons_cnt_x1, neurons_cnt_x1), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((neurons_cnt_x1, 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, ix_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except:
        c = tf.constant(0.1, dtype=tf.float64)
        input_mat += np.identity(input_mat.shape) * c
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    dx = tf.squeeze(dx)
    lm = opt_obj.apply_gradients([(-dx, params0)])
    # p2 = p.assign(p + dx)
    sess_values = kwargspred['sess']
    # print(sess_values.run(lasso_p)) 
    feed_dict[mu] = np.array([0.1], dtype=np.float64)
    i_cnt = 0
    step = 0
    mat_values = []
    sess_values.run(tf.global_variables_initializer())
    current_loss = sess_values.run(loss_val, feed_dict)

    while feed_dict[mu] > 1e-10 and step < 200:
        p0 = sess_values.run(params0)
        values_vec = np.where(params0 == 0)
        p0[values_vec] = 0.0
        step += 1
        sess.run(save_params_p)
        # sess.run(restore_params_p)
        if math.log(step, 2).is_integer():
            print('step', 'mu: ', 'current loss: ')
            print(step, feed_dict[mu][0], current_loss)
        success = False
        sess_values.run(jtj_store, feed_dict)
        sess_values.run(p_store)
        for _ in range(300):

            sess_values.run(save_jtj_jtr, feed_dict)
            sess_values.run(jtj_store, feed_dict)
            # p0 equals  session object with run of p2 and feed dict
            sess_values.run(lm, feed_dict)
            p0 = sess_values.run(params0)
            # p0 = tf.where(p == 0, tf.zeros_like(p), p)
            values_vec = np.where(p0 == 0.0)
            p0[values_vec] = 0.0
            new_loss = sess_values.run(loss_val, feed_dict)
            if new_loss < current_loss:
                # divide parameters to 2 groups: 1 for l1 and the other for structured l2
                mat_values.append(p0)
                i_cnt += 1
                if len(mat_values) == 3:
                    sgn1 = mat_values[0] * mat_values[1]
                    sgn2 = mat_values[1] * mat_values[2]  # store parameters
                    # checking if parameters are locally close to zero
                    px = mat_values[2]
                    values_vec = np.where((sgn1 < 0) & (sgn2 < 0))
                    px[values_vec] = 0.0
                    print(len(mat_values))
                    # join both sets of parameter lists here joined_params = np.concatenate(l2_params_set, new_p0)
                    p_values_send = func_collect_allparams(px, wb_sizes_classif, wb_shapes)
                    print(p_values_send.shape)
                    params0.assign(p_values_send)
                    i_cnt = 0
                    mat_values = []
                    # mat_values = [px]
                else:
                    params0.assign(p0)
                feed_dict[mu] /= 10
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu] *= 10
                params0.assign(p0)
                # sess.run(save_params_p)
                sess_values.run(restore_params_p)
        if not success:
            print('Failed to improve')
            break

    p_new = sess_values.run(restore_params_p)
    abs_p = np.abs(p_new)
    idx_absp = np.where(abs_p < 0.1)
    p_new[idx_absp] = 0.0
    # p_new[values_vec] = 0.0
    correct_prediction, feed_dict2, y_hat_classif_logits = predclassif(wb_sizes_classif, xydat, hidden, p_new,
                                                                       activation, wb_shapes)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print('ENDED ON STEP: ')
    print(step)
    print(' FINAL LOSS:')
    print(current_loss)
    print('Parameters: ')
    print(sess_values.run(restore_params_p))
    print('Parameters: ')
    print(p_new)
    print("Accuracy:", sess.run(accuracy, feed_dict2))
    correct_predictions = sess.run(y_hat_classif_logits, feed_dict2)
    return p_new, correct_predictions


def train_classifier_sgd(tf_optimizers, skt_dict, kwargs3):
    """
    Classifier to test dropout and compare with our method
    :param skt_dict: dictionary of user define training parameters
    :param tf_optimizers: step to print results and update check points
    :param kwargs3: command line user-defined parameters
    :return: current_loss values, prediction and optimal weights after training
    """
    step, batch_size = 0, 5
    optimizer = kwargs3.optimizer
    params0 = tf.Variable(skt_dict['initializer']([skt_dict['num_neurons']], dtype=tf.float64))
    loss, x, y,_, _, _ = TrainingModel().func_mse_loss(skt_dict, params0)
    train_step = tf_optimizers[optimizer](0.1).minimize(loss)
    feed_dict = {x: skt_dict['xtr'], y: skt_dict['ytr']}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # calc initial loss
    current_loss = session.run(loss, feed_dict)
    while current_loss > 1e-10 and step < 400:
        step += 1
        # log(step, current_loss, session.run(params))
        # session.run(train_step, feed_dict)
        current_loss = session.run(loss, feed_dict)
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (skt_dict['ytr'].shape[0] - batch_size)
        # Generate a mini-batch.
        batch_data = skt_dict['xtr'][offset:(offset + batch_size), :]
        batch_labels = skt_dict['ytr'][offset:(offset + batch_size), :]
        # Prepare a dictionary telling the
        # session where to feed the mini-batch.
        # The key of the dictionary is the
        # placeholder node of the graph to be fed
        # and the value is the numpy array to feed to it.
        feed_dict = {x: batch_data, y: batch_labels}
        _, l, predictions = session.run([optimizer, loss], feed_dict=feed_dict)
    return current_loss


def train_tf_classifier(tf_optimizers, skt_dict, kwargs3):
    step = 0
    optimizer = kwargs3.optimizer
    params0 = tf.Variable(skt_dict['initializer']([skt_dict['num_neurons']], dtype=tf.float64))
    loss, x, y, _, _ = TrainingModel().func_cross_entropy_loss(skt_dict, params0, 1)  # func_mse_loss(skt_dict, params0)
    train_step = tf_optimizers[optimizer](0.1).minimize(loss)
    feed_dict1 = {x: skt_dict['xtr'], y: skt_dict['ytr']}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    current_loss = session.run(loss, feed_dict1)
    while current_loss > 1e-10 and step < 400:
        step += 1
        log(step, current_loss, session.run(params0))
        session.run(train_step, feed_dict1)
        current_loss = session.run(loss, feed_dict1)


