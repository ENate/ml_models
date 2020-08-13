import tensorflow as tf


class TrainingModel(object):

    def __init__(self):
        self.opt_loss_mse, self.gen_sum_parameters, self.opt_loss = None, None, None
        self.b_values, self.shapes_classify, self.param_matrix = None, None, None
        self.h_params, self.l2_model_flat, self.y_hat_flat_values = None, None, None

    def build_mlp_structure(self, num_features, hparams, n_classes):
        """
        Takes in an hparams of dictionary values
        :param num_features number of input features in input data set
        :param n_classes: number of output classes determined from the given output data
        :param hparams: dict of number of features, network layers and classes
        :return: number of neurons, shapes of weights (biases and network) and sizes of each group of weights
        """
        # mlp_structure = [int(hparams.num_features)] + hparams.mlp_hid_structure + [int(hparams.n_classes)]
        mlp_structure = [int(num_features)] + hparams.mlp_hid_structure + [int(n_classes)]
        self.shapes_classify = []
        for idx in range(len(hparams.mlp_hid_structure) + 1):
            self.shapes_classify.append((mlp_structure[idx], mlp_structure[idx + 1]))
            self.shapes_classify.append((1, mlp_structure[idx + 1]))
        wb_sizes_classif = [hclassif * wclassif for hclassif, wclassif in self.shapes_classify]
        neurons_cnt_classif = sum(wb_sizes_classif)
        print('Total number of trainable parameters is', neurons_cnt_classif)
        return neurons_cnt_classif, self.shapes_classify, wb_sizes_classif, hparams.mlp_hid_structure

    def func_structured_l2pen_classifier(self, ws_matrix, bs_matrix):
        """
        Computes the structured l2 norm of each group pf weights emanating from each input neuron
        :param bs_matrix: parameters connecting input edges to the first hidden layer
        :param ws_matrix: bias parameters of the input layer
        :return: structured penalty
        """
        combine_wb_matrices = tf.concat([ws_matrix, bs_matrix], axis=0)
        sq_parameters = tf.square(combine_wb_matrices)
        column_wise_addition = tf.reduce_sum(sq_parameters, axis=1)
        sqrt_row_wise_params = tf.sqrt(column_wise_addition)
        self.gen_sum_parameters = tf.reduce_sum(sqrt_row_wise_params)
        return self.gen_sum_parameters

    def func_mse_loss(self, param_kwargs, p_initial):
        """
        Determine the loss function
        :param p_initial: initial parameters for building network
        :param param_kwargs: dictionary kwargs for all parameters needed in function
        :return:logits, place_holder inputs and outputs
        """
        x_place_hold = tf.placeholder(tf.float64, shape=[None, param_kwargs['n']])
        y = tf.placeholder(tf.float64, shape=[None, param_kwargs['y_n']])
        # p = tf.Variable(initializer([neurons_cnt_new], dtype=tf.float64))
        x_params = tf.split(p_initial, param_kwargs['wb_sizes'])
        for i_index in range(len(x_params)):
            x_params[i_index] = tf.reshape(x_params[i_index], param_kwargs['wb_shapes'][i_index])
        weights = x_params[0:][::2]
        biases = x_params[1:][::2]
        y_hat = x_place_hold
        print(param_kwargs['n'])
        for i in range(len(param_kwargs['nhidden'])):
            y_hat = tf.nn.relu(tf.matmul(y_hat, weights[i]) + biases[i])
            y_hat = tf.nn.dropout(y_hat, rate=0.2)  # , tf.compat.v1.placeholder("float"))
        y_hat = tf.matmul(y_hat, weights[-1]) + biases[-1]
        self.y_hat_flat_values = tf.squeeze(y_hat)
        if param_kwargs['choose_flag'] == 1:
            self.opt_loss_mse = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_hat, labels=y))
        else:
            self.opt_loss_mse = self.y_hat_flat_values
        return self.opt_loss_mse, y_hat, self.y_hat_flat_values, y, p_initial, x_place_hold

    def func_mse_l2(self, n_input_pts, p_init, ll_hid, kwargs_d):
        m_shape, m_sizes, m_neurons_cnt = kwargs_d['shapes'], kwargs_d['sizes'], kwargs_d['neurons_cnt']
        initializer, activation = kwargs_d['initializer'], kwargs_d['activation']
        x_holder = tf.placeholder(tf.float64, shape=[None, n_input_pts])
        y_holder = tf.placeholder(tf.float64, shape=[None, 1])
        x_init_parameters = tf.split(p_init, m_sizes)
        for k_idx in range(len(x_init_parameters)):
            x_init_parameters[k_idx] = tf.reshape(x_init_parameters[k_idx], m_shape[k_idx])
        l2_weights = x_init_parameters[0:][::2]
        l2_biases = x_init_parameters[1:][::2]
        l2_model = x_holder
        for l_idx in range(len(ll_hid)):
            l2_model = activation(tf.matmul(l2_model, l2_weights[l_idx]) + l2_biases[l_idx])
        l2_model = tf.matmul(l2_model, l2_weights[-1]) + l2_biases[-1]
        self.l2_model_flat = tf.squeeze(l2_model)
        # l2_norm_val = func_structured_l2pen(l2_weights[0], l2_biases[0])
        return l2_model, self.l2_model_flat, y_holder, x_holder  # , l2_norm_val

    def func_cross_entropy_loss(self, sec_kwarg, parameter_values, choose_flag_value):
        """
        This function computes the entropy loss for the MLP model to be trained. It uses the TensorFlow logits function
        :param wb_sizes_classify: input sizes of each group parameter in each layer including biases
        :param parameter_values: a tensorFlow vector of parameters for entire MLP
        :param choose_flag_value: set which of regression or classification to run
        :param sec_kwarg: dictionary of parameters 2
        :param kwargs_params: a dictionary of parameters required in constructing the training loss
        :return: loss: loss function, placeholders and value of structured_l2 penalty
        """
        x_classify = tf.placeholder(tf.float64, shape=[None, sec_kwarg['n']])
        y_classify = tf.placeholder(tf.float64, shape=[None, sec_kwarg['nclasses']])
        classify_tensors = tf.split(parameter_values, sec_kwarg['wb_sizes'], 0)
        for i in range(len(classify_tensors)):
            classify_tensors[i] = tf.reshape(classify_tensors[i], sec_kwarg['wb_shapes'][i])
        ws_classif = classify_tensors[0:][::2]
        bs_classif = classify_tensors[1:][::2]
        y_hat_classify = x_classify
        for i in range(len(sec_kwarg['nhidden'])):
            y_hat_classify = tf.nn.sigmoid(tf.matmul(y_hat_classify, ws_classif[i]) + bs_classif[i])
        y_hat_classify = tf.matmul(y_hat_classify, ws_classif[-1]) + bs_classif[-1]
        #################################################################################
        structured_l2_penalty = TrainingModel().func_structured_l2pen_classifier(ws_classif[0], bs_classif[0])
        ####################################################################################
        if choose_flag_value == 1:
            if sec_kwarg['reg_param'] == 1:
                self.opt_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_classify,
                                                                                      labels=y_classify))
            else:
                self.opt_loss = y_hat_classify
        else:
            self.opt_loss = y_hat_classify

        # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat_classify, labels=y_classify))
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat_classify,
        # labels=tf.cast(y_classify, tf.float64)))
        return self.opt_loss, x_classify, y_classify, y_hat_classify, structured_l2_penalty

    def train_tf_classifier(self, param_matrix):
        """
        A function to train a model using dropout method. Goal is to determine how using the dropout method
        differs from the continuous feature selection implemented using the structured l2 norm penalty.
        :param param_matrix: Total parameters to train MLP
        :return: parameter optimized matrix
        """
        self.param_matrix = param_matrix + 4
        return self.param_matrix

    def training_model_function(self, x_num):
        self.num = x_num + self.num
        return self.num

    def batch_sizes(self, b_values):
        self.b_values = b_values
        return self.b_values


if __name__ == '__main__':
    # call function and pass HPARAMS
    print("The code is running... ")
