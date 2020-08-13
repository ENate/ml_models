import argparse


def fun_dict_values(kwargs):
    a_values = kwargs['a'] * 2
    b_values = kwargs['b'] * 4
    return a_values, b_values


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp_hid_structure', help='Number of steps', default=[5, 3], type=object)
    parser.add_argument('--n_classes', help='Number of steps', default=3, type=int)
    parser.add_argument('--num_features', help='Number of steps', default=4, type=int)
    args = parser.parse_args()
    mlp_hid_structure = args.mlp_hid_structure
    n_classes = args.n_classes
    num_features = args.num_features
    h_params0 = {'mlp_hid_structure': mlp_hid_structure, 'n_classes': n_classes, 'num_features': num_features}  #
    # h_params0 = hparam.HParams(args.__dict__)
    return args


if __name__ == '__main__':
    kwargs_b = {}
    kwargs_b['a'] = 2
    kwargs_b['b'] = 4
    print(kwargs_b)
    print(fun_dict_values(kwargs_b))
    print('From the kwargs')
    hparams0 = create_parser()
    print(hparams0.n_classes)