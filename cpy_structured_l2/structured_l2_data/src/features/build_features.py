from sklearn.model_selection import train_test_split


class BuildFeatures(object):

    def __init__(self):
        self.in_values, self.x_train = None, None

    def build_features_function(self, in_values):
        self.in_values = in_values
        return self.in_values * 2

    def func_cancer_data_features(self, input_feature, label_output):
        self.x_train, x_test, y_train, y_test = train_test_split(input_feature, label_output,
                                                                 test_size=0.20, shuffle=False)
        return self.x_train, x_test, y_train, y_test


if __name__ == '__main__':
    BuildFeatures().build_features_function(4)