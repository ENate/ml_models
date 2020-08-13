#!/usr/bin/env conda run -n my_env python
import sys
sys.path.append('../data')
from scipy import stats
import numpy as np
import pandas as pd
from make_dataset import make_data_set
# from ..data.make_dataset import make_data_set


def generate_feats_and_labels(features_without_activity, feature_with_activity):
    """
    A function to prepare input features for training. Function takes formatted version
    of the features set(without labels and a feature file containing labels. Uses num_time_steps and num_features
    to generate a sequence for input to the LSTM model for training
    Returns segments and encoded labels (from the activity column) which are prepared to be trained.
    :param features_without_activity: Formatted features containing the 'activity' column (labels)
    :param feature_with_activity: Formatted features without the 'activity' column (labels)
    :return: num_features, reshaped_segments and labels
    """
    num_time_steps = 64
    step = 6
    segments = []
    labels = []
    for i in range(0, len(features_without_activity) - num_time_steps, step):
        xlist = [features_without_activity[cols].values[i: i + num_time_steps] for cols in features_without_activity.columns]
        label = stats.mode(feature_with_activity['activity'][i: i + num_time_steps])[0][0]
        segments.append(xlist)
        labels.append(label)
    shape_of_segment = np.array(segments).shape
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    num_features = shape_of_segment[1]
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, num_time_steps, num_features)
    return num_features, reshaped_segments, labels


# call: supply the input raw file folder
if __name__ == '__main__':
    raw_data_file = '~/tasks/talpa-datascience-task/data/raw/data_case_study.csv'
    dat_with_activity, dat_without_activity, y_label_col = make_data_set(raw_data_file)
    feature_num, segment_inputs, encoded_labels = generate_feats_and_labels(dat_without_activity, dat_with_activity)