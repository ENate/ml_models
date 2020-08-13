from scipy import stats
import numpy as np
import pandas as pd


def generate_feats_and_labels(features_files, feature_file_with_activity):
    """ A function to make features for training. Takes formatted version
    of the features (without labels and a feature file containing labels. Uses num_time_steps and num_features
    to generate a sequence for input to the LSTM model for training
    Returns segments and encoded labels (from the activity column)
    """
    num_time_steps = 100
    step = 20
    segments = []
    labels = []
    for i in range(0, len(features_files) - num_time_steps, step):
        xlist = [features_files[cols].values[i: i + num_time_steps] for cols in features_files.columns]
        label = stats.mode(feature_file_with_activity['activity'][i: i + num_time_steps])[0][0]
        segments.append(xlist)
        labels.append(label)
    shape_of_segment = np.array(segments).shape
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    num_features = shape_of_segment[1]
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, num_time_steps, num_features)
    return num_features, reshaped_segments, labels
