# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    :param input_filepath: input file containing data sets - ../raw
    :param output_filepath: destination file which is ../processed folder
    :return:
    """
    x_shape_0, input_features, output_labels = process_dataset_func(input_filepath)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


# Pre-process the data sets and send to the main function
def process_dataset_func(enter_filename):
    """
    The goal of this function is to import data, clean and pre-process
    :param enter_filename: file name containing data sets
    :return: training and testing sets.
    """
    df_dataset = pd.read_csv(enter_filename, sep=",", dtype={"diagnosis": "category"})
    # #######################################################################
    dummies = pd.get_dummies(df_dataset['diagnosis'], prefix='diagnosis', drop_first=False)
    dataxynew = pd.concat([df_dataset, dummies], axis=1)
    dataxynew1 = dataxynew.drop(['Unnamed: 32', 'id', 'diagnosis'], axis=1)
    output_labels = dataxynew1[['diagnosis_B', 'diagnosis_M']]
    input_features = dataxynew1.drop(['diagnosis_B', 'diagnosis_M'], axis=1)
    x_shape_0 = input_features.shape[0]
    return x_shape_0, input_features, output_labels


def wcds_preprocess(wsdata):
    data = pd.read_csv(wsdata, sep=',')
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
    y = data.diagnosis.values
    x_data = data.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).values
    xfeatures = data.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).values
    xfeats = pd.DataFrame(xfeatures)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(xfeats.corr(), fignum=f.number)
    plt.xticks(range(xfeats.shape[1]), xfeats.columns, fontsize=14, rotation=45)
    plt.yticks(range(xfeats.shape[1]), xfeats.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.show()
    onesx = np.ones((x_data.shape[0], 1))
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
    new_cat_features = y.reshape(-1, 1)
    ohe = OneHotEncoder(sparse=False)  # Easier to read
    ynew = ohe.fit_transform(new_cat_features)
    x = np.c_[x, onesx]
    x_train, x_test, y_train, y_test = train_test_split(x, ynew, test_size=0.20, random_state=42)
    # y_train = y_train[:, None]
    # y_test = y_test[:, None]
    return x_train, x_test, y_train, y_test


def func_solar_data(file_solar, dat_msg):
    mat_contents = sio.loadmat(file_solar, struct_as_record=False)
    oct_struct = mat_contents[dat_msg]
    if dat_msg == 'testData':
        valdata = oct_struct[0, 0].xyvalues
        x_normed = (valdata - valdata.min(0)) / valdata.ptp(0)
    elif dat_msg == 'solardatanorm':
        x_normed = oct_struct[0, 0].xyvalues
    else:
        valdata = oct_struct[0, 0].values
        x_normed = (valdata - valdata.min(0)) / valdata.ptp(0)
    input_data = x_normed[:, 0:-1]
    output_data = x_normed[:, -1]
    input_data_train, input_data_test, output_data_train, output_data_test = \
        train_test_split(input_data, output_data, test_size=0.30, shuffle=False)
    return input_data_train, input_data_test, output_data_train, output_data_test


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
