# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    :param input_filepath: Data source folder
    :param output_filepath: Data destination folder
    :return:
    """
    # return processed data and save in the output files
    in_data_y, y_output, in_data = make_data_set(input_filepath)
    in_data_y.to_csv(output_filepath)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    return in_data_y, y_output, in_data


def make_data_set(data_file_raw):
    """
    :param data_file_raw: input original data set for pre-processing
    :return: ds_without_activity: normalized training data set without output label.
             ds_with_activity: normalized training containing fewer columns without activity column
             output_label: activity of machine in different states
             Call function with the format: python make_dataset.py  /path/to/datasource/file /path/to/destination/file
    """
    df_dataset = pd.read_csv(data_file_raw, sep=',')
    # Drop both the 'Unnamed :0' and 'timestamp' columns from the original data set
    ds_with_activity = df_dataset[df_dataset.columns.difference(['timestamp', 'Unnamed: 0'])]
    # Drop both the 'Unnamed :0', 'timestamp' and 'activity' columns from the original data set
    ds_without_activity = df_dataset[df_dataset.columns.difference(['timestamp', 'Unnamed: 0', 'activity'])]
    # Assign the output label column containing the different machine states:
    output_label = df_dataset['activity']
    for col in ds_without_activity.columns:
        # feats_without_activity[col] = feats_without_activity[col].replace(-200, np.nan) # check nans
        # print(col, ':', feats_without_activity[col].isna().sum()/len(feats_without_activity))
        if ds_without_activity[col][:int(len(ds_without_activity) * 0.8)].isna().sum() \
                / int(len(ds_without_activity) * 0.8) > 0.5:  # at least 50% in train not nan
            ds_without_activity.drop(col, axis=1, inplace=True)
        else:  # fill nans
            ds_without_activity[col] = ds_without_activity[col].interpolate(method='linear', limit_direction='both')
        # Normalize data column-wise
    ds_without_activity /= ds_without_activity.max()
    # save normalized features in the interim folder
    ds_without_activity.to_csv('/home/nath/tasks/talpa-datascience-task/data/interim/'
                               'formatted_feats.csv', sep='\t', encoding='utf-8')
    # print(ds_without_activity.shape)
    # write data to a new .csv file in the ../processed folder
    return ds_with_activity, ds_without_activity, output_label


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # define the raw data folder name
    raw_data_file = 'data_case_study.csv'  # data_case_study.csv'
    interim_out_data_file = 'processed_data.csv'  # write to interim folder
    # x_train, y_train, x_train_no_y = make_data_set(raw_data_file)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
    # x_train, y_train, x_train_no_y = main(raw_data_file, interim_out_data_file)
    # print(x_train.shape)
