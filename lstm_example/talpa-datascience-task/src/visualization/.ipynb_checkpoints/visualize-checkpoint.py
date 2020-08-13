# ## PLOTTING UTILITY FUNCTIONS ###
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## SPLIT TRAIN TEST ###
os.chdir('/home/nath/talpa_cookiecutter/talpa-datascience-task/')
ds_case_file = 'data/raw/data_case_study.csv'
ds_case_data = pd.read_csv(ds_case_file, sep=',')
df = ds_case_data.assign(missing= np.nan)


train_date = df[:int(len(df)*0.8)]
train = df[:int(len(df)*0.8)].copy()

test_date = df[int(len(df)*0.8):]
test = df[int(len(df)*0.8):].copy()

print(train.shape, test.shape)


def plot_sensor(name):
    plt.figure(figsize=(16, 4))
    plt.plot(train_date, train[name], label='train')
    plt.plot(test_date, test[name], label='test')
    plt.ylabel(name)
    plt.legend()
    plt.show()


def plot_autocor(name, df):
    plt.figure(figsize=(16, 4))
    # pd.plotting.autocorrelation_plot(df[name])
    # plt.title(name)
    # plt.show()
    timeLags = np.arange(1, 100 * 24)
    plt.plot([df[name].autocorr(dt) for dt in timeLags])
    plt.title(name)
    plt.ylabel('autocorr')
    plt.xlabel('time lags')
    plt.show()
