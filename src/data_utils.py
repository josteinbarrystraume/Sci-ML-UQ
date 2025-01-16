import os
import glob
import pandas as pd
import torch as torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def get_filepaths_in_dir(dir_path):
    """
    Returns a list of full file paths (including directory) in the specified directory.
    Skips directories or non-regular files.
    """
    filepaths = []

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_dir():  # or entry.is_file() for files
                filepaths.append([entry.path])

    return filepaths


def make_dataframe(directory_paths, segments, maneuver):
    df_list = []

    for dir_path in directory_paths:
        for segment in segments:
            if maneuver == 'takeoff':
                regex_var = '*_takeoff_*'
            elif maneuver == 'climb':
                regex_var = '*_climb_*'
            elif maneuver == 'cruise':
                regex_var = '*_cruise_*'

            path = os.path.join(dir_path[0],
                                regex_var +
                                '*_segment_' +
                                str(segment) + '.csv')

            for file in glob.glob(path):
                #print(file)
                if file:
                    #print(file)
                    df = pd.read_csv(file)
                    df_list.append(df)

    df_concat = pd.concat(df_list)

    return df_concat


def scale_data(x, y, x_scaler, y_scaler, is_fit=True):
    if is_fit:
        x_scaled = pd.DataFrame(data=x_scaler.fit_transform(x), columns=x.columns)
        y_scaled = pd.Series(data=y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))[:, -1],
                             name=y.name)

    else:
        x_scaled = pd.DataFrame(data=x_scaler.transform(x), columns=x.columns)
        y_scaled = pd.Series(data=y_scaler.transform(y.to_numpy().reshape(-1, 1))[:, -1],
                             name=y.name)

    return x_scaled, y_scaled


def make_tensor(df, is_target=False):
    if is_target:
        # y variable, target variable
        tensor = torch.tensor(df.to_numpy(),
                              requires_grad=True,
                              dtype=torch.float32).reshape(df.shape[0], 1, 1)

    else:
        # x variables, predictor variables
        tensor = torch.tensor(df.to_numpy(),
                              requires_grad=True,
                              dtype=torch.float32).reshape(df.shape[0], 1, df.shape[1])

    return tensor
