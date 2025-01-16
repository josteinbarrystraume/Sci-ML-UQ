"""
Copyright (c) 2025 Jostein Barry-Straume

All rights reserved.
"""

import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVR

import data_utils


# Set default datatype
torch.set_default_dtype(torch.float32)

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Get file paths for a given engine
directory = '../data/trent_800/'
dir_paths = data_utils.get_filepaths_in_dir(directory)

#print(dir_paths)

# Create train and test dataframes
train_segments = [1]
test_segments = [2, 3, 4, 5]

train_df = data_utils.make_dataframe(dir_paths, train_segments, maneuver='takeoff')
test_df = data_utils.make_dataframe(dir_paths, test_segments, maneuver='takeoff')

train_y = train_df['TGTU']
test_y = test_df['TGTU']

predictor_variables = ['N1', 'N2', 'N3', 'P25', 'P3', 'T3']
train_x = train_df[predictor_variables]
test_x = test_df[predictor_variables]

#print(train_x.head())
#print(test_x.head())
#print(train_y.head())
#print(test_y.head())

# Create train and test tensors
tensor_train_x = data_utils.make_tensor(train_x, is_target=False)
tensor_test_x = data_utils.make_tensor(test_x, is_target=False)
tensor_train_y = data_utils.make_tensor(train_y, is_target=True)
tensor_test_y = data_utils.make_tensor(test_y, is_target=True)

print('x_train: shape:{} | type:{} | dtype:{}'.format(tensor_train_x.shape, type(tensor_train_x), tensor_train_x.dtype))
print('x_test: shape:{} | type:{} | dtype:{}'.format(tensor_test_x.shape, type(tensor_test_x), tensor_test_x.dtype))
print('y_train: shape:{} | type:{} | dtype:{}'.format(tensor_train_y.shape, type(tensor_train_y), tensor_train_y.dtype))
print('y_test: shape:{} | type:{} | dtype:{}'.format(tensor_test_y.shape, type(tensor_test_y), tensor_test_y.dtype))
