import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from util import *

# Specify path to dataset
path = './csse_covid_19_daily_reports_us/'

# Extract dataframes from dataset
data_2021_confirmed = generateDataframe(path=path, year="2021", column_name="Confirmed")

# Convert dataframe to tensor
tf_2021_confirmed = tf.convert_to_tensor(data_2021_confirmed, dtype="float")

# Find mean of data
mean_2021_confirmed = tf.math.reduce_mean(tf_2021_confirmed)
print(f'2021 Confirmed Mean: {mean_2021_confirmed}')

# Find variance of data
var_2021_confirmed = tf.math.reduce_variance(tf_2021_confirmed)
print(f'2021 Confirmed Variance: {var_2021_confirmed}')

# Find standard deviation of data
sd_2021_confirmed = tf.math.reduce_std(tf_2021_confirmed)
print(f'2021 Confirmed Standard Deviation: {sd_2021_confirmed}')