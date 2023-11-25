import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from util import *

# Assuming you have cloned the repository and the path is correct
directory = './csse_covid_19_daily_reports_us/'

# Handle missing values
full_data_2021 = full_data_2021.fillna(0)  # Or another appropriate value
full_data_2022 = full_data_2022.fillna(0)  # Or another appropriate value
full_data_2023 = full_data_2023.fillna(0)  # Or another appropriate value
full_data = full_data.fillna(0)

confirmed_data_2021 = full_data_2021["Confirmed"]
confirmed_data_2022 = full_data_2022["Confirmed"]
confirmed_data_2023 = full_data_2023["Confirmed"]
confirmed_data = full_data_2023["Confirmed"]

# print(sum(confirmed_data), "sum confirmed")
# print((confirmed_data))

print(len(full_data))
print(len(full_data_2021))
print(len(full_data_2022))
print(len(full_data_2023))

# Convert to TensorFlow Dataset
# train_data_2021, eval_data_2021 = train_test_split(confirmed_data_2021, test_size=0.2)
# train_data_2022, eval_data_2022 = train_test_split(confirmed_data_2022, test_size=0.2)
# train_data_2023, eval_data_2023 = train_test_split(confirmed_data_2023, test_size=0.2)
# train_data, eval_data = train_test_split(confirmed_data, test_size=0.2)

train_data_2021 = confirmed_data_2021
train_data_2022 = confirmed_data_2022
train_data_2023 = confirmed_data_2023
train_data = confirmed_data

# print(len(confirmed_data_2021))
# print(len(train_data_2021))
# print(len(eval_data_2021))

# train_dataset = tf.ragged.constant(train_data, dtype="float")
# eval_dataset = tf.ragged.constant(eval_data, dtype="float")

# train_dataset = tf.convert_to_tensor((train_data.drop('Confirmed', axis=1), train_data['Confirmed']), dtype="float")
train_dataset_2021 = tf.convert_to_tensor(train_data_2021, dtype="float")
train_dataset_2022 = tf.convert_to_tensor(train_data_2022, dtype="float")
train_dataset_2023 = tf.convert_to_tensor(train_data_2023, dtype="float")
train_dataset = tf.convert_to_tensor(train_data, dtype="float")
# eval_dataset = tf.convert_to_tensor((eval_data.drop('Confirmed', axis=1), eval_data['Confirmed']), dtype="float")

# eval_dataset_2021 = tf.convert_to_tensor(eval_data_2021, dtype="float")
# eval_dataset_2022 = tf.convert_to_tensor(eval_data_2022, dtype="float")
# eval_dataset_2023 = tf.convert_to_tensor(eval_data_2023, dtype="float")
# eval_dataset = tf.convert_to_tensor(eval_data, dtype="float")

# print("TRAIN")
# print(train_dataset_2021)
# print(train_dataset_2022)
# print(train_dataset_2023)
# print(train_dataset)

# print("\nEVAL")
# print(eval_dataset_2021)
# print(eval_dataset_2022)
# print(eval_dataset_2023)
# print(eval_dataset)

totalMean = tf.reduce_mean(train_dataset)
mean2021 = tf.reduce_mean(train_dataset_2021)
mean2022 = tf.reduce_mean(train_dataset_2022)
mean2023 = tf.reduce_mean(train_dataset_2023)
print("Mean (Total): ", totalMean)
print("Mean (2021): ", mean2021)
print("Mean (2022): ", mean2022)
print("Mean (2023): ", mean2023)

totalVariance = tf.math.reduce_variance(train_dataset)
var2021 = tf.reduce_mean(train_dataset_2021)
var2022 = tf.reduce_mean(train_dataset_2022)
var2023 = tf.reduce_mean(train_dataset_2023)
print("Variance: ", totalVariance)
print("Variance: ", var2021)
print("Variance: ", var2022)
print("Variance: ", var2023)

totalSD = tf.math.reduce_std(train_dataset)
sd2021 = tf.reduce_mean(train_dataset_2021)
sd2022 = tf.reduce_mean(train_dataset_2022)
sd2023 = tf.reduce_mean(train_dataset_2023)
print("SD: ", totalSD)
print("SD: ", sd2021)
print("SD: ", sd2022)
print("SD: ", sd2023)

# totalMeanAD = tf.math.reduce_mean_absolute_deviation(train_dataset)
# print("MAD: ", totalMeanAD)



# Batch and shuffle the data for training
#train_dataset = train_dataset.shuffle(len(train_data)).batch(32)
#eval_dataset = eval_dataset.batch(32)


# Further processing and model training
# ...
