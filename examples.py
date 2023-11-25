import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from util import *
import fnmatch
import numpy as np
import statistics

# Specify path to dataset
path = './csse_covid_19_daily_reports_us/'

file_count = len(fnmatch.filter(os.listdir(path), '*.*'))
# print('File Count:', file_count)

# # Extract dataframes from dataset
# data_2021_confirmed = generateDataframe(path=path, year="2021", column_name="Confirmed")

# # Convert dataframe to tensor
# tf_2021_confirmed = tf.convert_to_tensor(data_2021_confirmed, dtype="float")

# # Find mean of data
# mean_2021_confirmed = tf.math.reduce_mean(tf_2021_confirmed)
# print(f'2021 Confirmed Mean: {mean_2021_confirmed}')

# # Find variance of data
# var_2021_confirmed = tf.math.reduce_variance(tf_2021_confirmed)
# print(f'2021 Confirmed Variance: {var_2021_confirmed}')

# # Find standard deviation of data
# sd_2021_confirmed = tf.math.reduce_std(tf_2021_confirmed)
# print(f'2021 Confirmed Standard Deviation: {sd_2021_confirmed}')


# Get MEAN
data_March9_2023 = generateDataframe(path=path, year="03-09-2023", column_name="Confirmed")
# print(data_March9_2023)
tf_March9_2023_confirmed = tf.convert_to_tensor(data_March9_2023, dtype="float")
sum_March9_2023_confirmed = tf.math.reduce_sum(tf_March9_2023_confirmed)
# print(sum_March9_2023_confirmed)

print(f'TOTAL CONFIRMED CASES: {sum_March9_2023_confirmed}')
print(f'Average Confirmed Cases: {sum_March9_2023_confirmed / file_count}')



def sd(self,data):
    print("standard deviation")
    dataset = np.array(data)
    std_dev = np.std(dataset)
    print(f"the Standard deviation for confirmed cases: {std_dev}")  



# data_confirmed_all = (generateDataframe(path=path, column_name="Confirmed")).sort_values(ascending=True)
# print(f'Total Days in data_confirmed_all: {len(data_confirmed_all) / 58}')

# print(data_confirmed_all[:58])


data_confirmed_2020 = generateDataframe(path=path, year="2020", column_name="Confirmed")
data_confirmed_2021 = generateDataframe(path=path, year="2021", column_name="Confirmed")
data_confirmed_2022 = generateDataframe(path=path, year="2022", column_name="Confirmed")
data_confirmed_2023 = generateDataframe(path=path, year="2023", column_name="Confirmed")

data_confirmed_all = pd.concat([data_confirmed_2020, data_confirmed_2021, data_confirmed_2022, data_confirmed_2023])

i = 0
j = 58
sumList = []
prev_sum = 0
prev_conf = []
while i < len(data_confirmed_all) - 58:

    confirmedSum = sum(data_confirmed_all[i:j])
    sumList.append(confirmedSum - prev_sum)
    # prev_conf.append((confirmedSum, prev_sum))
    prev_sum = confirmedSum
    i += 58
    j += 58    



sumTensor = tf.convert_to_tensor(sumList, dtype="float")
sumTensor = tf.convert_to_tensor(sumList, dtype="float")
mean_sumTensor = tf.math.reduce_mean(sumTensor)
sum_sumTensor = tf.math.reduce_sum(sumTensor)
std_sumTensor = tf.math.reduce_std(sumTensor)
variance_sumTensor = tf.math.reduce_variance(sumTensor) 
print(f'Average Confirmed Cases per day: {mean_sumTensor}')
print(f'Variance of Confirmed Cases: {variance_sumTensor}')
print(f'Standard Deviation of Confirmed Cases: {std_sumTensor}')
print('Variance of confirm', statistics.variance(sumList))
print(f'Total Number of Confirmed Cases: {sum_sumTensor}')
