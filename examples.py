import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from util import *
import fnmatch
import numpy as np
import statistics
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Specify path to dataset
path = './csse_covid_19_daily_reports_us/'

file_count = len(fnmatch.filter(os.listdir(path), '*.*'))
# print('File Count:', file_count)

# Get MEAN
data_March9_2023 = generateDataframe(path=path, year="03-09-2023", column_name="Confirmed")
# print(data_March9_2023)
tf_March9_2023_confirmed = tf.convert_to_tensor(data_March9_2023, dtype="float")
sum_March9_2023_confirmed = tf.math.reduce_sum(tf_March9_2023_confirmed)
# print(sum_March9_2023_confirmed)

# print(f'TOTAL CONFIRMED CASES: {sum_March9_2023_confirmed}')
# print(f'Average Confirmed Cases: {sum_March9_2023_confirmed / file_count}')



def sd(self,data):
    print("standard deviation")
    dataset = np.array(data)
    std_dev = np.std(dataset)
   



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
sumList2020 = []
sumList2021 = []
sumList2022 = []
sumList2023 = []
prev_sum = 0
prev_conf = []

#processing data to turn comeulative data into iterative data 

while i < len(data_confirmed_2020) - 58:

    confirmedSum = sum(data_confirmed_2020[i:j])
    sumList2020.append(confirmedSum - prev_sum)
    # prev_conf.append((confirmedSum, prev_sum))
    prev_sum = confirmedSum
    i += 58
    j += 58   

prev_sum = 0
i = 0
j = 58

while i < len(data_confirmed_all) - 58:

    confirmedSum = sum(data_confirmed_all[i:j])
    sumList.append(confirmedSum - prev_sum)
    # prev_conf.append((confirmedSum, prev_sum))
    prev_sum = confirmedSum
    i += 58
    j += 58  

prev_sum = 0
i = 0
j = 58

#processing data to turn comeulative data into iterative data 


while i < len(data_confirmed_2021) - 58:

    confirmedSum = sum(data_confirmed_2021[i:j])
    sumList2021.append(confirmedSum - prev_sum)
    # prev_conf.append((confirmedSum, prev_sum))
    prev_sum = confirmedSum
    i += 58
    j += 58  

prev_sum = 0
i = 0
j = 58

#processing data to turn comeulative data into iterative data 


while i < len(data_confirmed_2022) - 58:

    confirmedSum = sum(data_confirmed_2022[i:j])
    sumList2022.append(confirmedSum - prev_sum)
    # prev_conf.append((confirmedSum, prev_sum))
    prev_sum = confirmedSum
    i += 58
    j += 58  

prev_sum = 0
i = 0
j = 58


#processing data to turn comeulative data into iterative data 


while i < len(data_confirmed_2023) - 58:

    confirmedSum = sum(data_confirmed_2023[i:j])
    sumList2023.append(confirmedSum - prev_sum)
    # prev_conf.append((confirmedSum, prev_sum))
    prev_sum = confirmedSum
    i += 58
    j += 58 





overallStats = {}
overallStats2020 = {}
overallStats2021 = {}
overallStats2022 = {}
overallStats2023 = {}

sumTensor = tf.convert_to_tensor(sumList, dtype="float")
sumTensor = tf.convert_to_tensor(sumList, dtype="float")
mean_sumTensor = tf.math.reduce_mean(sumTensor)

print("Stats for overall across 3 years")
overallStats["Mean"] = statistics.mean(sumList) 
overallStats["Variance"] = statistics.variance(sumList) 
overallStats["Standard Deviation"] = statistics.stdev(sumList)

print("Stats for 2020")
overallStats2020["Mean"] = statistics.mean(sumList2020) 
overallStats2020["Variance"] = statistics.variance(sumList2020) 
overallStats2020["Standard Deviation"] = statistics.stdev(sumList2020)


print("Stats for 2021")
overallStats2021["Mean"] = statistics.mean(sumList2021) 
overallStats2021["Variance"] = statistics.variance(sumList2021) 
overallStats2021["Standard Deviation"] = statistics.stdev(sumList2021)

print("Stats for 2022")
overallStats2022["Mean"] = statistics.mean(sumList2022) 
overallStats2022["Variance"] = statistics.variance(sumList2022) 
overallStats2022["Standard Deviation"] = statistics.stdev(sumList2022)

print("Stats for 2023")
overallStats2023["Mean"] = statistics.mean(sumList2023) 
overallStats2023["Variance"] = statistics.variance(sumList2023) 
overallStats2023["Standard Deviation"] = statistics.stdev(sumList2023)



sum_sumTensor = tf.math.reduce_sum(sumTensor)
std_sumTensor = tf.math.reduce_std(sumTensor)

variance_sumTensor = tf.math.reduce_variance(sumTensor) 

# print(f'Average Confirmed Cases per day: {mean_sumTensor}')
# print(f'Variance of Confirmed Cases: {variance_sumTensor}')
# print(f'Standard Deviation of Confirmed Cases: {std_sumTensor}')
# print('Variance of confirm', statistics.variance(sumList))
# print(f'Total Number of Confirmed Cases: {sum_sumTensor}')

# print(overallStats)
# print(overallStats2020)
# print(overallStats2021)
# print(overallStats2022)
# print(overallStats2023)


df = generateDataframe(path=path)

# Filter out provinces
provinceList = ['American Samoa', 'Diamond Princess', 'Grand Princess', 'Guam', 'Northern Mariana Islands', 'Puerto Rico', 'Recovered', 'Virgin Islands']
df = df[~df.Province_State.isin(provinceList)]

#processing to get monthly avg data 

df['_date'] = pd.to_datetime(df['_date'], format='%m-%d-%Y')

testFormat = "2020-4-30"
df_apr_2020 = df[df['_date'].dt.strftime('%Y-%m-%d') == '2020-04-30']
monthlyAvg2020 = {}

yearFormat = "2020-0"
monthformat = 4
dateFormat = "-30"

df_may_2020 = df[df['_date'].dt.strftime('%Y-%m-%d') == '2020-05-30']
print("may avg", sum(df_apr_2020["Confirmed"])/30)



for i in range(3,12):
    if i == 13:
        break
    if monthformat == 10:
        yearFormat = "2020-"

    monthStringFormat = str(monthformat)
    finalFormat = yearFormat + monthStringFormat + dateFormat
  
    monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == finalFormat]
    if monthformat == 4:
        avg = sum(monthlyDF["Confirmed"])/18
        monthlyAvg2020[finalFormat] = avg
  
    avg = sum(monthlyDF["Confirmed"])/30
    monthlyAvg2020[finalFormat] = avg
    monthformat += 1

monthlyAvg2021 = {}

yearFormat = "2021-0"
monthformat = 1
dateFormat = "-30"

for i in range(12):
    if i == 13:
        break
    if monthformat == 10:
        yearFormat = "2021-"

    monthStringFormat = str(monthformat)
    finalFormat = yearFormat + monthStringFormat + dateFormat
    if monthformat == 2:
         monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == "2021-04-28"]
    else:
         monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == finalFormat]
    if monthformat == 2:
        avg = sum(monthlyDF["Confirmed"])/28
        monthlyAvg2021[finalFormat] = avg
  
    avg = sum(monthlyDF["Confirmed"])/30
    monthlyAvg2021[finalFormat] = avg
    monthformat += 1


monthlyAvg2022 = {}

yearFormat = "2022-0"
monthformat = 1
dateFormat = "-30"

for i in range(12):
    if i == 13:
        break
    if monthformat == 10:
        yearFormat = "2022-"

    monthStringFormat = str(monthformat)
    finalFormat = yearFormat + monthStringFormat + dateFormat
    print(finalFormat)
    if monthformat == 2:
         monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == "2022-04-28"]
    else:
         monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == finalFormat]

    print(monthlyDF)
    if monthformat == 2:
        avg = sum(monthlyDF["Confirmed"])/28
        monthlyAvg2022[finalFormat] = avg
  
    avg = sum(monthlyDF["Confirmed"])/30
    monthlyAvg2022[finalFormat] = avg
    monthformat += 1

monthlyAvg2023 = {}

yearFormat = "2023-0"
monthformat = 1
dateFormat = "-30"

for i in range(3):

    monthStringFormat = str(monthformat)
    finalFormat = yearFormat + monthStringFormat + dateFormat

    if monthformat == 3:
         monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == "2023-03-09"]
    else:
         monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == finalFormat]
    if monthformat == 2:
         monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == "2023-02-28"]

    print(monthlyDF)
    if monthformat == 2:
        avg = sum(monthlyDF["Confirmed"])/28
        monthlyAvg2023[finalFormat] = avg

    if monthformat == 3:
        avg = sum(monthlyDF["Confirmed"])/9
        monthlyAvg2023[finalFormat] = avg
  
    avg = sum(monthlyDF["Confirmed"])/30
    monthlyAvg2023[finalFormat] = avg
    monthformat += 1

print("Mobthly Avgs for 2020")
print(monthlyAvg2020)
print("Mobthly Avgs for 2022")
print(monthlyAvg2021)
print("Mobthly Avgs for 2022")
print(monthlyAvg2022)
print("Mobthly Avgs for 2023")
print(monthlyAvg2023)



stateAvg2020 = {}

yearFormat = "2020-0"
monthformat = 4
dateFormat = "-30"

for i in range(3,12):
    if monthformat == 13:
        break
    if monthformat == 10:
        yearFormat = "2020-"

    monthStringFormat = str(monthformat)
    finalFormat = yearFormat + monthStringFormat + dateFormat

    monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == finalFormat]

    stateAvg2020[finalFormat] = []

    if monthformat == 4:
        for j in monthlyDF["Confirmed"]:
            avg = j/18
            stateAvg2020[finalFormat].append(avg)

    else:
        for j in monthlyDF["Confirmed"]:
            avg = j/30
            stateAvg2020[finalFormat].append(avg)

  
    # avg = sum(monthlyDF["Confirmed"])/30
    # monthlyAvg2020[finalFormat] = avg
    monthformat += 1

    
    

stateAvg2021 = {}

yearFormat = "2021-0"
monthformat = 1
dateFormat = "-30"

for i in range(12):
    if monthformat == 13:
        break
    if monthformat == 10:
        yearFormat = "2021-"

    monthStringFormat = str(monthformat)
    finalFormat = yearFormat + monthStringFormat + dateFormat

    monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == finalFormat]

    stateAvg2021[finalFormat] = []

    if monthformat == 2:
        for j in monthlyDF["Confirmed"]:
            avg = j/28
            stateAvg2021[finalFormat].append(avg)

    else:
        for j in monthlyDF["Confirmed"]:
            avg = j/30
            stateAvg2021[finalFormat].append(avg)

  
    # avg = sum(monthlyDF["Confirmed"])/30
    # monthlyAvg2020[finalFormat] = avg
    monthformat += 1
    

stateAvg2022 = {}

yearFormat = "2022-0"
monthformat = 1
dateFormat = "-30"

for i in range(12):
    if monthformat == 13:
        break
    if monthformat == 10:
        yearFormat = "2022-"

    monthStringFormat = str(monthformat)
    finalFormat = yearFormat + monthStringFormat + dateFormat
    monthlyDF =  df[df['_date'].dt.strftime('%Y-%m-%d') == finalFormat]

    stateAvg2022[finalFormat] = []

    if monthformat == 2:
        for j in monthlyDF["Confirmed"]:
            avg = j/28
            stateAvg2022[finalFormat].append(avg)

    else:
        for j in monthlyDF["Confirmed"]:
            avg = j/30
            stateAvg2022[finalFormat].append(avg)

  
    # avg = sum(monthlyDF["Confirmed"])/30
    # monthlyAvg2020[finalFormat] = avg
    monthformat += 1
    


