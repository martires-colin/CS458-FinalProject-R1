import os
import pandas as pd

def generateDataframe(path, year=None, column_name=None):
    """
    Extract a dataframe from the collection of CSV files

    :param path: path to the dataset
    :param year: target year, leave blank to retrieve all years
    :param column_name: target column, leave blank to retrieve all columns
    :returns: a dataframe
    """
    suffix = ".csv"
    dataframe = []
    if(year):
        suffix = year + ".csv"
    csv_files = [f for f in os.listdir(path) if f.endswith(suffix)]
    for file in csv_files:
        df = pd.read_csv(path + file)
        df["_date"] = file[:-4]
        dataframe.append(df)
    data = pd.concat(dataframe, ignore_index=True).fillna(0)
    if(column_name):
        return data[column_name]
    else:
        return data

path = './csse_covid_19_daily_reports_us/'
df = generateDataframe(path=path)

df['_date'] = pd.to_datetime(df['_date'], format='%m-%d-%Y')

df_2020 = df[df['_date'].dt.strftime('%Y') == '2020']
df_2021 = df[df['_date'].dt.strftime('%Y') == '2021']
df_2022 = df[df['_date'].dt.strftime('%Y') == '2022']
df_2023 = df[df['_date'].dt.strftime('%Y') == '2023']

# To get specific month
# just add on to the strftime filter
# Ex: get Jan 2021
df_jan_2021 = df[df['_date'].dt.strftime('%Y-%m') == '2021-01']
print(df_jan_2021)

# Same for Specific Day
# Ex: get Jan 31, 2021
df_jan31_2021 = df[df['_date'].dt.strftime('%Y-%m-%d') == '2021-01-31']
print(df_jan31_2021)
