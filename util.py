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
        dataframe.append(df)
    data = pd.concat(dataframe, ignore_index=True).fillna(0)
    if(column_name):
        return data[column_name]
    else:
        return data
