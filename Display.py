import matplotlib.pyplot as plt
import numpy as np

class MonthDisplay:
    def __init__(self, month_data):
        self.month_data = month_data
        print(self.month_data)
    def show_plot(self):
        days = np.arange(1, len(self.month_data) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(days, self.month_data, marker='o')
        plt.title("Data for the Month")
        plt.xlabel("Day of the Month")
        plt.ylabel("Data Value")
        plt.grid(True)
        plt.show()

class YearDisplay:
    def __init__(self, year_data):
        self.year_data = year_data

    def show_plot(self):
        months = np.arange(1, len(self.year_data) + 1)
        plt.figure(figsize=(10, 6))
        plt.bar(months, self.year_data)
        plt.title("Data for the Year")
        plt.xlabel("Month")
        plt.ylabel("Data Value")
        plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True)
        plt.show()

# Example usage
month_data = []  # replace with actual daily data for a month
year_data = []  # replace with actual monthly data for a year

month_plot = MonthDisplay(month_data)
month_plot.show_plot()

year_plot = YearDisplay(year_data)
year_plot.show_plot()#display class 
import matplotlib.pyplot as plt 
import pandas as pd

class Display:
    def __init__(self, data):
        self.data = data
    
    def plot_trend(self, data_column, metric):
        self.data[data_column] = pd.to_datetime(self.data[data_column])

        trend = self.data.groupby(data_column).sum()[metric]

        plt.figure(figsize= (12, 6))
        plt.plot(trend, label= metric)
        plt.legend()
        plt.grid(True)
        plt.show()

#to use this 
# Covid_display = Display(fulldata) 
# Covis_display.plot_trend('last_update', "confirmed")