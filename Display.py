#display class 
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