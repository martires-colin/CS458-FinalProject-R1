import pandas
import os
import re
# Import the classes created earlier
from Display import MonthDisplay, YearDisplay

class CSVDataProcessor:
    def __init__(self, directory):
        
        self.directory = './csse_covid_19_daily_reports_us/'

    def filter_csv_files(self):
        all_files = os.listdir(self.directory)
        valid_files = [f for f in all_files if re.match(
            r'\d{2}-\d{2}-2022\.csv', f)]
        return valid_files

    def process_files(self):
        year_data = []
        for file in self.filter_csv_files():
            month_data = self.process_single_file(file)
            # Assuming you want the sum of the month data for the year
            year_data.append(sum(month_data))
            month_plot = MonthDisplay(month_data)
            month_plot.show_plot()  # Display each month

        year_plot = YearDisplay(year_data)
        year_plot.show_plot()  # Display aggregated year data

    def process_single_file(self, file_name):
        file_path = os.path.join(self.directory, file_name)
        df = pandas.read_csv(file_path)
        # Assuming 'Confirmed' is the column name
        
        return df['Confirmed'].tolist()


# Example usage
processor = CSVDataProcessor('/path/to/csv/directory')
processor.process_files()
