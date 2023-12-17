import pandas as pd
import os
from collections import defaultdict
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class YearlyCovidSummary:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(int))

    def add_file(self, file_path):
        date_str = os.path.basename(file_path).split('.')[0]
        date_obj = datetime.strptime(date_str, "%m-%d-%Y")

        df = pd.read_csv(file_path)

        month = date_obj.month
        year = date_obj.year
        self.data[year][month] += df['Confirmed'].sum()

    def get_yearly_summary(self, year):
        return self.data[year]

    def process_directory(self, directory_path):
        for file_path in glob.glob(os.path.join(directory_path, '*.csv')):
            self.add_file(file_path)
    def plot_multiple_years_summary(self, years):
        num_years = len(years)
        plt.figure(figsize=(12, 6 * num_years))

        for i, year in enumerate(years, start=1):
            if year not in self.data:
                print(f"No data available for the year {year}")
                continue

            months = range(1, 13)  # Months from January to December
            cases = [self.data[year][month] for month in months]

            plt.subplot(num_years, 1, i)
            plt.bar(months, cases, color='blue')
            plt.ylabel(f'Confirmed Cases in {year}')
            plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        plt.tight_layout()
        plt.show()

class YearlyFileOrganizer:
    def __init__(self):
        self.files_by_year = defaultdict(list)

    def add_file(self, file_path):
        year = datetime.strptime(os.path.basename(
            file_path).split('.')[0], "%m-%d-%Y").year
        self.files_by_year[year].append(file_path)

    def get_files_for_year(self, year):
        return self.files_by_year[year]

    def process_directory(self, directory_path):
        for file_path in glob.glob(os.path.join(directory_path, '*.csv')):
            self.add_file(file_path)


class RegionalCovidTrends:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def add_file(self, file_path):
        date_str = os.path.basename(file_path).split('.')[0]
        date_obj = datetime.strptime(date_str, "%m-%d-%Y")

        df = pd.read_csv(file_path)

        month = date_obj.month
        year = date_obj.year
        for _, row in df.iterrows():
            region = row['Province_State']
            self.data[region][year][month] += row['Confirmed']

    def process_directory(self, directory_path):
        for file_path in glob.glob(os.path.join(directory_path, '*.csv')):
            self.add_file(file_path)

    def plot_region_daily(self, region, year, month):
        if region not in self.daily_data:
            print(f"No data available for {region}")
            return

        dates = [date for date in self.daily_data[region].keys(
        ) if date.year == year and date.month == month]
        dates.sort()
        cases = [self.daily_data[region][date] for date in dates]

        plt.figure(figsize=(15, 6))
        plt.plot(dates, cases, marker='o')
        plt.xlabel('Date')
        plt.ylabel('Confirmed Cases')
        plt.title(
            f'Daily COVID-19 Confirmed Cases in {region} ({month}/{year})')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_all_regions(self, year, regions_per_figure=10):
        regions = list(self.data.keys())
        num_regions = len(regions)
        total_figures = (num_regions + regions_per_figure - 1) // regions_per_figure

        for fig_num in range(total_figures):
            start_index = fig_num * regions_per_figure
            end_index = min(start_index + regions_per_figure, num_regions)
            current_regions = regions[start_index:end_index]

            rows = cols = int(regions_per_figure ** 0.5)
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            axes = axes.flatten()

            for i, region in enumerate(current_regions):
                months = range(1, 13)
                cases = [self.data[region][year][month] for month in months]

                axes[i].plot(months, cases, marker='o')
                axes[i].set_title(region)
                axes[i].set_xticks(months)
                axes[i].set_xticklabels(['Jn', 'F', 'Mc', 'Ap', 'My', 'Jn', 'Jl', 'Ag', 'Sp', 'Oc', 'Nv', 'Dc'])
                axes[i].grid(True)

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout(pad=3.0)
            plt.show()


# Example usage


# Example usage
summary = YearlyCovidSummary()
organizer = YearlyFileOrganizer()

# Directory containing CSV files
directory_path = "C:/Users/davis/OneDrive/Desktop/CS WORK/cs458/CS458-FinalProject-R1/csse_covid_19_daily_reports_us"

summary.process_directory(directory_path)

organizer.process_directory(directory_path)
summary.plot_multiple_years_summary([2020])
summary.plot_multiple_years_summary([2021])
summary.plot_multiple_years_summary([2022])
summary.plot_multiple_years_summary([2023])

# # Get summary for a specific year
# print(summary.get_yearly_summary(2020))

# # Get files for a specific year
# print(organizer.get_files_for_year(2021))
trends = RegionalCovidTrends()
trends.process_directory(directory_path)

# Plot all regions for a specific year
#trends.plot_all_regions(2020, regions_per_figure=16)
#trends.plot_all_regions(2021, regions_per_figure=16)
trends.plot_all_regions(2022, regions_per_figure=16)
#trends.plot_all_regions(2023, regions_per_figure=16)

class MonthlyRegoinalTrends:
    def __init__(self):
        # Dictionary to store monthly and daily data
        self.monthly_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.daily_data = defaultdict(lambda: defaultdict(int))

    def add_file(self, file_path):
        date_str = os.path.basename(file_path).split('.')[0]
        date_obj = datetime.strptime(date_str, "%m-%d-%Y")

        df = pd.read_csv(file_path)

        month = date_obj.month
        year = date_obj.year
        for _, row in df.iterrows():
            region = row['Province_State']
            self.monthly_data[region][year][month] += row['Confirmed']
            self.daily_data[region][date_obj] += row['Confirmed']

    def process_directory(self, directory_path):
        for file_path in glob.glob(os.path.join(directory_path, '*.csv')):
            self.add_file(file_path)

    def plot_region_daily(self, region, year, month):
        if region not in self.daily_data:
            print(f"No data available for {region}")
            return

        dates = [date for date in self.daily_data[region].keys() if date.year == year and date.month == month]
        dates.sort()
        cases = [self.daily_data[region][date] for date in dates]

        plt.figure(figsize=(15, 6))
        plt.plot(dates, cases, marker='o')
        plt.xlabel('Date')
        plt.ylabel('Confirmed Cases')
        plt.title(f'Daily COVID-19 Confirmed Cases in {region} ({month}/{year})')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

 

# Example usage
trends = MonthlyRegoinalTrends()
trends.process_directory(directory_path)

# Plot daily trends for a specific region, year, and month
# Example: nevada in April 2020
#for x in range(1, 13):
#    trends.plot_region_daily('Nevada', 2022, x)

