import pandas as pd
from util import generateDataframe

# Read COVID dataset
path = './csse_covid_19_daily_reports_us/'
df = generateDataframe(path=path)

# Filter out provinces
provinceList = ['American Samoa', 'Diamond Princess', 'Grand Princess', 'Guam', 'Northern Mariana Islands', 'Puerto Rico', 'Recovered', 'Virgin Islands']
df = df[~df.Province_State.isin(provinceList)]

df = df.filter(items=['Province_State', 'Confirmed', '_date', 'confirmed_per_capita'])
df['_date'] = pd.to_datetime(df['_date'])
df = df.sort_values(by=['_date'])
print(df)

df = df[df._date == pd.to_datetime('2023-03-09')]
total_cases = df['Confirmed'].sum()
print(total_cases)

# Get total cases per day (loop)