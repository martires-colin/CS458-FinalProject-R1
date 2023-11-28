import pandas as pd
from util import generateDataframe

# Read US populations dataset
filename = 'us-state-populations.csv'
population_data = pd.read_csv(filename)

# Read COVID dataset
path = './csse_covid_19_daily_reports_us/'
df = generateDataframe(path=path)

# Filter out provinces
provinceList = ['American Samoa', 'Diamond Princess', 'Grand Princess', 'Guam', 'Northern Mariana Islands', 'Puerto Rico', 'Recovered', 'Virgin Islands']
df = df[~df.Province_State.isin(provinceList)]

# Filter df to only contain total confirmed cases (end of month)
monthEndDates = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30', '07-31', '08-31', '09-30', '10-31', '11-30', '12-31']
df = df[df._date.str.contains('|'.join(monthEndDates))]

def calculateConfirmedPerCapita(row):
    targetState = row["Province_State"]
    targetStatePopulation = population_data.loc[population_data['state'] == targetState, 'pop_2014'].item()
    numConfirmed = row['Confirmed']
    return numConfirmed / targetStatePopulation

df['confirmed_per_capita'] = df.apply(calculateConfirmedPerCapita, axis=1)
print(df)
