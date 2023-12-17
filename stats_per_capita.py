import pandas as pd
from util import generateDataframe
import matplotlib.pyplot as plt

# Read US populations dataset
filename = 'NST-EST2022-ALLDATA.csv'
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
    if("2020" in row._date):
        targetStatePopulation = population_data.loc[population_data['NAME'] == targetState, 'POPESTIMATE2020'].item()
    elif("2021" in row._date):
        targetStatePopulation = population_data.loc[population_data['NAME'] == targetState, 'POPESTIMATE2021'].item()
    else:
        targetStatePopulation = population_data.loc[population_data['NAME'] == targetState, 'POPESTIMATE2022'].item()
    numConfirmed = row['Confirmed']
    return numConfirmed / targetStatePopulation

df['confirmed_per_capita'] = df.apply(calculateConfirmedPerCapita, axis=1)
df = df.filter(items=['Province_State', 'Confirmed', '_date', 'confirmed_per_capita'])
df['_date'] = pd.to_datetime(df['_date'])
df = df.sort_values(by=['_date'])


# Data Visualization


# # Visualize Data - Nevada
df_nv = df[df.Province_State == 'Nevada']
plt.figure(figsize=(10, 5))
plt.plot(df_nv["_date"], df_nv["confirmed_per_capita"])
plt.xlabel("Date")
plt.ylabel("Confirmed Cases per Capita")
plt.title("Nevada Confirmed Cases per Capita")
plt.show()

# # Gnerate Side Bar Graph - US States Confirmed Cases per Capita as of 2023-02-28
df_lastEntry = df[df._date == pd.to_datetime('2023-02-28')]
df_lastEntry = df_lastEntry.sort_values(by=['confirmed_per_capita'])
plt.figure(figsize=(12, 9))
plt.barh(y=df_lastEntry['Province_State'], width=df_lastEntry['confirmed_per_capita'])
plt.xlabel("Confirmed Cases per Capita")
plt.ylabel("State")
plt.title("US States Confirmed Cases per Capita as of 2023-02-28")
plt.show()

# Gnerate Side Bar Graph - US States Confirmed Cases per Capita as of 2023-02-28
df_lastEntry = df[df._date == pd.to_datetime('2023-02-28')]
df_lastEntry_top5 = df_lastEntry.sort_values(by=['confirmed_per_capita'], ascending=False)
df_lastEntry_top5 = df_lastEntry_top5.sort_values(by=['confirmed_per_capita'], ascending=False).head(5)
df_lastEntry_bottom5 = df_lastEntry.sort_values(by=['confirmed_per_capita']).head(5)
df_lastEntry_bottom5 = df_lastEntry_bottom5.sort_values(by=['confirmed_per_capita'], ascending=False).head(5)
df_lastEntry = pd.concat([df_lastEntry_top5, df_lastEntry_bottom5])
df_lastEntry = df_lastEntry.sort_values(by=["confirmed_per_capita"], ascending=True)
color = ["orange"]*5 + ["purple"]*5
plt.figure(figsize=(12, 8))
plt.barh(y=df_lastEntry['Province_State'], width=df_lastEntry['confirmed_per_capita'], color=color)
plt.xlabel("Confirmed Cases per Capita")
plt.ylabel("State")
plt.title("US States Confirmed Cases per Capita as of 2023-02-28")
plt.show()

# Plot trend in confirmed cases per capita for all 50 States
state_names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
plt.figure(figsize=(15, 12))
for state in state_names:
    df_state = df[df.Province_State == state]
    plt.plot(df_state["_date"], df_state["confirmed_per_capita"], label=state)
plt.legend(loc='upper left', ncol=3)
plt.xlabel("Date")
plt.ylabel("Confirmed Cases per Capita")
plt.title("US States Confirmed Cases per Capita")
plt.show()