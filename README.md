# CS458 Final Project Group R1

Make a COVID-19 dataset for Tensorflow and perform simple analysis. Extract data from https://github.com/CSSEGISandData/COVID-19 (or any other COVID-19 repo that is not formatted in Tensorflow format). Then use Tensorflow’s API to create a dataset that is ready for Tensorflow: https://www.tensorflow.org/datasets/add_dataset. Please make sure you clean the data first. You need to separate the data into training and evaluation datasets. Please also make some simple analysis for the data as well: e.g., compute the mean, standard deviation, variance, ADD, MAD, and distribution that we introduced in the class. You can also use visualization to demonstrate the data and your analysis results.

## Install Necessary Libraries

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install -r requirements.txt
```

## Usage

To analyze statistics per capita, run:
```bash
python stats_per_capita.py
```
This will generate 4 graphs:

1. Nevada's Confirmed Cases Per Capita from 2020 - 2023

2. Confirmed Cases Per Capita for all 50 US States as of 2023-02-28

3. Top 5 and Bottom 5 US States for Number of Confirmed Cases Per Capita

4. Overall view of the increase in confirmed cases per capita for all 50 states

to analyze the daily stats of nevada for an entire year overall confirmed cases:
```bash
python display.py
```

1. this graph will display 2020 to 2023 overall comfirmed cases 

2. regional trends for the year 2022 (it will bring in a 16 by 16 there will be multiple pages)

3. Nevada daily confrimed cases on a monthly basis. (there will be a page per month.)

to get the raw data stats and display dataframes of the covid cases on state and monthly avg use

```bash
python3 examples.py
```

1. Prints out mean, std, var for 2020-23

2. Displays monthly average data for 2020-23

3. Displays Dataframes for each state with its info
