import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.file_utils import write_data_file

name = 'test4'

def calculate_profit(row):
    multiplier = 1
    if row['Decision'] == 'Sell':
        multiplier = -1
    return row['Difference'] * multiplier

def convert_date(row):
    return datetime.strptime(row['Datetime'], '%Y-%m-%d %H:%M:%S')

def get_day_of_week(row):
    date = row['Datetime']
    return date.weekday()

def get_hour(row):
    date = row['Datetime']
    return date.hour

def get_minute(row):
    date = row['Datetime']
    return date.minute

def filter_dataframe_by_days(dataframe, days):
    return dataframe.loc[dataframe['Weekday'].isin(days)]

def filter_by_hours(dataframe, hours):
    return dataframe.loc[dataframe['Hour'].isin(hours)]

def filter_by_minutes(dataframe, minutes):
    return dataframe.loc[dataframe['Minute'].isin(minutes)]

def get_header_row(dataframe):
    return ','.join(list(dataframe.columns))

def append_moving_wealth_row(dataframe, start_wealth):
    df = dataframe.reset_index()
    wealth_array = [start_wealth]
    for i in range(1, len(df)):
        wealth_array.append(wealth_array[-1] * (1 + df.loc[i, 'Profit']))
    dataframe['Wealth'] = np.array(wealth_array)


base_data = pd.read_csv(f'models/{name}/ann/analysis/base_results.csv') 
base_data['Profit'] = base_data.apply(lambda row: calculate_profit(row), axis=1)
base_data['Datetime'] = base_data.apply(lambda row: convert_date(row), axis=1)
base_data['Weekday'] = base_data.apply(lambda row: get_day_of_week(row), axis=1)
base_data['Hour'] = base_data.apply(lambda row: get_hour(row), axis=1)
base_data['Minute'] = base_data.apply(lambda row: get_minute(row), axis=1)
append_moving_wealth_row(base_data, 10000)

max_profit = base_data['Profit'].max()
max_loss = base_data['Profit'].min()

mondays = filter_dataframe_by_days(base_data, [0])
append_moving_wealth_row(mondays, 10000)
tuesdays = filter_dataframe_by_days(base_data, [1])
wednesdays = filter_dataframe_by_days(base_data, [2])
thursdays = filter_dataframe_by_days(base_data, [3])
fridays = filter_dataframe_by_days(base_data, [4])
weekdays_1800_2200 = filter_by_hours(base_data, [18,19,20,21,22])

def print_analysis(dataframe, name):
    num_trades = len(dataframe.index)
    total_profit = dataframe.sum()['Profit']
    avg_profit = dataframe.mean()['Profit']
    max_profit = dataframe.max()['Profit']
    max_loss = dataframe.min()['Profit']
    print(f'\n\nSummary for data filtered by {name}')
    print('-------------------------------------------')
    print(f'Total Trades Made: {num_trades}')
    print(f'Total Profit Made: {total_profit}')
    print(f'Average Profit Made Per Trade: {avg_profit}')
    print(f'Largest Gain: {max_profit}')
    print(f'Largest Loss: {max_loss}')
    print('-------------------------------------------')

def plot_dataframe(dataframe):
    dataframe.plot(x='Datetime', y='Wealth')

print_analysis(mondays, 'mondays')
print_analysis(tuesdays, 'tuesdays')
print_analysis(wednesdays, 'wednesdays')
print_analysis(thursdays, 'thursdays')
print_analysis(fridays, 'fridays')
print_analysis(weekdays_1800_2200, 'weekdays 6pm-10pm')

plt.close('all')
plot_dataframe(mondays)
plot_dataframe(base_data)

plt.show()





