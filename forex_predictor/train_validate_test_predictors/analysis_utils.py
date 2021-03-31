import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_profit(row):
    if row['Decision'] == 'Sell':
        return row['Difference'] * -1
    elif row['Decision'] == 'Buy':
        return row['Difference']
    else:
        return 0

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

def append_moving_wealth_row(dataframe, start_wealth, bid_ask_spread=0, wealth_column_name='Wealth'):
    if bid_ask_spread and wealth_column_name == 'Wealth':
        wealth_column_name='Adjusted Wealth'
    df = dataframe.reset_index()
    wealth_array = [start_wealth]
    for i in range(1, len(df)):
        if df.loc[i, 'Decision'] =='None':
            wealth_array.append(wealth_array[-1])
        else:
            wealth_array.append(wealth_array[-1] * (1 + df.loc[i, 'Profit'] - bid_ask_spread))
    dataframe[wealth_column_name] = np.array(wealth_array)

def plot_dataframe(dataframe, title, wealth_column_name='Wealth', adjusted_wealth_column_name='Adjusted Wealth'):
    fig, ax = plt.subplots()
    ax = dataframe.plot(ax=ax, x='Datetime', y=wealth_column_name, c='blue', title=title)
    ax = dataframe.plot(ax=ax, x='Datetime', y=adjusted_wealth_column_name, c='red', title=title)

def print_analysis(dataframe, filter_name):
    num_trades = len(dataframe.index)
    total_profit = dataframe.sum()['Profit']
    avg_profit = dataframe.mean()['Profit']
    max_profit = dataframe.max()['Profit']
    max_loss = dataframe.min()['Profit']
    print(f'\n\nSummary for data filtered by {filter_name}')
    print('-------------------------------------------')
    print(f'Total Trades Made: {num_trades}')
    print(f'Total Profit Made: {total_profit}')
    print(f'Average Profit Made Per Trade: {avg_profit}')
    print(f'Largest Gain: {max_profit}')
    print(f'Largest Loss: {max_loss}')
    print('-------------------------------------------')

def add_columns_to_base_data(base_data, start_wealth=10000):
    base_data['Profit'] = base_data.apply(lambda row: calculate_profit(row), axis=1)
    base_data['Datetime'] = base_data.apply(lambda row: convert_date(row), axis=1)
    base_data['Weekday'] = base_data.apply(lambda row: get_day_of_week(row), axis=1)
    base_data['Hour'] = base_data.apply(lambda row: get_hour(row), axis=1)
    base_data['Minute'] = base_data.apply(lambda row: get_minute(row), axis=1)
    # append_moving_wealth_row(base_data, start_wealth)
    # append_moving_wealth_row(base_data, start_wealth, wealth_column_name='Adjusted_Wealth', bid_ask_spread=0.0002)