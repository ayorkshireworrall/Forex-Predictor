import pandas as pd
from datetime import timedelta

#Aim is to conert open-high-low-close (ohlc) data into inputs and output categories for training a model
#Categories will be sell (0) and buy(1)
#Inputs will be slices of

#----------------------------------global variables----------------------------------
#variables
intervals = [] #input ohlc data periods to create, working from right to left decides how far from the target date
target_interval = timedelta(minutes=60) #The size of the target interval (for buying/selling)
market = 'EUR/GBP'

#setters
def set_intervals(interval_array):
    global intervals
    intervals = interval_array

def set_const_intervals(interval_length, num_intervals):
    intervals = [interval_length] * num_intervals
    set_intervals(intervals)

def set_target_interval(interval):
    global target_interval
    target_interval = interval

def set_market(new_market):
    global market
    market = new_market

#getters(mainly for tests)
def get_intervals():
    global intervals
    return intervals

def get_target_interval():
    global target_interval
    return target_interval

def get_market():
    global market
    return market

#----------------------------------process raw data from csv---------------------------------------------

def apply_category_label(open, close):
    if close - open < 0:
        return 0
    else:
        return 1

def load_market_csv(market):
    market = market.replace('/', '_')
    return pd.read_csv(f'data/{market}.csv')

def apply_category_label_for_date(dataframe, datetime):
    open = df.iloc[0]['open']
    close = df.iloc[-1]['close']
    return apply_category_label(open, close)