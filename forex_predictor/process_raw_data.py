from datetime import timedelta

#Aim is to conert open-high-low-close (ohlc) data into inputs and output categories for training a model
#Categories will be sell (0) and buy(1)
#Inputs will be slices of

#----------------------------------global variables----------------------------------
#variables
intervals = [] #input ohlc data periods to create, working from right to left decides how far from the target date
target_interval = timedelta(minutes=60) #The size of the target interval (for buying/selling)

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

#getters(mainly for tests)
def get_intervals():
    global intervals
    return intervals

def get_target_interval():
    global target_interval
    return target_interval
#--------------------------------------------------------------------------------------
