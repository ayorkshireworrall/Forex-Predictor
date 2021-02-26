import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from decimal import Decimal
from tqdm import tqdm
from utils.timer_utils import Timer

#Aim is to conert open-high-low-close (ohlc) data into inputs and output categories for training a model
#Categories will be sell (0) and buy(1)
#Inputs will be slices of

#----------------------------------global variables----------------------------------
#variables
intervals = [] #input ohlc data periods to create, working from right to left decides how far from the target date
target_interval = timedelta(minutes=60) #The size of the target interval (for buying/selling)
market = 'EUR/GBP'
max_input_minutes_missing = 0

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

def set_max_input_minutes_missing(minutes):
    global max_input_minutes_missing
    max_input_minutes_missing = minutes

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

def get_max_input_minutes_missing():
    global max_input_minutes_missing
    return max_input_minutes_missing 

#----------------------------------process raw data from csv---------------------------------------------

def apply_category_label(open, close):
    if close - open < 0:
        return 0
    else:
        return 1

def load_market_csv(market):
    market = market.replace('/', '_')
    return pd.read_csv(f'data/{market}.csv')

def apply_category_label_for_dataframe(df):
    open = df.iloc[0]['open']
    close = df.iloc[-1]['close']
    return apply_category_label(open, close)

def get_dates(training_start, validation_start, test_start, test_end):
    """get dates for the various datapoints based on global config

    Args:
        training_start (datetime.datetime): the start date for the training data
        validation_start (datetime.datetime): the start date for validation data (and hence end of training data)
        test_start (datetime.datetime): the start date for test data (and hence end of validation data)
        test_end (datetime.datetime): the end date for the test data

    Returns:
        array: an array of each of the arrays of dates
    """
    global intervals
    training_dates = [training_start]
    validation_dates = [validation_start]
    test_dates = [test_start] 
    min_spacing = timedelta(minutes=sum(intervals))
    while(training_start < validation_start - min_spacing):
        training_start = training_start + min_spacing
        training_dates.append(training_start)
    while(validation_start < test_start - min_spacing ):
        validation_start = validation_start + min_spacing
        validation_dates.append(validation_start)
    while(test_start < test_end - min_spacing):
        test_start = test_start + min_spacing
        test_dates.append(test_start)
    return training_dates, validation_dates, test_dates        

#Not tested for now because is incomplete
def create_data(dates, df_width):
    with Timer() as T:
        dataframe = load_market_csv(market)
        training_start, validation_start, test_start, test_end = dates
        start_index = find_start_date_index(dataframe, start) - sum(intervals)
        smaller_df = dataframe.iloc[start_index: start_index + df_width, :]
        training_dates = get_dates(training_start, validation_start, test_start, test_end)[0]
        rows = []
        iteration_count = 0
        for date in tqdm(training_dates):
            if iteration_count * sum(intervals) > df_width:
                start_index = find_start_date_index(dataframe, date)
                end_index = int(start_index + (target_interval.total_seconds()//60)) + df_width
                start_index -= sum(intervals)
                smaller_df = dataframe.iloc[start_index: end_index, :]
                iteration_count = 0
                write_data_file(rows)
                rows = []
            relevant_df = get_relevant_data(smaller_df, date)
            if not relevant_df.empty:
                try:
                    rows.append(create_relevant_data_row(relevant_df, date))
                except:
                    pass
            iteration_count += 1
        write_data_file(rows)
    print(f'Time taken: {T._context_timed}')

def get_dataframe_from_dates(start_date, end_date, dataframe):
    """Generates a subset from a dataframe between two dates. Because it compares multiple string matching values 
    the dataframe should be small and the dates not too greatly separated to minimise computational cost

    Args:
        start_date (datetime.datetime): start date
        end_date (datetime.datetime): end date
        dataframe (pandas.Dataframe): full dataset

    Returns:
        pandas.Dataframe: subset dataframe
    """
    datestrings = []
    working_date = start_date
    while(working_date < end_date):
        datestrings.append(datetime.strftime(working_date, '%Y-%m-%d %H:%M:%S'))
        working_date = working_date + timedelta(minutes=1)
    return dataframe.loc[dataframe['datetime'].str.findall(f"({'|'.join(datestrings)})").str.join(', ').str.contains('-')]

def find_start_date_index(dataframe, target_date):
    """Finds the index of the data item with the closest date before the target date. It finds this because there is possibility that data is missing

    Args:
        dataframe (pandas.Dataframe): The full dataset dataframe
        target_date (datetime.datetime): the earliest date that should appear in the dataframe
        error (int): the amount of minutes before the target data that will be considered

    Returns:
        [pandas.DataFrame]: target item dataframe
    """
    earliest_date = target_date - timedelta(minutes=1000)
    acceptable_dates_df = get_dataframe_from_dates(earliest_date, target_date, dataframe)
    while acceptable_dates_df.empty:
        earliest_date -= timedelta(minutes=1000)
        acceptable_dates_df = get_dataframe_from_dates(earliest_date, target_date, dataframe)
    return acceptable_dates_df.index.tolist()[-1]

def get_relevant_data(dataframe, target_date):
    """Extracts the data which will be used for each input and output from the dataframe based on the target_date

    Args:
        dataframe ([type]): [description]
        target_date ([type]): [description]

    Returns:
        [type]: [description]
    """
    start_date = target_date - timedelta(minutes=sum(intervals))
    end_date = target_date + target_interval
    return get_dataframe_from_dates(start_date, end_date, dataframe)

def create_relevant_data_row(dataframe, target_date):
    """Transform data points into a row with inputs and one output for consumption by the machine

    Args:
        dataframe (pandas.dataframe): the selected datapoints for the given date
        target_date (datetime.datetime): the datetime for which we're creating data for

    Returns:
       numpy.array : a row of data that can be consumed by a model
    """
    start_date = target_date - timedelta(minutes=sum(intervals))
    end_date = target_date + target_interval
    input_df = get_dataframe_from_dates(start_date, target_date, dataframe)
    output_df = get_dataframe_from_dates(target_date, end_date, dataframe)
    processed_inputs = process_input_data(input_df)
    output_category = apply_category_label_for_dataframe(output_df)
    return create_row(processed_inputs, output_category)

def process_input_data(dataframe):
    """converts minute by minute OHLC data into OHLC data for the global intervals

    Args:
        dataframe (pandas.DataFrame): the input OHLC data points in a dataframe

    Raises:
        RuntimeError: when there are not enough data points

    Returns:
        pandas.DataFrame: reduced DataFrame
    """
    if dataframe.shape[0] < sum(intervals) - max_input_minutes_missing:
        raise RuntimeError('Insufficient data to process for this number of intervals')

    micro_frames = []
    i = 0
    for interval in intervals:
        micro_frames.append(dataframe[i:(i + interval)])
        i = i + interval
    processed_rows = []
    for micro_frame in micro_frames:
        date = micro_frame.iloc[0]['datetime']
        open_price = micro_frame.iloc[0]['open']
        high = micro_frame['high'].max()
        low = micro_frame['low'].min()
        close_price = micro_frame.iloc[-1]['close']
        appendable = {'datetime':date, 'open':open_price, 'high':high, 'low':low, 'close':close_price}
        processed_rows.append(appendable)
    return pd.DataFrame(processed_rows) 

def create_row(input_values, output_category):
    """Converts input dataframe and the corresponding output category into a numpy array

    Args:
        input_values (pandas.DataFrame): the processed OHLC data (to match global intervals)
        output_category (int): output category

    Returns:
        numpy.array: a data row that can be consumed by a machine learning model
    """
    values = input_values.iloc[:, 1:].values
    values = values.flatten()
    values = values.reshape(1, len(values))
    values = values[:, :len(intervals)*4]
    start_value = values[0][0]
    values = values[:, 1:]
    for i in range(0, len(values[0])):
        values[0][i] = Decimal(str(start_value)) - Decimal(str(values[0][i]))
    return np.hstack((values, [[output_category]]))

def write_data_file(rows):
    pass

# set_intervals([15,15,15,15])
# start = datetime.strptime('2014-05-22 09:55:00', '%Y-%m-%d %H:%M:%S')
# end = datetime.strptime('2014-06-22 12:59:00', '%Y-%m-%d %H:%M:%S')
# dates = [start, end, end, end]
# all_dataframes = create_data(dates, 25000)
# print('finished')
