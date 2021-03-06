import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from decimal import Decimal
from tqdm import tqdm
from utils.timer_utils import Timer
from utils.file_utils import write_data_file

#Aim is to conert open-high-low-close (ohlc) data into inputs and output categories for training a model
#Categories will be sell (0) and buy(1)
#Inputs will be slices of

#----------------------------------global variables----------------------------------
#variables
intervals = [] #input ohlc data periods to create, working from right to left decides how far from the target date
target_interval = timedelta(minutes=60) #The size of the target interval (for buying/selling)
market = 'EUR/GBP'
max_input_minutes_missing = 0 #allowance for missing data points from inputs
name = 'test'
df_width = 25000 #the size of the market data subset for batch processing
big_gain_boundary = 0.0001
big_loss_boundary = -0.0001

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

def set_name(new_name):
    global name
    name = new_name

def set_df_width(width):
    global df_width
    df_width = width

def set_big_gain_boundary(boundary):
    global big_gain_boundary
    big_gain_boundary = boundary

def set_big_loss_boundary(boundary):
    global big_loss_boundary
    big_loss_boundary = boundary

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

def get_name():
    global name
    return name

def get_df_width():
    global df_width
    return df_width

#----------------------------------process raw data from csv---------------------------------------------

def apply_category_label_binary(open, close):
    """ 0 = loss, 1 = gain
    """
    if close - open < 0:
        return 0
    else:
        return 1

def apply_category_label_4(open, close):
    """ 0 = 'big' loss, 1 = 'little' loss, 2 = 'little' gain, 3 = 'big' gain
    """
    if close - open < big_loss_boundary:
        return 0
    elif close - open < 0:
        return 1
    elif close - open > big_gain_boundary:
        return 3
    else:
        return 2

def load_market_csv(market):
    market = market.replace('/', '_')
    return pd.read_csv(f'data/{market}.csv')

def apply_category_label_for_dataframe(df):
    open = df.iloc[0]['open']
    close = df.iloc[-1]['close']
    return apply_category_label_binary(open, close)

def apply_binary_category_label_for_vector(x):
    open, close = x
    return apply_category_label_binary(open, close)

def apply_4_category_label_for_vector(x):
    open, close = x
    return apply_category_label_4(open, close)

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

#TODO write tests
def create_data(dates):
    """creates the csv files for training, validation and test

    Args:
        dates (list(datetime.datetime)): the 4 dates: training_start, training_end/validation_start, validation_end/test_start, test_end

    Returns:
        [type]: [description]
    """
    dataframe = load_market_csv(market)
    training_start, validation_start, test_start, test_end = dates
    training_dates, validation_dates, test_dates = get_dates(training_start, validation_start, test_start, test_end)       
    training_file_time = create_data_file(training_dates, dataframe, 'training')
    validation_file_time = create_data_file(validation_dates, dataframe, 'validation')
    test_file_time = create_data_file(test_dates, dataframe, 'test')
    return training_file_time, validation_file_time, test_file_time

#TODO write tests
def create_data_file(dates, dataframe, filename):
    """writes a csv file gathering OHLC data in the required format for the set of dates provided

    Args:
        dates (list[datetime.datetime]): a list of all of the dates to try and get data for
        dataframe (pandas.DataFrame): a dataframe containing all of the market data
        filename (str): data file name

    Returns:
        float: the time to run in seconds
    """
    with Timer() as T:
        print(f'Creating file {filename}.csv')
        start_index = find_start_date_index(dataframe, dates[0]) - sum(intervals)
        smaller_df = dataframe.iloc[start_index: start_index + df_width, :]
        rows = []
        iteration_count = 0
        for date in tqdm(dates):
            if iteration_count * sum(intervals) > df_width:
                write_data_file(filename, f'models/{name}/data', rows)
                start_index = find_start_date_index(dataframe, date)
                end_index = int(start_index + (target_interval.total_seconds()//60)) + df_width
                start_index -= sum(intervals)
                smaller_df = dataframe.iloc[start_index: end_index, :]
                iteration_count = 0
                rows = []
            relevant_df = get_relevant_data(smaller_df, date)
            if not relevant_df.empty:
                try:
                    relevant_data = create_relevant_data_row(relevant_df, date)
                    datestring = datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
                    row = np.hstack(([[datestring]], relevant_data)).flatten()
                    rows.append(row)
                except:
                    pass
            iteration_count += 1
        write_data_file(filename, f'models/{name}/data', rows)
    print(f'Time taken creating {filename}.csv: {T._context_timed}')
    return T._context_timed

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
    """Transform data points into a row with inputs based on the intervals and the open and low price for the given date

    Args:
        dataframe (pandas.dataframe): the selected datapoints for the given date
        target_date (datetime.datetime): the datetime for which we're creating data for

    Returns:
       numpy.array : a row of data that can be consumed by a model
    """
    start_date = target_date - timedelta(minutes=sum(intervals))
    input_df = get_dataframe_from_dates(start_date, target_date, dataframe)
    processed_inputs = process_input_data(input_df)
    outputs = get_open_and_close_for_period(dataframe, target_date)
    return create_row(processed_inputs, outputs)

def get_open_and_close_for_period(dataframe, target_date):
    """Locate the open price and close price for the target date given the global interval

    Args:
        dataframe (pandas.DataFrame): The raw data to search from
        target_date (datetime.datetime): The start time of the period

    Returns:
        list(int, int): open and close prices
    """
    end_date = target_date + target_interval - timedelta(minutes=1) #necessary for actual open / close
    open_row = dataframe.loc[dataframe['datetime'] == datetime.strftime(target_date, '%Y-%m-%d %H:%M:%S')]
    close_row = dataframe.loc[dataframe['datetime'] == datetime.strftime(end_date, '%Y-%m-%d %H:%M:%S')]
    if open_row.empty or close_row.empty:
        raise RuntimeError(f'Open-close data unavailable for {target_date} and interval of {int(target_interval.total_seconds()//60)} minutes')
    open = open_row.iloc[0]['open']
    close = close_row.iloc[0]['close']
    return open, close

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

def create_row(input_values, outputs):
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
    return np.hstack((values, [outputs]))
