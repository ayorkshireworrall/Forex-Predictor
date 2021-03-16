from decimal import Decimal
import unittest, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
from forex_predictor.data_extraction.process_raw_data import create_relevant_data_row, create_row, find_start_date_index, get_dataframe_from_dates, get_dates, get_market, get_max_input_minutes_missing, get_open_and_close_for_period, get_relevant_data, process_input_data, set_intervals, set_const_intervals, set_market, set_max_input_minutes_missing, set_target_interval, get_intervals, get_target_interval, apply_category_label_binary, load_market_csv


class Test_Process_Raw_Data(unittest.TestCase):

    #Test helper methods
    def test_convert_datestring_array_to_datetime(self):
        datestrings = ['2020-01-01 00:00:00', '2020-01-02 00:00:00', '2020-01-01 03:00:00']
        expected_datetimes = [datetime.strptime('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2020-01-02 00:00:00', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2020-01-01 03:00:00', '%Y-%m-%d %H:%M:%S')]
        self.assertEqual(expected_datetimes, convert_datestring_array_to_datetime(datestrings))

    def test_create_expected_row(self):
        input_row = [5,4,3,2,1]
        expected_row = np.array([[1,2,3,4,1,2]])
        actual_row = create_expected_row(input_row, [1,2])
        self.assertTrue(np.array_equal(expected_row, actual_row))

    #Test process_raw_data methods
    def test_set_intervals(self):
        intervals = [5, 5, 5]
        set_intervals(intervals)
        self.assertEqual(intervals, get_intervals())
    
    def test_set_target_interval(self):
        interval = timedelta(minutes=69)
        set_target_interval(interval)
        self.assertEqual(interval, get_target_interval())

    def test_set_const_intervals(self):
        expected_intervals = [3, 3, 3, 3, 3]
        set_const_intervals(3, 5)
        self.assertEqual(expected_intervals, get_intervals())

    def test_set_max_input_minutes_missing(self):
        minutes = 69
        set_max_input_minutes_missing(minutes)
        self.assertEqual(minutes, get_max_input_minutes_missing())

    def test_set_market(self):
        market = 'GBP/JPY'
        set_market(market)
        self.assertEqual(market, get_market())

    def test_categorise_data(self):
        self.assertEqual(1, apply_category_label_binary(1.2222, 1.2223))
        self.assertEqual(0, apply_category_label_binary(1.2223, 1.2222))

    @patch('forex_predictor.data_extraction.process_raw_data.pd')
    def test_load_market_csv(self, mock_pd):
        load_market_csv('EUR/GBP')
        mock_pd.read_csv.assert_called_with('data/EUR_GBP.csv')

    def test_get_dates(self):
        intervals = [5, 5, 5]
        set_intervals(intervals)
        training_start = datetime.strptime('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        validation_start = datetime.strptime('2020-01-01 01:00:00', '%Y-%m-%d %H:%M:%S')
        test_start = datetime.strptime('2020-01-01 02:00:00', '%Y-%m-%d %H:%M:%S')
        test_end = datetime.strptime('2020-01-01 03:00:00', '%Y-%m-%d %H:%M:%S')
        actual_training_dates, actual_validation_dates, actual_test_dates = get_dates(training_start, validation_start, test_start, test_end)
        expected_training_dates = convert_datestring_array_to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:15:00', '2020-01-01 00:30:00', '2020-01-01 00:45:00'])
        expected_validation_dates = convert_datestring_array_to_datetime(['2020-01-01 01:00:00', '2020-01-01 01:15:00', '2020-01-01 01:30:00', '2020-01-01 01:45:00'])
        expected_test_dates = convert_datestring_array_to_datetime(['2020-01-01 02:00:00', '2020-01-01 02:15:00', '2020-01-01 02:30:00', '2020-01-01 02:45:00'])
        self.assertEqual(expected_training_dates, actual_training_dates)
        self.assertEqual(expected_validation_dates, actual_validation_dates)
        self.assertEqual(expected_test_dates, actual_test_dates)

    @patch('forex_predictor.data_extraction.process_raw_data.get_dataframe_from_dates')
    def test_get_relevant_data(self, mock_method):
        set_intervals([15,15,15,15])
        set_target_interval(timedelta(minutes=60))
        df = pd.read_csv('tests/resources/dataframe_data.csv')
        target_date = datetime.strptime('2014-07-17 00:00:00', '%Y-%m-%d %H:%M:%S')
        get_relevant_data(df, target_date)
        start_date = datetime.strptime('2014-07-16 23:00:00', '%Y-%m-%d %H:%M:%S')
        end_date = datetime.strptime('2014-07-17 01:00:00', '%Y-%m-%d %H:%M:%S')
        mock_method.assert_called_with(start_date, end_date, df)

    def test_get_dataframe_from_dates(self):
        original_df = pd.read_csv('tests/resources/dataframe_data.csv')
        start_date = datetime.strptime('2014-07-17 00:00:00', '%Y-%m-%d %H:%M:%S')
        end_date = datetime.strptime('2014-07-17 00:05:00', '%Y-%m-%d %H:%M:%S')
        actual_df = get_dataframe_from_dates(start_date, end_date, original_df)
        expected_df = original_df.iloc[74:79, :]
        self.assertTrue(expected_df.equals(actual_df))

    def test_find_start_date_index(self):
        target_date = datetime.strptime('2014-07-18 08:46:00', '%Y-%m-%d %H:%M:%S')
        df = pd.read_csv('tests/resources/dataframe_data.csv') 
        actual_index = find_start_date_index(df, target_date)
        expected_index = 1994
        self.assertEqual(expected_index, actual_index)

    def test_process_input_data(self):
        set_intervals([5, 5, 5])
        df = pd.read_csv('tests/resources/dataframe_data.csv').iloc[1998:2013, :]
        test_data = {
            'datetime': ['2014-07-18 08:49:00', '2014-07-18 08:54:00', '2014-07-18 08:59:00'],
            'open': [0.79227, 0.79223, 0.79315],
            'high': [0.79231, 0.79312, 0.79325],
            'low': [0.79216, 0.79219, 0.79279],
            'close': [0.79222, 0.79312, 0.79284]
        }
        expected_input_data = pd.DataFrame(data=test_data)
        actual_input_data = process_input_data(df)
        self.assertTrue(expected_input_data.equals(actual_input_data))

    def test_process_input_data_error(self):
        set_intervals([5, 5, 5, 60])
        df = pd.read_csv('tests/resources/dataframe_data.csv').iloc[1998:2013, :]
        expected_error_message = 'Insufficient data to process for this number of intervals'
        try:
            actual_input_data = process_input_data(df)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info() 
        self.assertEqual(expected_error_message, str(exc_value))


    def test_create_row(self):
        set_intervals([5,5,5])
        test_data = {
            'datetime': ['2014-07-18 08:49:00', '2014-07-18 08:54:00', '2014-07-18 08:59:00'],
            'open': [0.79227, 0.79223, 0.79315],
            'high': [0.79231, 0.79312, 0.79325],
            'low': [0.79216, 0.79219, 0.79279],
            'close': [0.79222, 0.79312, 0.79284]
        }
        input_values = pd.DataFrame(data=test_data)
        expected_row = create_expected_row([0.79227, 0.79231, 0.79216, 0.79222, 0.79223, 0.79312, 0.79219, 0.79312, 0.79315, 0.79325, 0.79279, 0.79284], [1, 2])
        actual_row = create_row(input_values, [1,2])
        self.assertTrue(np.array_equal(expected_row, actual_row))

    def test_create_relevant_data_row(self):
        set_intervals([5,5,5])
        set_target_interval(timedelta(minutes=5))
        df = pd.read_csv('tests/resources/dataframe_data.csv').iloc[1998:2018, :]
        expected_row = create_expected_row([0.79227, 0.79231, 0.79216, 0.79222, 0.79223, 0.79312, 0.79219, 0.79312, 0.79315, 0.79325, 0.79279, 0.79284], [0.79283, 0.79258])
        actual_row = create_relevant_data_row(df, datetime.strptime('2014-07-18 09:04:00', '%Y-%m-%d %H:%M:%S'))
        self.assertTrue(np.array_equal(expected_row, actual_row))

    def test_get_open_and_close_for_period(self):
        set_target_interval(timedelta(minutes=60))
        df = pd.read_csv('tests/resources/dataframe_data.csv') 
        start_date = datetime.strptime('2014-07-21 18:00:00', '%Y-%m-%d %H:%M:%S')
        open, close = get_open_and_close_for_period(df, start_date)
        self.assertEqual(0.79194, open)
        self.assertEqual(0.79193, close)

    def test_get_open_and_close_for_period_error(self):
        set_target_interval(timedelta(minutes=60))
        df = pd.read_csv('tests/resources/dataframe_data.csv') 
        start_date = datetime.strptime('2014-07-21 19:00:00', '%Y-%m-%d %H:%M:%S')
        expected_error_message = 'Open-close data unavailable for 2014-07-21 19:00:00 and interval of 60 minutes'
        try:
            open, close = get_open_and_close_for_period(df, start_date)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
        self.assertEqual(expected_error_message, str(exc_value))

        
        


def convert_datestring_array_to_datetime(datestrings):
    """For readability when working with large amounts of datetimes
    """
    datetimes = []
    for datestring in datestrings:
        datetimes.append(datetime.strptime(datestring, '%Y-%m-%d %H:%M:%S'))
    return datetimes

def create_expected_row(input_row, outputs):
    """Create a row similar to how it is done in process_raw_data.py but with the advantage that this takes inputs as a python list 
        making it much easier to test. Can then use it in more integrated test with expected dataframe values
    """
    values = np.array([input_row])
    start_value = values[0][0]
    values = values[:, 1:]
    for i in range(0, len(values[0])):
        values[0][i] = Decimal(str(start_value)) - Decimal(str(values[0][i]))
    return np.hstack((values, [outputs]))
