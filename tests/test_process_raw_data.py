import unittest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch
from forex_predictor.process_raw_data import find_start_date_index, get_dataframe_from_dates, get_dates, get_market, get_relevant_data, set_intervals, set_const_intervals, set_market, set_target_interval, get_intervals, get_target_interval, apply_category_label, load_market_csv


class Test_Process_Raw_Data(unittest.TestCase):

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

    def test_set_market(self):
        market = 'GBP/JPY'
        set_market(market)
        self.assertEqual(market, get_market())

    def test_categorise_data(self):
        self.assertEqual(1, apply_category_label(1.2222, 1.2223))
        self.assertEqual(0, apply_category_label(1.2223, 1.2222))

    @patch('forex_predictor.process_raw_data.pd')
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

    @patch('forex_predictor.process_raw_data.get_dataframe_from_dates')
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

def convert_datestring_array_to_datetime(datestrings):
    """For readability when working with large amounts of datetimes
    """
    datetimes = []
    for datestring in datestrings:
        datetimes.append(datetime.strptime(datestring, '%Y-%m-%d %H:%M:%S'))
    return datetimes