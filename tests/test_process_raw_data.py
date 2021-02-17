import unittest
from datetime import timedelta
from unittest.mock import patch
from forex_predictor.process_raw_data import get_market, set_intervals, set_const_intervals, set_market, set_target_interval, get_intervals, get_target_interval, apply_category_label, load_market_csv


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

if __name__ == '__main__':
    unittest.main()