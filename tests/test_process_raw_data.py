import unittest, sys, os
path = os.path.abspath(os.path.dirname('forex_predictor/process_raw_data.py'))
if path not in sys.path:
    sys.path.insert(0, path)
from datetime import timedelta
from process_raw_data import set_intervals, set_const_intervals, set_target_interval, get_intervals, get_target_interval


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