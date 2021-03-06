#Edit values in here and run to produce new consumable data
from datetime import timedelta, datetime 
from forex_predictor.data_extraction.process_raw_data import create_data, set_df_width, set_intervals, set_name, set_market, set_target_interval, set_max_input_minutes_missing
from utils.file_utils import write_meta_data_file

#---------------------------------------Editable variables------------------------------------------

name = 'test2' #Make sure name doesn't conflict with any existing files before running
intervals = [2] * 30
target_interval_minutes = 40
market = 'EUR/GBP'
input_minutes_missing_allowance = 2
training_start_str = '2015-01-01 00:00:00'
validation_start_str = '2018-01-01 00:00:00'
test_start_str = '2019-01-01 00:00:00'
test_end_str = '2021-01-01 00:00:00'
dataframe_batch_size = 20000

#---------------------------------Script code do not modify------------------------------------------
set_name(name) 
set_intervals(intervals)
set_target_interval(timedelta(minutes=target_interval_minutes))
set_market(market)
set_max_input_minutes_missing(input_minutes_missing_allowance)
set_df_width(dataframe_batch_size)

training_start = datetime.strptime(training_start_str, '%Y-%m-%d %H:%M:%S')
validation_start = datetime.strptime(validation_start_str, '%Y-%m-%d %H:%M:%S')
test_start = datetime.strptime(test_start_str, '%Y-%m-%d %H:%M:%S')
test_end = datetime.strptime(test_end_str, '%Y-%m-%d %H:%M:%S')
dates = [training_start, validation_start, test_start, test_end]

timings = create_data([training_start, validation_start, test_start, test_end])

meta_data = {
    'name': name,
    'intervals': intervals,
    'target_interval_minutes': target_interval_minutes,
    'market': market,
    'training_start': training_start_str,
    'validation_start': validation_start_str,
    'test_start': test_start_str,
    'test_end': test_end_str,
    'dataframe_batch_size': dataframe_batch_size,
    'time_to_write_training': timings[0],
    'time_to_write_validation': timings[1],
    'time_to_write_test': timings[2]
}

write_meta_data_file(f'models/{name}/data', meta_data)

