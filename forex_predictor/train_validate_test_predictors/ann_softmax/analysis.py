import pandas as pd
import matplotlib.pyplot as plt
from forex_predictor.train_validate_test_predictors.analysis_utils import append_moving_wealth_row, filter_by_buy_or_sell, filter_dataframe_by_days, filter_by_hours, print_analysis, plot_dataframe, add_columns_to_base_data

name = 'test4'

base_data = pd.read_csv(f'models/{name}/ann_softmax/analysis/base_results.csv') 
add_columns_to_base_data(base_data)

#Create filtered dataframes
mondays = filter_dataframe_by_days(base_data, [0])
tuesdays = filter_dataframe_by_days(base_data, [1])
wednesdays = filter_dataframe_by_days(base_data, [2])
thursdays = filter_dataframe_by_days(base_data, [3])
fridays = filter_dataframe_by_days(base_data, [4])
weekdays_1800_2200 = filter_by_hours(base_data, [18,19,20,21,22])
buys = filter_by_buy_or_sell(base_data, 'Buy')
sells = filter_by_buy_or_sell(base_data, 'Sell')

#Print analysis summaries of dataframes
print_analysis(base_data, 'All data')
print_analysis(mondays, 'mondays')
print_analysis(tuesdays, 'tuesdays')
print_analysis(wednesdays, 'wednesdays')
print_analysis(thursdays, 'thursdays')
print_analysis(fridays, 'fridays')
print_analysis(weekdays_1800_2200, 'weekdays 6pm-10pm')
print_analysis(buys, 'buys')
print_analysis(sells, 'sells')

#Append moving wealth columns to dataframes for plotting timeseries data
append_moving_wealth_row(base_data, 10000)
append_moving_wealth_row(base_data, 10000, bid_ask_spread=0.0002)
append_moving_wealth_row(base_data, 10000, wealth_column_name='Leverage', bid_ask_spread=0.0002, leverage=10)
append_moving_wealth_row(mondays, 10000)
append_moving_wealth_row(mondays, 10000, bid_ask_spread=0.0002)
#Plot timeseries wealth gains
plt.close('all')
plot_dataframe(base_data, 'All Data', ['Wealth', 'Adjusted Wealth', 'Leverage'])
plot_dataframe(mondays, 'Mondays', ['Wealth', 'Adjusted Wealth'])
plt.show()





