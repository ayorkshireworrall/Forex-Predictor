import pandas as pd
import matplotlib.pyplot as plt
from forex_predictor.train_validate_test_predictors.analysis_utils import append_moving_wealth_row, filter_by_buy_or_sell, filter_dataframe_by_days, filter_by_hours, print_analysis, plot_dataframe, add_columns_to_base_data

name = 'test4'

base_data = pd.read_csv(f'models/{name}/xgboost_4_options/analysis/base_results.csv') 
add_columns_to_base_data(base_data)

#Create filtered dataframes
sells = filter_by_buy_or_sell(base_data, 'Sell')
buys = filter_by_buy_or_sell(base_data, 'Buy')

#Print analysis summaries of dataframes
print_analysis(base_data, 'Base Data')
print_analysis(sells, 'Sells')
print_analysis(buys, 'Buys')

#Append moving wealth columns to dataframes for plotting timeseries data
append_moving_wealth_row(base_data, 10000)
append_moving_wealth_row(base_data, 10000, bid_ask_spread=0.0002)
append_moving_wealth_row(base_data, 10000, wealth_column_name='Leverage', bid_ask_spread=0.0002, leverage=10)

append_moving_wealth_row(sells, 10000)
append_moving_wealth_row(sells, 10000, bid_ask_spread=0.0002)
append_moving_wealth_row(sells, 10000, wealth_column_name='Leverage', bid_ask_spread=0.0002, leverage=10)

append_moving_wealth_row(buys, 10000)
append_moving_wealth_row(buys, 10000, bid_ask_spread=0.0002)
append_moving_wealth_row(buys, 10000, wealth_column_name='Leverage', bid_ask_spread=0.0002, leverage=10)

append_moving_wealth_row(base_data, 10000)
append_moving_wealth_row(base_data, 10000, bid_ask_spread=0.0002)
append_moving_wealth_row(base_data, 10000, wealth_column_name='Leverage', bid_ask_spread=0.0002, leverage=10)

#Plot timeseries wealth gains
plt.close('all')#
plot_dataframe(base_data, 'All Data', ['Wealth', 'Adjusted Wealth', 'Leverage'])
plot_dataframe(buys, 'Buys', ['Wealth', 'Adjusted Wealth', 'Leverage'])
plot_dataframe(sells, 'Sells', ['Wealth', 'Adjusted Wealth', 'Leverage'])

plt.show()