import pandas as pd
import matplotlib.pyplot as plt
from forex_predictor.train_validate_test_predictors.analysis_utils import append_moving_wealth_row, filter_dataframe_by_days, filter_by_hours, print_analysis, plot_dataframe, add_columns_to_base_data

name = 'test4'

base_data = pd.read_csv(f'models/{name}/xgboost/analysis/base_results.csv') 
add_columns_to_base_data(base_data)

max_profit = base_data['Profit'].max()
max_loss = base_data['Profit'].min()

mondays = filter_dataframe_by_days(base_data, [0])
append_moving_wealth_row(mondays, 10000)
append_moving_wealth_row(mondays, 10000, bid_ask_spread=0.0002)
tuesdays = filter_dataframe_by_days(base_data, [1])
wednesdays = filter_dataframe_by_days(base_data, [2])
append_moving_wealth_row(wednesdays, 10000)
append_moving_wealth_row(wednesdays, 10000, bid_ask_spread=0.0002)
thursdays = filter_dataframe_by_days(base_data, [3])
fridays = filter_dataframe_by_days(base_data, [4])
weekdays_1800_2200 = filter_by_hours(base_data, [18,19,20,21,22])

print_analysis(mondays, 'mondays')
print_analysis(tuesdays, 'tuesdays')
print_analysis(wednesdays, 'wednesdays')
print_analysis(thursdays, 'thursdays')
print_analysis(fridays, 'fridays')
print_analysis(weekdays_1800_2200, 'weekdays 6pm-10pm')

plt.close('all')
plot_dataframe(mondays, 'Mondays')
plot_dataframe(wednesdays, 'Wednesdays')


append_moving_wealth_row(base_data, 10000)
append_moving_wealth_row(base_data, 10000, bid_ask_spread=0.0002)
plot_dataframe(base_data, 'All Data')

plt.show()





