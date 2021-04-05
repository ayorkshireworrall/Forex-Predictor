import numpy as np
import pandas as pd
import pickle
from utils.file_utils import write_data_file

#Match the name of data generated in other module
name = 'test4'

#Load trained model
path = f'models/{name}/xgboost_4_options/pickle'
with open(f"{path}/classifier.pk", 'rb') as rf_pickle_file:
    classifier = pickle.load(rf_pickle_file)

#Load test dataset
test_dataset = pd.read_csv(f'models/{name}/data/test.csv')
X_test = test_dataset.iloc[:, 1:-2].values
y_test_outputs_dates = test_dataset.iloc[:, [0, -2, -1]].values
y_test_outputs = y_test_outputs_dates[:, [1,2]]

#Make predictions
y_pred = classifier.predict(X_test)
buy_pred = y_pred == 3
sell_pred = y_pred == 0

#Save predictions and actual price differences for future analysis
def calculate_change(x):
    open, close = x
    return close - open
actual_price_change = np.apply_along_axis(calculate_change, 1, y_test_outputs)
results_array = np.hstack((y_test_outputs_dates, np.transpose([actual_price_change]), np.transpose([buy_pred]), np.transpose([sell_pred])))
results_array = np.vstack((['Datetime', 'Open', 'Close', 'Difference', 'Buy', 'Sell'], results_array))
write_data_file('base_results', f'models/{name}/xgboost_4_options/analysis', results_array)