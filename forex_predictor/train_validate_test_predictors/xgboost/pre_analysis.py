import numpy as np
import pandas as pd
import pickle
from utils.file_utils import write_data_file

#Changeable variables
name = 'test4'

#Load trained model
path = f'models/{name}/xgboost/pickle/binary'
with open(f"{path}/classifier.pk", 'rb') as rf_pickle_file:
    classifier = pickle.load(rf_pickle_file)

#Load training dataset
training_dataset = pd.read_csv(f'models/{name}/data/test.csv')
X_test = training_dataset.iloc[:, 1:-2].values
y_test_outputs_dates = training_dataset.iloc[:, [0, -2, -1]].values
y_test_outputs = y_test_outputs_dates[:, [1,2]]

#Make predictions (configure probabilities depending on desired model confidence)
y_pred = classifier.predict(X_test)
buy_pred = y_pred > 0.55 
sell_pred = y_pred < 0.45

#Save predictions and actual price differences for future analysis
def calculate_price_change(x):
    open, close = x
    return open - close
actual_price_change = np.apply_along_axis(calculate_price_change, 1, y_test_outputs)
results_array = np.hstack((y_test_outputs_dates, np.transpose([actual_price_change]), np.transpose([buy_pred]), np.transpose([sell_pred])))
results_array = np.vstack((['Datetime', 'Open', 'Close', 'Difference', 'Buy', 'Sell'], results_array))

write_data_file('base_results', f'models/{name}/xgboost/analysis', results_array)