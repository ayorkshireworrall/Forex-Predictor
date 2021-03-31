import numpy as np
import pandas as pd
import pickle
from forex_predictor.data_extraction.process_raw_data import apply_4_category_label_for_vector, apply_binary_category_label_for_vector, set_big_gain_boundary, set_big_loss_boundary
from utils.file_utils import write_data_file

def calculate_gains(x):
    open, close = x
    return open - close

def buy_or_sell(y_pred):
    if y_pred == 0:
        return 'Sell'
    elif y_pred == 3:
        return 'Buy'
    else:
        return None

#Changeable variables
name = 'test4'
binary_categories = False
big_gain_boundary = 0.0003
big_loss_boundary = -0.0003

#Categorise based on buy/sell or big buy/ big sell
if binary_categories:
    categorisation_method = apply_binary_category_label_for_vector
else:
    categorisation_method = apply_4_category_label_for_vector
    set_big_gain_boundary(big_gain_boundary)
    set_big_loss_boundary(big_loss_boundary)

#Load trained model
if binary_categories:
    path = f'models/{name}/xgboost/pickle/binary'
else:
    path = f'models/{name}/xgboost/pickle/4_options'

with open(f"{path}/classifier.pk", 'rb') as rf_pickle_file:
    classifier = pickle.load(rf_pickle_file)

#Load training dataset
training_dataset = pd.read_csv(f'models/{name}/data/test.csv')
X_test = training_dataset.iloc[:, 1:-2].values
y_test_outputs_dates = training_dataset.iloc[:, [0, -2, -1]].values
y_test_outputs = y_test_outputs_dates[:, [1,2]]
y_test = np.apply_along_axis(categorisation_method, 1, y_test_outputs)

gains = np.apply_along_axis(calculate_gains, 1, y_test_outputs)

y_pred = classifier.predict(X_test)
if binary_categories:
    decision = np.where(np.transpose([y_pred]), 'Buy', 'Sell')
else:
    decision = np.transpose([np.apply_along_axis(buy_or_sell, 1, np.transpose([y_pred]))])

results_array = np.hstack((y_test_outputs_dates, np.transpose([gains]), decision))
results_array = np.vstack((['Datetime', 'Open', 'Close', 'Difference', 'Decision'], results_array))

write_data_file('base_results', f'models/{name}/xgboost/analysis', results_array)