from functools import total_ordering
import numpy as np
import pandas as pd
import tensorflow as tf
from forex_predictor.data_extraction.process_raw_data import apply_4_category_label_for_vector, apply_binary_category_label_for_vector, set_big_gain_boundary, set_big_loss_boundary
from utils.file_utils import write_data_file

def calculate_gains(x):
    open, close = x
    return open - close

#Changeable variables
name = 'test4'
binary_categories = True
big_gain_boundary = 0.0002
big_loss_boundary = -0.0002

#Categorise based on buy/sell or big buy/ big sell
if binary_categories:
    categorisation_method = apply_binary_category_label_for_vector
else:
    categorisation_method = apply_4_category_label_for_vector
    set_big_gain_boundary(big_gain_boundary)
    set_big_loss_boundary(big_loss_boundary)

#Load trained model
ann = tf.keras.models.load_model(f'models/{name}/ann/model')

#Load training dataset
training_dataset = pd.read_csv(f'models/{name}/data/validation.csv')
X_test = training_dataset.iloc[:, 1:-2].values
y_test_outputs_dates = training_dataset.iloc[:, [0, -2, -1]].values
y_test_outputs = y_test_outputs_dates[:, [1,2]]
y_test = np.apply_along_axis(categorisation_method, 1, y_test_outputs)

gains = np.apply_along_axis(calculate_gains, 1, y_test_outputs)

y_pred = ann.predict(X_test)

buy_pred = y_pred > 0.5
sell_pred = y_pred < 0.5

buy_results = np.where(buy_pred[:,0], gains, 0)
sell_results = np.where(sell_pred[:,0], gains * -1, 0)

total_buys = len(np.where(buy_pred)[0])
total_sales = len(np.where(sell_pred)[0])

buys_sum = np.sum(buy_results)
sell_sum = np.sum(sell_results)

results_array = np.hstack((y_test_outputs_dates, np.transpose([gains]), np.where(buy_pred, 'Buy', 'Sell')))
results_array = np.vstack((['Datetime', 'Open', 'Close', 'Difference', 'Decision'], results_array))

print(f'\nTotal buys: {total_buys}')
print(f'Avg buy profit: {buys_sum/total_buys}')
print(f'\nTotal sells: {total_sales}')
print(f'Avg sell profit: {sell_sum/total_sales}')

write_data_file('base_results', f'models/{name}/ann/analysis', results_array)