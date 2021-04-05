import numpy as np
import pandas as pd
import tensorflow as tf
from utils.file_utils import write_data_file

#Match the name of data generated in other module
name = 'test4'

#Load trained model
ann = tf.keras.models.load_model(f'models/{name}/ann_softmax/model')

#Load test dataset
test_dataset = pd.read_csv(f'models/{name}/data/test.csv')
X_test = test_dataset.iloc[:, 1:-2].values
y_test_outputs_dates = test_dataset.iloc[:, [0, -2, -1]].values
y_test_outputs = y_test_outputs_dates[:, [1,2]]

#Make predictions
y_pred = ann.predict(X_test)
##Get column for buy predictions
buy_pred_args = y_pred.argmax(axis=1) == 3
buy_pred_vals = y_pred.max(axis=1) > 0.3 #configure based on desired confidence from the ann
combined = np.vstack((buy_pred_args, buy_pred_vals))
combined = np.transpose(combined)
buy_pred = combined.all(axis=1)
buy_pred = np.where(buy_pred, 1, 0)
##Get column for sell predictions
sell_pred_args = y_pred.argmax(axis=1) == 0
sell_pred_vals = y_pred.max(axis=1) > 0.3 #configure based on desired confidence from the ann
combined = np.vstack((sell_pred_args, sell_pred_vals))
combined = np.transpose(combined)
sell_pred = combined.all(axis=1)
sell_pred = np.where(sell_pred, 1, 0)

#Save predictions and actual price differences for future analysis
def calculate_change(x):
    open, close = x
    return close - open
actual_price_change = np.apply_along_axis(calculate_change, 1, y_test_outputs)
results_array = np.hstack((y_test_outputs_dates, np.transpose([actual_price_change]), np.transpose([buy_pred]), np.transpose([sell_pred])))
results_array = np.vstack((['Datetime', 'Open', 'Close', 'Difference', 'Buy', 'Sell'], results_array))
write_data_file('base_results', f'models/{name}/ann_softmax/analysis', results_array)