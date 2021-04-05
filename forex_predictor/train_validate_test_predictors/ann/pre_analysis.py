import numpy as np
import pandas as pd
import tensorflow as tf
from utils.file_utils import write_data_file

#Match the name of data generated in other module
name = 'test4'

#Load trained model
ann = tf.keras.models.load_model(f'models/{name}/ann/model')

#Load training dataset
training_dataset = pd.read_csv(f'models/{name}/data/test.csv')
X_test = training_dataset.iloc[:, 1:-2].values
y_test_outputs_dates = training_dataset.iloc[:, [0, -2, -1]].values
y_test_outputs = y_test_outputs_dates[:, [1,2]]

#Make predictions
y_pred = ann.predict(X_test)
buy_pred = y_pred > 0.55
sell_pred = y_pred < 0.45

#Save predictions and actual price differences for future analysis
def calculate_change(x):
    open, close = x
    return close - open
actual_price_change = np.apply_along_axis(calculate_change, 1, y_test_outputs)
results_array = np.hstack((y_test_outputs_dates, np.transpose([actual_price_change]), buy_pred, sell_pred))
results_array = np.vstack((['Datetime', 'Open', 'Close', 'Difference', 'Buy', 'Sell'], results_array))
write_data_file('base_results', f'models/{name}/ann/analysis', results_array)