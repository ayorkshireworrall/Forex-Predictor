import pandas as pd
import numpy as np
import tensorflow as tf
from utils.file_utils import create_sub_directories
from forex_predictor.data_extraction.process_raw_data import apply_4_category_label_for_vector_string, apply_binary_category_label_for_vector, apply_4_category_label_for_vector, apply_binary_category_label_for_vector_string, set_big_gain_boundary, set_big_loss_boundary

def calculate_gains(x):
    open, close = x
    return open - close

name = 'test4'
big_gain_boundary = 0.0002
big_loss_boundary = -0.0002
categorisation_method = apply_4_category_label_for_vector

# Importing the dataset
set_big_gain_boundary(big_gain_boundary)
set_big_loss_boundary(big_loss_boundary)
train_dataset = pd.read_csv(f'models/{name}/data/training.csv', header=None)
X_train = train_dataset.iloc[:, 1:-2].values
y_train_outputs = train_dataset.iloc[:, -2:].values
y_train = np.apply_along_axis(categorisation_method, 1, y_train_outputs)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y_train = ct.fit_transform(y_train.reshape(-1,1)).toarray()

validation_dataset = pd.read_csv(f'models/{name}/data/validation.csv')
X_val = validation_dataset.iloc[:, 1:-2].values
y_val_outputs = validation_dataset.iloc[:, -2:].values
y_val = np.apply_along_axis(categorisation_method, 1, y_val_outputs)

#Creating the ANN structure
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=36, activation='relu'))
ann.add(tf.keras.layers.Dense(units=36, activation='relu'))
ann.add(tf.keras.layers.Dense(units=36, activation='relu'))
ann.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# Part 3 - Training the ANN
ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 100, epochs = 250)
y_pred = ann.predict(X_val)

actual_gains = np.apply_along_axis(calculate_gains, 1, y_val_outputs)

buy_pred_args = y_pred.argmax(axis=1) == 3
buy_pred_vals = y_pred.max(axis=1) > 0.3
combined = np.vstack((buy_pred_args, buy_pred_vals))
combined = np.transpose(combined)
buy_pred = combined.all(axis=1)

sell_pred_args = y_pred.argmax(axis=1) == 0
sell_pred_vals = y_pred.max(axis=1) > 0.3
combined = np.vstack((sell_pred_args, sell_pred_vals))
combined = np.transpose(combined)
sell_pred = combined.all(axis=1)

buy_results = np.where(buy_pred, actual_gains, 0)
sell_results = np.where(sell_pred, actual_gains * -1, 0)

total_buys = len(np.where(buy_pred)[0])
total_sales = len(np.where(sell_pred)[0])

training_buys = len(np.where(y_train.argmax(axis=1) == 3)[0])
training_sells = len(np.where(y_train.argmax(axis=1) == 0)[0])

print(f'\nNumber training buys: {training_buys}')
print(f'Number training sells: {training_sells}')

print(f'\nNumber buys: {total_buys}')
print(f'Number sells: {total_sales}')

print(f'\navg buy: {np.sum(buy_results) / total_buys}')
print(f'avg sell: {np.sum(sell_results) / total_sales}')

ann.save(f'models/{name}/ann_softmax/model')