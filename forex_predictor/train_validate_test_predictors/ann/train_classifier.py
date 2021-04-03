import pandas as pd
import numpy as np
import tensorflow as tf
from utils.file_utils import create_sub_directories
from forex_predictor.data_extraction.process_raw_data import apply_4_category_label_for_vector_string, apply_binary_category_label_for_vector, apply_4_category_label_for_vector, apply_binary_category_label_for_vector_string, set_big_gain_boundary, set_big_loss_boundary

def calculate_gains(x):
    open, close = x
    return open - close

name = 'test4'

categorisation_method = apply_binary_category_label_for_vector

# Importing the dataset
train_dataset = pd.read_csv(f'models/{name}/data/training.csv', header=None)
X_train = train_dataset.iloc[:, 1:-2].values
y_train_outputs = train_dataset.iloc[:, -2:].values
y_train = np.apply_along_axis(categorisation_method, 1, y_train_outputs)

validation_dataset = pd.read_csv(f'models/{name}/data/validation.csv')
X_val = validation_dataset.iloc[:, 1:-2].values
y_val_outputs = validation_dataset.iloc[:, -2:].values
y_val = np.apply_along_axis(categorisation_method, 1, y_val_outputs)

#Creating the ANN structure
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=36, activation='relu'))
ann.add(tf.keras.layers.Dense(units=36, activation='relu'))
ann.add(tf.keras.layers.Dense(units=36, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 100, epochs = 250)
y_pred = ann.predict(X_val)

actual_gains = np.apply_along_axis(calculate_gains, 1, y_val_outputs)

#set the threshold for the machine's confidence
buy_pred = y_pred > 0.6
sell_pred = y_pred < 0.4

buy_results = np.where(buy_pred[:,0], actual_gains, 0)
sell_results = np.where(sell_pred[:,0], actual_gains * -1, 0)

total_buys = len(np.where(buy_pred)[0])
total_sales = len(np.where(sell_pred)[0])

training_buys = len(np.where(y_train == 1)[0])
training_sells = len(np.where(y_train == 0)[0])

print(f'\nNumber training buys: {training_buys}')
print(f'Number training sells: {training_sells}')

print(f'\nNumber buys: {total_buys}')
print(f'Number sells: {total_sales}')

print(f'\navg buy: {np.sum(buy_results) / total_buys}')
print(f'avg sell: {np.sum(sell_results) / total_sales}')

#confusion matrix and score not used in the traditional sense (if  buy/sell thresholds != 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_val, buy_pred)
score = accuracy_score(y_val, buy_pred)
print(f'\nBuy confusion matrix:\n{cm}\n{score}')
cm = confusion_matrix(y_val, sell_pred)
score = accuracy_score(y_val, sell_pred)
print(f'\Sell confusion matrix:\n{cm}\n{score}')

ann.save(f'models/{name}/ann/model')