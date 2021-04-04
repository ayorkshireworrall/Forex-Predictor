import pandas as pd
import numpy as np
import tensorflow as tf
from utils.file_utils import create_sub_directories
from forex_predictor.data_extraction.process_raw_data import apply_binary_category_label_for_vector

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
ann.save(f'models/{name}/ann/model')