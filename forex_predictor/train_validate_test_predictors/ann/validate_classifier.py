import numpy as np
import pandas as pd
from forex_predictor.data_extraction.process_raw_data import apply_4_category_label_for_vector, apply_binary_category_label_for_vector, set_big_gain_boundary, set_big_loss_boundary
import tensorflow as tf

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

#Load validation dataset
validation_dataset = pd.read_csv(f'models/{name}/data/validation.csv')
X_val = validation_dataset.iloc[:, 1:-2].values
y_val_outputs = validation_dataset.iloc[:, -2:].values
y_val = np.apply_along_axis(categorisation_method, 1, y_val_outputs)

#make prediction as classification
y_pred = ann.predict(X_val)
y_pred = y_pred > 0.5

#validate results
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_val, y_pred)
score = accuracy_score(y_val, y_pred)
print(f'\nConfusion matrix:\n{cm}\n\nAccuracy Score:{score}')


