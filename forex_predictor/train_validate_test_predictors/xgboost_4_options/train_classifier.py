#This classifier is used to make quick estimates to give an idea what sort of config we should apply to the CNN model
import pandas as pd
import numpy as np
import pickle
from utils.file_utils import create_sub_directories
from forex_predictor.data_extraction.process_raw_data import apply_4_category_label_for_vector, set_big_gain_boundary, set_big_loss_boundary

name = 'test4'
big_gain_boundary = 0.0003
big_loss_boundary = -0.0003
categorisation_method = apply_4_category_label_for_vector


# Importing the dataset
set_big_gain_boundary(big_gain_boundary)
set_big_loss_boundary(big_loss_boundary)
train_dataset = pd.read_csv(f'models/{name}/data/training.csv', header=None)
X_train = train_dataset.iloc[:, 1:-2].values
y_train = train_dataset.iloc[:, -2:].values
y_train = np.apply_along_axis(categorisation_method, 1, y_train)
validation_dataset = pd.read_csv(f'models/{name}/data/validation.csv')
X_val = validation_dataset.iloc[:, 1:-2].values
y_val = validation_dataset.iloc[:, -2:].values
y_val = np.apply_along_axis(categorisation_method, 1, y_val)

# Applying KPCA 
# from sklearn.decomposition import KernelPCA
# pca = KernelPCA(n_components=30)
# X_train = pca.fit_transform(X_train)
# X_val = pca.transform(X_val)

# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
print(cm)

big_sell_accuracy = (cm[3][3] + cm[2][3]) / cm.sum(axis=0)[3]
print(f'\nBig sell accuracy: {big_sell_accuracy}')

# # Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#Save model
path = f'models/{name}/xgboost_4_options/pickle'
create_sub_directories(path)
with open(f'{path}/classifier.pk', 'wb') as f:
    pickle.dump(classifier, f)