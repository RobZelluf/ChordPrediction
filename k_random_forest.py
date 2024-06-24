from random import randint

import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt

from data import get_data

notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
chord_types = ['maj', 'min']


X_train, X_test, y_train, y_test = get_data()

""" Train Random Forest model """
param_dist = {'n_estimators': np.arange(10, 1000, 20),
              'max_depth': np.arange(1, 50, 10)}

rf = RandomForestClassifier()

rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, verbose=True, n_jobs=-1)
rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
y_pred = best_rf.predict(X_test)

""" Show evaluation metrics"""
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

""" Show confusion matrix"""
n_labels = int(len(chord_types) * len(notes))
cm = confusion_matrix(y_test, y_pred)
cmp = ConfusionMatrixDisplay(cm, display_labels=np.arange(n_labels))
fig, ax = plt.subplots(figsize=(10, 10))
cmp.plot(ax=ax)

plt.show()
