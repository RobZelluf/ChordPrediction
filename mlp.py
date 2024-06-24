import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt

from data import get_data

notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
chord_types = ['maj', 'min']

X_train, X_test, y_train, y_test = get_data()

""" Train a multi-layer perceptron """
mlp = MLPClassifier(solver="adam", learning_rate="adaptive", learning_rate_init=0.001, hidden_layer_sizes=(128, 64, 32, 16),
                    max_iter=1000, early_stopping=True, n_iter_no_change=20, verbose=True, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

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
