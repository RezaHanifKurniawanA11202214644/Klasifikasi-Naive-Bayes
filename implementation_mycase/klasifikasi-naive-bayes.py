# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import preprocessing as pp

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(pp.X_train, pp.y_train)

# Predicting the Test set results
y_pred = classifier.predict(pp.X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pp.y_test, y_pred)
print("cm:\n", cm, "\n")

import seaborn as sns
# Membuat visualisasi confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Naive Bayes Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()