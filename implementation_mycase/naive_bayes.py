# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import preprocessing as pp

# Train the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(pp.X_train, pp.y_train)

# Predict the Test set results
y_pred = classifier.predict(pp.X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pp.y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualising the Training set results 3D
from matplotlib.colors import ListedColormap
X_set, y_set = pp.X_train, pp.y_train
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
for i, j in enumerate(np.unique(y_set)):
    ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], X_set[y_set == j, 2],
               c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
ax.set_title('Naive Bayes (Training set)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.show()

# Visualising the Test set results
X_set, y_set = pp.X_test, pp.y_test
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
for i, j in enumerate(np.unique(y_set)):
    ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], X_set[y_set == j, 2],
               c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
ax.set_title('Naive Bayes (Test set)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.show()