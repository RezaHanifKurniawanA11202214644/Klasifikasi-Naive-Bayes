# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import preprocessing as pp

# Reduksi dimensi data X menjadi 2 dimensi menggunakan PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(pp.X_train)
X_test_pca = pca.transform(pp.X_test)

# Mencetak variance explained ratio untuk melihat seberapa besar varian yang dijelaskan oleh masing-masing komponen
explained_variance = pca.explained_variance_ratio_
print("\nExplained variance ratio (komponen 1 dan 2):", explained_variance)


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train_pca, pp.y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_pca)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pp.y_test, y_pred)
print("cm:\n", cm, "\n")

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train_pca, pp.y_train
# Membuat meshgrid untuk visualisasi area decision boundary
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# Membatasi sumbu x dan y sesuai dengan range data
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Plot titik-titik data pada plot
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
# Menambahkan judul dan label
plt.title('Naive Bayes (Training set - PCA)')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.legend()
plt.show()


# Visualising the Test set results
X_set, y_set = X_test_pca, pp.y_test
# Membuat meshgrid untuk visualisasi area decision boundary
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# Membatasi sumbu x dan y sesuai dengan range data
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Plot titik-titik data pada plot
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
# Menambahkan judul dan label
plt.title('Naive Bayes (Training set - PCA)')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.legend()
plt.show()