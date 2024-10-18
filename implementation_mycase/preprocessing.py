# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset dan membagi data ke variabel X sebagai attribute reguler dan y sebagai attribute label
dataset = pd.read_csv('bakery_customer.csv')
X = dataset.iloc[:, [2, 4, 5, 8]].values
y = dataset.iloc[:, -1].values
print("X:\n", X, "\n")
print("y:\n", y, "\n")

# Mengecek apakah ada missing value
miss = dataset.isnull().sum()
print(miss)

# Encoding data kategori ke numerik pada variabel X (attribute reguler) menggunakan OneHotEncoder dan ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2, 3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Menampilkan data yang sudah di encoding
print("\nData variabel X setelah di encoding:\n", X)

# Encoding data kategori ke numerik pada variabel y (attribute label) menggunakan OneHotEncoder dan ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Menampilkan data yang sudah di encoding
print("\nData variabel Y setelah di encoding:\n", y)

# Membagi data menjadi data training dan data testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Mencetak data training dan data testing
print("X_train:\n", X_train, "\n")
print("X_test:\n", X_test, "\n")
print("y_train:\n", y_train, "\n")
print("y_test:\n", y_test, "\n")

# Feature Scaling atau normalisasi data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Mencetak data training dan data testing setelah feature scaling
print("X_train:\n", X_train, "\n")
print("X_test:\n", X_test, "\n")
