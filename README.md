# Import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load dataset Iris
iris = load_iris()
data = iris.data

# Proses K-Means
kmeans = KMeans(n_clusters=3,
random_state=42)
kmeans.fit(data)

# Ambil hasil cluster
labels = kmeans.labels_

# Visualisasi hasil
plt.scatter(data[:, 0], data[:,1], c=labels)
plt.xlabel("Panjang Sepal")
plt.ylabel("Lebar Sepal")
plt.title("Pengelompokan Data Bunga Iris Menggunakan Metode K-Means")
plt.show()
