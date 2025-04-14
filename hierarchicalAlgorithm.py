import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# 1. Генерация данных
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)

# 2. Визуализация исходных данных
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Исходные данные")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")

# 3. Построение дендрограммы
linked = linkage(X, method='ward')  # Метод Варда (минимизация дисперсии)
plt.subplot(1, 2, 2)
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Дендрограмма")
plt.xlabel("Индекс точки")
plt.ylabel("Расстояние")

plt.tight_layout()
plt.show()

# 4. Кластеризация с заданным числом кластеров
n_clusters = 3
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
labels = cluster.fit_predict(X)

# 5. Визуализация кластеров
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title(f"Агломеративная кластеризация (n_clusters={n_clusters})")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.colorbar()
plt.show()