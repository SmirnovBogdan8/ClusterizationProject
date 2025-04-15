import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import time


def get_user_input():
    """Функция для ввода параметров пользователем."""
    print("Настройте параметры датасета и кластеризации:")
    n_samples = int(input("Количество точек (например, 150): ") or 150)
    n_centers = int(input("Количество кластеров (например, 3): ") or 3)
    cluster_std = float(input("Разброс кластеров (например, 0.8): ") or 0.8)
    random_state = int(input("Random state (например, 42, или 0 для случайного): ") or 42)
    n_clusters = int(input("Число кластеров для агломеративной кластеризации: ") or n_centers)

    return n_samples, n_centers, cluster_std, random_state, n_clusters


def main():
    start_time = time.time()  # Засекаем общее время выполнения

    # Получаем параметры от пользователя
    n_samples, n_centers, cluster_std, random_state, n_clusters = get_user_input()

    # 1. Генерация данных
    gen_start = time.time()
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state if random_state != 0 else None
    )
    gen_time = time.time() - gen_start
    print(f"\nГенерация данных: {gen_time:.4f} сек")

    # 2. Визуализация исходных данных
    plot1_start = time.time()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.title("Исходные данные")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")

    # 3. Построение дендрограммы
    dendro_start = time.time()
    linked = linkage(X, method='ward')
    dendro_time = time.time() - dendro_start
    print(f"Построение дендрограммы: {dendro_time:.4f} сек")

    plt.subplot(1, 2, 2)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title("Дендрограмма")
    plt.xlabel("Индекс точки")
    plt.ylabel("Расстояние")

    plt.tight_layout()
    plot1_time = time.time() - plot1_start
    print(f"Построение графиков (1): {plot1_time:.4f} сек")
    plt.show()

    # 4. Кластеризация с заданным числом кластеров
    cluster_start = time.time()
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = cluster.fit_predict(X)
    cluster_time = time.time() - cluster_start
    print(f"Кластеризация: {cluster_time:.4f} сек")

    # 5. Визуализация кластеров
    plot2_start = time.time()
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f"Агломеративная кластеризация (n_clusters={n_clusters})")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.colorbar()
    plot2_time = time.time() - plot2_start
    print(f"Построение графиков (2): {plot2_time:.4f} сек")
    plt.show()

    total_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {total_time:.4f} сек")


if __name__ == "__main__":
    main()