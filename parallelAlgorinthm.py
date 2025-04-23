import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import time
from joblib import parallel_backend


def get_user_input():
    """Функция для ввода параметров пользователем."""
    print("Настройте параметры кластеризации:")
    min_magnitude = float(input("Минимальная магнитуда для фильтрации (например, 3.0): ") or 3.0)
    n_clusters = int(input("Число кластеров для агломеративной кластеризации: ") or 3)
    return min_magnitude, n_clusters


def load_and_filter_data(filename, min_magnitude):
    """Загрузка данных из CSV и фильтрация по магнитуде."""
    df = pd.read_csv(filename)
    filtered_df = df[df['mag'] >= min_magnitude].reset_index(drop=True)
    return filtered_df


def main():
    start_time = time.time()

    # Получаем параметры от пользователя
    min_magnitude, n_clusters = get_user_input()

    # Фиксируем число потоков
    n_jobs = 10

    # 1. Загрузка и фильтрация данных
    data_load_start = time.time()
    df = load_and_filter_data('2.5_day.csv', min_magnitude)
    X = df[['longitude', 'latitude']].values
    magnitudes = df['mag'].values
    load_time = time.time() - data_load_start
    print(f"\nЗагрузка и фильтрация данных: {load_time:.4f} сек")
    print(f"Найдено {len(X)} землетрясений с магнитудой ≥ {min_magnitude}")

    # 2. Построение дендрограммы
    plt.figure(figsize=(15, 6))

    dendro_start = time.time()
    with parallel_backend('threading', n_jobs=n_jobs):
        linked = linkage(X, method='ward', optimal_ordering=True)
    dendro_time = time.time() - dendro_start
    print(f"Построение дендрограммы (10 потоков): {dendro_time:.4f} сек")

    plt.subplot(1, 2, 1)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title("Дендрограмма")
    plt.xlabel("Индекс точки")
    plt.ylabel("Расстояние")

    # 3. Кластеризация и визуализация
    cluster_start = time.time()
    with parallel_backend('threading', n_jobs=n_jobs):
        cluster = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
        labels = cluster.fit_predict(X)
    cluster_time = time.time() - cluster_start
    print(f"Кластеризация (10 потоков): {cluster_time:.4f} сек")

    plt.subplot(1, 2, 2)
    # Размер точек зависит от магнитуды
    sizes = (magnitudes - min_magnitude + 1) * 20
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20', s=sizes, alpha=0.7)

    # Добавляем подписи индексов ко всем точкам
    for i, (x, y) in enumerate(zip(X[:, 0], X[:, 1])):
        plt.text(x, y, str(i), fontsize=8, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.colorbar(scatter, label='Кластер')
    plt.title(f"Кластеризация землетрясений (n_clusters={n_clusters})")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")

    plt.tight_layout()
    plt.show()

    total_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {total_time:.4f} сек")


if __name__ == "__main__":
    main()