import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import time
from joblib import parallel_backend
from matplotlib.colors import LinearSegmentedColormap


def get_user_input():
    """Функция для ввода параметров пользователем."""
    print("Настройте параметры кластеризации:")
    n_clusters = int(input("Число кластеров для агломеративной кластеризации (по умолчанию 3): ") or 3)
    return n_clusters


def load_data(filename):
    """Загрузка данных из CSV."""
    df = pd.read_csv(filename)
    return df


def create_custom_colormap():
    """Создает градиент от зеленого к красному."""
    colors = ["green", "yellow", "red"]
    return LinearSegmentedColormap.from_list("mag_colormap", colors)


def main():
    start_time = time.time()

    # Получаем параметры от пользователя
    n_clusters = get_user_input()

    # Фиксируем число потоков
    n_jobs = 10

    # 1. Загрузка данных
    data_load_start = time.time()
    df = load_data('query.csv')

    # Автоматическая фильтрация (берем все данные)
    X = df[['longitude', 'latitude']].values
    magnitudes = df['mag'].values
    load_time = time.time() - data_load_start
    print(f"\nЗагрузка данных: {load_time:.4f} сек")
    print(f"Всего землетрясений: {len(X)}")

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

    # Создаем кастомную цветовую карту
    cmap = create_custom_colormap()

    # Нормализуем магнитуды от 2 до 9
    norm_magnitudes = np.clip((magnitudes - 2) / (9 - 2), 0, 1)

    # Размер точек зависит от магнитуды
    sizes = (magnitudes - 2 + 1) * 20

    # Рисуем точки с цветом по магнитуде и формой по кластеру
    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c=norm_magnitudes,
        cmap=cmap,
        s=sizes,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )

    # Добавляем подписи индексов ко всем точкам
    for i, (x, y) in enumerate(zip(X[:, 0], X[:, 1])):
        plt.text(x, y, str(i), fontsize=8, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    # Настраиваем цветовую шкалу
    cbar = plt.colorbar(scatter, label='Магнитуда')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['2.0', '5.5', '9.0+'])

    plt.title(f"Кластеризация землетрясений (n_clusters={n_clusters})")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")

    plt.tight_layout()
    plt.show()

    total_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {total_time:.4f} сек")


if __name__ == "__main__":
    main()