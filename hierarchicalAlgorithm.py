import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import time
from matplotlib.colors import LinearSegmentedColormap


def get_user_input():
    """Функция для ввода параметров пользователем."""
    print("Настройте параметры кластеризации:")
    n_clusters = int(input("Число кластеров для агломеративной кластеризации (по умолчанию 3): ") or 3)
    return n_clusters


def load_and_clean_data(filename):
    """Загрузка и очистка данных."""
    df = pd.read_csv(filename)

    # Удаляем строки с NaN в координатах или магнитуде
    df_clean = df.dropna(subset=['longitude', 'latitude', 'mag'])

    # Удаляем возможные бесконечные значения
    df_clean = df_clean[np.isfinite(df_clean['longitude']) &
                        np.isfinite(df_clean['latitude']) &
                        np.isfinite(df_clean['mag'])]

    return df_clean


def create_custom_colormap():
    """Создает градиент от зеленого к красному."""
    colors = ["green", "yellow", "red"]
    return LinearSegmentedColormap.from_list("mag_colormap", colors)


def main():
    start_time = time.time()

    # Получаем параметры от пользователя
    n_clusters = get_user_input()

    # 1. Загрузка и очистка данных
    data_load_start = time.time()
    df = load_and_clean_data('query.csv')

    if len(df) == 0:
        raise ValueError("После очистки не осталось данных для анализа!")

    X = df[['longitude', 'latitude']].values
    magnitudes = df['mag'].values

    load_time = time.time() - data_load_start
    print(f"\nЗагрузка данных: {load_time:.4f} сек")
    print(f"Очищенных землетрясений: {len(X)}")

    # 2. Построение дендрограммы (однопоточное)
    plt.figure(figsize=(15, 6))

    dendro_start = time.time()
    try:
        linked = linkage(X, method='ward', optimal_ordering=True)
    except ValueError as e:
        print(f"Ошибка при построении дендрограммы: {e}")
        print("Попробуйте уменьшить количество данных или изменить параметры.")
        return

    dendro_time = time.time() - dendro_start
    print(f"Построение дендрограммы (однопоточное): {dendro_time:.4f} сек")

    plt.subplot(1, 2, 1)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title("Дендрограмма")
    plt.xlabel("Индекс точки")
    plt.ylabel("Расстояние")

    # 3. Кластеризация и визуализация (однопоточная)
    cluster_start = time.time()
    cluster = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )
    labels = cluster.fit_predict(X)
    cluster_time = time.time() - cluster_start
    print(f"Кластеризация (однопоточная): {cluster_time:.4f} сек")

    plt.subplot(1, 2, 2)

    # Создаем кастомную цветовую карту
    cmap = create_custom_colormap()

    # Нормализуем магнитуды от 2 до 9
    norm_magnitudes = np.clip((magnitudes - 2) / (9 - 2), 0, 1)

    # Размер точек зависит от магнитуды
    sizes = np.clip((magnitudes - 2 + 1) * 20, 20, 200)

    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c=norm_magnitudes,
        cmap=cmap,
        s=sizes,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )

    # Добавляем подписи только к точкам с магнитудой >4
    for i, (x, y, mag) in enumerate(zip(X[:, 0], X[:, 1], magnitudes)):
        if mag > 4:
            plt.text(x, y, str(i), fontsize=8, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

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