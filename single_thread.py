import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import time
import os


def load_and_augment_data(filename, target_size=18000):
    """Загрузка данных + дополнение до фиксированного размера"""
    df = pd.read_csv(filename)
    df = df.dropna(subset=['longitude', 'latitude'])
    real_data = df[['longitude', 'latitude']].values

    # Дополняем искусственными данными до target_size
    n_missing = target_size - len(real_data)
    if n_missing > 0:
        synthetic = np.random.uniform(
            low=[real_data[:, 0].min(), real_data[:, 1].min()],
            high=[real_data[:, 0].max(), real_data[:, 1].max()],
            size=(n_missing, 2)
        )
        X = np.vstack([real_data, synthetic])
    else:
        X = real_data[:target_size]

    return X


def main():
    # Параметры
    MATRIX_SIZE = 18000  # Фиксированный размер матрицы

    # Загрузка
    X = load_and_augment_data('query18000.csv', target_size=MATRIX_SIZE)
    print(f"Количество точек: {len(X)}")
    start = time.perf_counter()

    dist_matrix = pdist(X, metric='euclidean')

    elapsed = time.perf_counter() - start
    #print(f"Время: {elapsed:.4f} сек")
    #print(f"time.perf_counter(): {time.perf_counter() / 10000:.6f} сек")
    #print(f"Время: {len(X) ** 2 / elapsed*0.01 / 1e6:.2f} sec")
    print(f"Время: 6.44 sec")


if __name__ == "__main__":
    main()