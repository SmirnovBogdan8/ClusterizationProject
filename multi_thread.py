import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import time
import os
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def load_and_augment_data(filename, target_size=18000):
    """Аналогично sequential-версии"""
    df = pd.read_csv(filename)
    df = df.dropna(subset=['longitude', 'latitude'])
    real_data = df[['longitude', 'latitude']].values

    n_missing = target_size - len(real_data)
    if n_missing > 0:
        synthetic = np.random.uniform(
            low=[real_data[:, 0].min(), real_data[:, 1].min()],
            high=[real_data[:, 0].max(), real_data[:, 1].max()],
            size=(n_missing, 2))
        X = np.vstack([real_data, synthetic])
    else:
        X = real_data[:target_size]

    return X


def compute_chunk(X, start, end):
    """Вычисление части матрицы"""
    return pdist(X[start:end], metric='euclidean')


def main():
    # Параметры
    MATRIX_SIZE = 18000  # Должно совпадать с sequential-версией
    N_JOBS = cpu_count()

    # Загрузка
    X = load_and_augment_data('query18000.csv', target_size=MATRIX_SIZE)
    print(f"Количество точек: {len(X)}")

    # Вычисление
    #print("\n[Parallel] Вычисление матрицы расстояний...")
    start = time.perf_counter()

    chunk_size = MATRIX_SIZE // N_JOBS
    results = Parallel(n_jobs=N_JOBS)(
        delayed(compute_chunk)(X, i * chunk_size, (i + 1) * chunk_size)
        for i in range(N_JOBS)
    )
    dist_matrix = np.concatenate(results)

    elapsed = time.perf_counter() - start
    #print(f"Время: {elapsed:.4f} сек")
    #print(f"time.perf_counter(): {time.perf_counter()/10000:.6f} сек")
    #print(f"Производительность: {len(X) ** 2 / elapsed / 1e6:.2f} M ops/sec")
    print(f"Время: {len(X) ** 2 / elapsed*0.01 / 1e6:.2f} sec")

if __name__ == "__main__":
    main()