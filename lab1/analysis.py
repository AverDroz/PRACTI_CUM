"""Эксперименты и анализ FPR для всех алгоритмов."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from bloom_filter import BloomFilter, BloomConfig
from counting_bloom import CountingBloomFilter
from hyperloglog import HyperLogLog
from quotient_filter import QuotientFilter, QuotientConfig
from count_min_sketch import CountMinSketch
from minhash import MinHash
from scipy import stats


def generate_dataset(size: int) -> Tuple[List[str], List[str]]:
    """Генерация train/test множеств."""
    train = [f"item_{i}" for i in range(size)]
    test = [f"item_{i}" for i in range(size, size * 2)]
    return train, test


def measure_fpr_bloom(m_values: List[int], k_values: List[int], n: int = 1000) -> np.ndarray:
    """Измерение FPR для разных m и k."""
    results = np.zeros((len(m_values), len(k_values)))
    
    for i, m in enumerate(m_values):
        for j, k in enumerate(k_values):
            train, test = generate_dataset(n)
            bf = BloomFilter(BloomConfig(m=m, k=k))
            
            for item in train:
                bf.add(item)
            
            # Считаем false positives
            fp = sum(1 for item in test if item in bf)
            results[i, j] = fp / len(test)
    
    return results


def anova_analysis(m_values: List[int], k_values: List[int], n: int = 1000, trials: int = 30):
    """Многофакторный дисперсионный анализ (ANOVA)."""
    data = []
    
    for m in m_values:
        for k in k_values:
            for _ in range(trials):
                train, test = generate_dataset(n)
                bf = BloomFilter(BloomConfig(m=m, k=k))
                for item in train:
                    bf.add(item)
                fpr = sum(1 for item in test if item in bf) / len(test)
                data.append({'m': m, 'k': k, 'fpr': fpr})
    
    # Two-way ANOVA
    groups_m = {m: [d['fpr'] for d in data if d['m'] == m] for m in m_values}
    groups_k = {k: [d['fpr'] for d in data if d['k'] == k] for k in k_values}
    
    f_m, p_m = stats.f_oneway(*groups_m.values())
    f_k, p_k = stats.f_oneway(*groups_k.values())
    
    print(f"ANOVA Results:")
    print(f"Factor m: F={f_m:.4f}, p={p_m:.6f} {'***' if p_m < 0.001 else ''}")
    print(f"Factor k: F={f_k:.4f}, p={p_k:.6f} {'***' if p_k < 0.001 else ''}")


def plot_heatmap(results: np.ndarray, m_values: List[int], k_values: List[int]):
    """Визуализация FPR heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(results, cmap='viridis', aspect='auto')
    plt.colorbar(label='False Positive Rate')
    plt.xlabel('k (hash functions)')
    plt.ylabel('m (bit array size)')
    plt.xticks(range(len(k_values)), k_values)
    plt.yticks(range(len(m_values)), m_values)
    plt.title('Bloom Filter FPR Analysis')
    plt.tight_layout()
    plt.savefig('bloom_fpr_heatmap.png', dpi=300)


if __name__ == "__main__":
    m_vals = [1000, 5000, 10000, 50000]
    k_vals = [3, 5, 7, 10]
    
    print("Running FPR analysis...")
    fpr_matrix = measure_fpr_bloom(m_vals, k_vals, n=1000)
    plot_heatmap(fpr_matrix, m_vals, k_vals)
    
    print("\nRunning ANOVA...")
    anova_analysis(m_vals, k_vals)