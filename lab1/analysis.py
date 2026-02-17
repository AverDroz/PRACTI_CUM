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


def generate_dataset(size: int, seed: int = None) -> tuple:
    # train: item_0 ... item_{size-1}
    # test: item_{size} ... item_{2*size-1}  — ГАРАНТИРОВАННО не пересекаются
    salt = np.random.randint(0, 10**6) if seed is None else seed
    train = [f"train_{salt}_{i}" for i in range(size)]
    test  = [f"test_{salt}_{i}"  for i in range(size)]  # другой префикс = нет пересечений
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


def plot_all_dependencies():
    m_vals = [2000, 5000, 10000, 30000, 100000]
    k_vals = [1, 2, 3, 5, 7, 10, 15]
    n = 1000
    RUNS = 50

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("FPR Dependencies Analysis", fontsize=16)

    # ── 1. FPR vs m (k=7 fixed) ──────────────────────────────────────────
    ax = axes[0, 0]
    fprs, theory_fprs = [], []
    for m in m_vals:
        run_fprs = []
        for _ in range(RUNS):
            bf = BloomFilter(BloomConfig(m=m, k=7))
            train, test = generate_dataset(n)
            for item in train:
                bf.add(item)
            run_fprs.append(sum(1 for t in test if t in bf) / len(test))
        fprs.append(np.mean(run_fprs))
        theory_fprs.append((1 - np.exp(-7 * n / m)) ** 7)  # теория для графика 1

    ax.plot(m_vals, fprs,        'o-',  color='steelblue', label='Реальный')
    ax.plot(m_vals, theory_fprs, 's--', color='gray',      label='Теория', alpha=0.7)
    ax.set_xlabel("m (размер битового массива)")
    ax.set_ylabel("FPR")
    ax.set_title("FPR vs m (k=7 фиксировано)")
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 2. FPR vs k (m=10000 fixed) ──────────────────────────────────────
    ax = axes[0, 1]
    fprs, theory_fprs = [], []
    for k in k_vals:
        run_fprs = []
        for _ in range(RUNS):
            bf = BloomFilter(BloomConfig(m=10000, k=k))
            train, test = generate_dataset(n)
            for item in train:
                bf.add(item)
            run_fprs.append(sum(1 for t in test if t in bf) / len(test))
        fprs.append(np.mean(run_fprs))
        theory_fprs.append((1 - np.exp(-k * n / 10000)) ** k)  # теория для графика 2

    k_opt = (10000 / n) * np.log(2)
    ax.plot(k_vals, fprs,        'o-',  color='tomato', label='Реальный')
    ax.plot(k_vals, theory_fprs, 's--', color='gray',   label='Теория', alpha=0.7)
    ax.axvline(k_opt, color='black', linestyle=':', alpha=0.5, label=f'k_opt≈{k_opt:.1f}')
    ax.set_xlabel("k (количество хеш-функций)")
    ax.set_ylabel("FPR")
    ax.set_title("FPR vs k (m=10000 фиксировано)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. Теория vs Практика ────────────────────────────────────────────
    ax = axes[0, 2]
    theoretical, actual = [], []
    for m in m_vals:
        theoretical.append(min((1 - np.exp(-7 * n / m)) ** 7, 1.0))
        run_fprs = []
        for _ in range(RUNS):
            bf = BloomFilter(BloomConfig(m=m, k=7))
            train, test = generate_dataset(n)
            for item in train:
                bf.add(item)
            run_fprs.append(sum(1 for t in test if t in bf) / len(test))
        actual.append(np.mean(run_fprs))

    ax.plot(m_vals, theoretical, 'o--', label='Теоретический', color='green')
    ax.plot(m_vals, actual,      's-',  label='Реальный',      color='orange')
    ax.set_xlabel("m")
    ax.set_ylabel("FPR")
    ax.set_title("Теория vs Практика (n=1000, k=7)")
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 4. HyperLogLog: точность vs precision ────────────────────────────
    ax = axes[1, 0]
    precisions = [8, 10, 12, 14, 16]
    errors, theory_errors = [], []

    n_for_p = {8: 500, 10: 2000, 12: 8000, 14: 30000, 16: 120000}

    for p in precisions:
        n_hll = n_for_p[p]
        hll = HyperLogLog(precision=p)
        for i in range(n_hll):
            hll.add(f"item_{i}")
        est = hll.count()
        errors.append(abs(est - n_hll) / n_hll * 100)
        theory_errors.append(104 / np.sqrt(1 << p))

    ax.plot(precisions, errors,        'o-',  color='purple', label='Реальная ошибка')
    ax.plot(precisions, theory_errors, 's--', color='gray',   label='Теор. 104/√m')
    ax.set_xlabel("precision (p)")
    ax.set_ylabel("Ошибка (%)")
    ax.set_title("HyperLogLog: точность vs precision")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 5. MinHash: оценка vs реальность ─────────────────────────────────
    ax = axes[1, 1]
    overlaps = [0, 10, 25, 50, 75, 90, 100]
    real_j, estimated_j = [], []

    for overlap in overlaps:
        set_a = set(f"item_{i}" for i in range(100))
        set_b = set(f"item_{i}" for i in range(100 - overlap, 200 - overlap))
        real = len(set_a & set_b) / len(set_a | set_b)

        estimates = []
        for _ in range(5):
            mh1, mh2 = MinHash(num_perm=256), MinHash(num_perm=256)
            mh1.update_batch(set_a)
            mh2.update_batch(set_b)
            estimates.append(mh1.jaccard(mh2))

        real_j.append(real)
        estimated_j.append(np.mean(estimates))

    ax.plot(overlaps, real_j,      'o--', label='Реальный Jaccard', color='blue')
    ax.plot(overlaps, estimated_j, 's-',  label='MinHash оценка',   color='red')
    ax.set_xlabel("Пересечение (%)")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("MinHash: оценка vs реальность")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 6. Count-Min Sketch: ошибка vs ширина ────────────────────────────
    ax = axes[1, 2]
    widths = [100, 500, 1000, 5000, 10000]
    errors = []
    true_freq = 10

    for w in widths:
        cms = CountMinSketch(width=w, depth=5)
        for i in range(100):
            for _ in range(true_freq):
                cms.add(f"item_{i}")
        estimates = [cms.estimate(f"item_{i}") for i in range(100)]
        errors.append(
            np.mean([abs(e - true_freq) / true_freq for e in estimates]) * 100
        )

    ax.plot(widths, errors, 'o-', color='darkgreen')
    ax.set_xlabel("width (ширина таблицы)")
    ax.set_ylabel("Средняя ошибка (%)")
    ax.set_title("Count-Min Sketch: ошибка vs ширина")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("all_dependencies.png", dpi=300)
    plt.show()
    print("Сохранено: all_dependencies.png")


if __name__ == "__main__":
    m_vals = [1000, 5000, 10000, 50000]
    k_vals = [3, 5, 7, 10]
    
    print("Running FPR analysis...")
    fpr_matrix = measure_fpr_bloom(m_vals, k_vals, n=1000)
    plot_heatmap(fpr_matrix, m_vals, k_vals)
    
    print("\nRunning ANOVA...")
    anova_analysis(m_vals, k_vals)
    plot_all_dependencies()