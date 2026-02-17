"""Тесты для всех вероятностных структур."""

import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from bloom_filter import BloomFilter, BloomConfig
from counting_bloom import CountingBloomFilter
from hyperloglog import HyperLogLog
from quotient_filter import QuotientFilter, QuotientConfig
from count_min_sketch import CountMinSketch
from minhash import MinHash


class TestBloomFilter:
    """Тесты Bloom Filter."""
    
    def test_basic_operations(self):
        bf = BloomFilter(BloomConfig(m=1000, k=3))
        bf.add("test")
        assert "test" in bf
        # не проверяем отсутствие - может быть FP
    
    def test_union(self):
        bf1 = BloomFilter(BloomConfig(m=1000, k=3))
        bf2 = BloomFilter(BloomConfig(m=1000, k=3))
        bf1.add("a")
        bf2.add("b")
        union = bf1 | bf2
        assert "a" in union
        assert "b" in union
    
    def test_intersection(self):
        bf1 = BloomFilter(BloomConfig(m=1000, k=3))
        bf2 = BloomFilter(BloomConfig(m=1000, k=3))
        bf1.add("common")
        bf2.add("common")
        bf1.add("unique1")
        bf2.add("unique2")
        intersection = bf1 & bf2
        assert "common" in intersection
    
    def test_fpr_bounds(self):
        """FPR должен быть близок к теоретическому значению."""
        config = BloomConfig(m=10000, k=5)
        bf = BloomFilter(config)
        
        # Добавляем элементы
        train = [f"train_{i}" for i in range(1000)]
        for item in train:
            bf.add(item)
        
        # Проверяем FP
        test = [f"test_{i}" for i in range(1000)]
        fp_count = sum(1 for item in test if item in bf)
        actual_fpr = fp_count / len(test)
        
        # Теоретический FPR с погрешностью
        expected = bf.fpr
        assert abs(actual_fpr - expected) < 0.05


class TestCountingBloomFilter:
    """Тесты Counting Bloom Filter."""
    
    def test_add_remove(self):
        cbf = CountingBloomFilter(BloomConfig(m=1000, k=3))
        cbf.add("test")
        assert "test" in cbf
        assert cbf.remove("test")
        assert "test" not in cbf
    
    def test_remove_nonexistent(self):
        cbf = CountingBloomFilter(BloomConfig(m=1000, k=3))
        assert not cbf.remove("not_here")
    
    def test_multiple_adds(self):
        cbf = CountingBloomFilter(BloomConfig(m=1000, k=3))
        cbf.add("test")
        cbf.add("test")
        assert "test" in cbf
        cbf.remove("test")
        assert "test" in cbf  # еще одна копия
        cbf.remove("test")
        assert "test" not in cbf
    
    def test_union(self):
        cbf1 = CountingBloomFilter(BloomConfig(m=1000, k=3))
        cbf2 = CountingBloomFilter(BloomConfig(m=1000, k=3))
        cbf1.add("a")
        cbf2.add("b")
        union = cbf1 | cbf2
        assert "a" in union
        assert "b" in union


class TestHyperLogLog:
    """Тесты HyperLogLog."""
    
    def test_cardinality_small(self):
        hll = HyperLogLog(precision=12)
        items = [f"item_{i}" for i in range(100)]
        for item in items:
            hll.add(item)
        
        estimate = hll.count()
        error = abs(estimate - 100) / 100
        assert error < 0.1  # 10% для малых множеств
    
    def test_cardinality_medium(self):
        hll = HyperLogLog(precision=14)
        items = [f"item_{i}" for i in range(10000)]
        for item in items:
            hll.add(item)
        
        estimate = hll.count()
        error = abs(estimate - 10000) / 10000
        assert error < 0.03  # 3% для средних
    
    def test_cardinality_large(self):
        hll = HyperLogLog(precision=16)
        items = [f"item_{i}" for i in range(100000)]
        for item in items:
            hll.add(item)
        
        estimate = hll.count()
        error = abs(estimate - 100000) / 100000
        assert error < 0.02  # 2% для больших
    
    def test_union(self):
        hll1 = HyperLogLog(precision=12)
        hll2 = HyperLogLog(precision=12)
        
        for i in range(1000):
            hll1.add(f"item_{i}")
        for i in range(500, 1500):
            hll2.add(f"item_{i}")
        
        union = hll1 | hll2
        estimate = union.count()
        # Ожидаем ~1500 уникальных (0-1499)
        error = abs(estimate - 1500) / 1500
        assert error < 0.1


class TestQuotientFilter:
    """Тесты Quotient Filter."""
    
    def test_basic_operations(self):
        qf = QuotientFilter(QuotientConfig(q=10, r=6))
        qf.add("test")
        assert "test" in qf
    
    def test_load_factor(self):
        qf = QuotientFilter(QuotientConfig(q=8, r=6))
        for i in range(100):
            qf.add(f"item_{i}")
        assert 0 < qf.load_factor < 1


class TestCountMinSketch:
    """Тесты Count-Min Sketch."""
    
    def test_frequency_estimation(self):
        cms = CountMinSketch(width=1000, depth=5)
        
        # Добавляем элементы с известными частотами
        for i in range(100):
            cms.add("frequent", count=1)
        for i in range(10):
            cms.add("rare", count=1)
        
        assert cms.estimate("frequent") >= 100
        assert cms.estimate("rare") >= 10
    
    def test_merge(self):
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=1000, depth=5)
        
        cms1.add("a", 50)
        cms2.add("a", 30)
        
        merged = cms1 + cms2
        assert merged.estimate("a") >= 80


class TestMinHash:
    """Тесты MinHash."""
    
    def test_identical_sets(self):
        mh1 = MinHash(num_perm=128)
        mh2 = MinHash(num_perm=128)
        
        items = ["a", "b", "c", "d", "e"]
        mh1.update_batch(set(items))
        mh2.update_batch(set(items))
        
        similarity = mh1.jaccard(mh2)
        assert similarity > 0.9  # должна быть ~1.0
    
    def test_disjoint_sets(self):
        mh1 = MinHash(num_perm=128)
        mh2 = MinHash(num_perm=128)
        
        mh1.update_batch({"a", "b", "c"})
        mh2.update_batch({"x", "y", "z"})
        
        similarity = mh1.jaccard(mh2)
        assert similarity < 0.2  # должна быть ~0.0
    
    def test_partial_overlap(self):
        mh1 = MinHash(num_perm=256)
        mh2 = MinHash(num_perm=256)
        
        set1 = set(f"item_{i}" for i in range(100))
        set2 = set(f"item_{i}" for i in range(50, 150))
        
        mh1.update_batch(set1)
        mh2.update_batch(set2)
        
        similarity = mh1.jaccard(mh2)
        # Jaccard = 50 / 150 = 0.33
        assert 0.2 < similarity < 0.5


class TestPerformance:
    """Тесты производительности."""
    
    def test_bloom_scalability(self):
        """Bloom Filter должен работать быстро даже с большими m."""
        import time
        
        config = BloomConfig(m=1000000, k=7)
        bf = BloomFilter(config)
        
        start = time.time()
        for i in range(10000):
            bf.add(f"item_{i}")
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # должно быть <1 сек
    
    def test_hll_memory_efficiency(self):
        """HLL должен использовать O(m) памяти."""
        hll = HyperLogLog(precision=14)  # 2^14 = 16K registers
        
        # Добавляем миллион элементов
        for i in range(1000000):
            hll.add(f"item_{i}")
        
        # Память должна быть ~16KB (16384 uint8)
        memory_bytes = hll.registers.nbytes
        assert memory_bytes < 20000  # <20KB


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])