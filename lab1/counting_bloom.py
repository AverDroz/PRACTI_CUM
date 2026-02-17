"""Counting Bloom Filter - поддержка удаления элементов."""

import numpy as np
from dataclasses import dataclass
from bloom_filter import HashFunction, BloomConfig


class CountingBloomFilter:
    """Bloom Filter со счетчиками вместо битов."""
    
    def __init__(self, config: BloomConfig, counter_bits: int = 4):
        self.config = config
        self.counters = np.zeros(config.m, dtype=np.uint8)
        self.max_count = (1 << counter_bits) - 1
        self.hashes = [HashFunction(i * 97 + 13, config.m) for i in range(config.k)]
        self.n = 0
    
    def add(self, item: str) -> None:
        for h in self.hashes:
            idx = h(item)
            if self.counters[idx] < self.max_count:
                self.counters[idx] += 1
        self.n += 1
    
    def remove(self, item: str) -> bool:
        """Удаление элемента. Возвращает False если элемента не было."""
        if item not in self:
            return False
        for h in self.hashes:
            idx = h(item)
            if self.counters[idx] > 0:
                self.counters[idx] -= 1
        self.n -= 1
        return True
    
    def __contains__(self, item: str) -> bool:
        return all(self.counters[h(item)] > 0 for h in self.hashes)
    
    def __or__(self, other: 'CountingBloomFilter') -> 'CountingBloomFilter':
        if (self.config.m, self.config.k) != (other.config.m, other.config.k):
            raise ValueError("Incompatible filters")
        result = CountingBloomFilter(self.config)
        result.counters = np.minimum(self.counters + other.counters, self.max_count)
        result.n = self.n + other.n
        return result
    
    @property
    def fpr(self) -> float:
        if self.n == 0:
            return 0.0
        return (1 - np.exp(-self.config.k * self.n / self.config.m)) ** self.config.k