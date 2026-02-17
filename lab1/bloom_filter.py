"""Bloom Filter - вероятностная структура для проверки принадлежности."""

from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class BloomConfig:
    m: int  # размер битового массива
    k: int  # количество хеш-функций
    
    @property
    def optimal_n(self) -> int:
        """Оптимальное количество элементов: n = (m/k) * ln(2)."""
        return int(self.m * np.log(2) / self.k)


class HashFunction:
    """Кастомная хеш-функция (polynomial rolling hash)."""
    
    def __init__(self, seed: int, m: int):
        self.seed = seed
        self.m = m
        self.prime = 2654435761  # Knuth multiplicative hash
    
    def __call__(self, item: str) -> int:
        h = self.seed
        for c in item:
            h = (h * self.prime + ord(c)) & 0xFFFFFFFF
        return h % self.m


class BloomFilter:
    """Bloom Filter с O(k) add/contains."""
    
    def __init__(self, config: BloomConfig):
        self.config = config
        self.bits = np.zeros(config.m, dtype=bool)
        self.hashes = [HashFunction(i * 97 + 13, config.m) for i in range(config.k)]
        self.n = 0
    
    def add(self, item: str) -> None:
        for h in self.hashes:
            self.bits[h(item)] = True
        self.n += 1
    
    def __contains__(self, item: str) -> bool:
        return all(self.bits[h(item)] for h in self.hashes)
    
    def __or__(self, other: 'BloomFilter') -> 'BloomFilter':
        """Объединение фильтров."""
        if (self.config.m, self.config.k) != (other.config.m, other.config.k):
            raise ValueError("Incompatible filters")
        result = BloomFilter(self.config)
        result.bits = self.bits | other.bits
        result.n = self.n + other.n
        return result
    
    def __and__(self, other: 'BloomFilter') -> 'BloomFilter':
        """Пересечение фильтров."""
        if (self.config.m, self.config.k) != (other.config.m, other.config.k):
            raise ValueError("Incompatible filters")
        result = BloomFilter(self.config)
        result.bits = self.bits & other.bits
        return result
    
    @property
    def fpr(self) -> float:
        """False Positive Rate: (1 - e^(-kn/m))^k."""
        if self.n == 0:
            return 0.0
        return (1 - np.exp(-self.config.k * self.n / self.config.m)) ** self.config.k