"""Count-Min Sketch - частотный анализ потоков."""

import numpy as np
from bloom_filter import HashFunction


class CountMinSketch:
    """Approximate frequency counting with bounded error."""
    
    def __init__(self, width: int, depth: int):
        self.w = width  # buckets per row
        self.d = depth  # number of rows
        self.table = np.zeros((depth, width), dtype=np.uint32)
        self.hashes = [HashFunction(i * 89 + 7, width) for i in range(depth)]
        self.total = 0
    
    def add(self, item: str, count: int = 1) -> None:
        """Увеличить счетчик элемента."""
        for i, h in enumerate(self.hashes):
            self.table[i, h(item)] += count
        self.total += count
    
    def estimate(self, item: str) -> int:
        """Оценка частоты (верхняя граница с высокой вероятностью)."""
        return min(self.table[i, h(item)] for i, h in enumerate(self.hashes))
    
    def __add__(self, other: 'CountMinSketch') -> 'CountMinSketch':
        """Объединение sketches."""
        if (self.w, self.d) != (other.w, other.d):
            raise ValueError("Incompatible sketches")
        result = CountMinSketch(self.w, self.d)
        result.table = self.table + other.table
        result.total = self.total + other.total
        return result