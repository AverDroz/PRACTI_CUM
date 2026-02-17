"""MinHash - оценка Jaccard similarity для множеств."""

import numpy as np
from typing import Set
from bloom_filter import HashFunction


class MinHash:
    """MinHash signature for set similarity estimation."""
    
    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.hashes = [HashFunction(i * 101 + 17, 2**32 - 1) for i in range(num_perm)]
        self.signature = np.full(num_perm, np.inf, dtype=np.float64)
    
    def update(self, item: str) -> None:
        """Обновить сигнатуру новым элементом."""
        for i, h in enumerate(self.hashes):
            self.signature[i] = min(self.signature[i], h(item))
    
    def update_batch(self, items: Set[str]) -> None:
        """Обновить сигнатуру множеством элементов."""
        for item in items:
            self.update(item)
    
    def jaccard(self, other: 'MinHash') -> float:
        """Оценка Jaccard similarity: |A ∩ B| / |A ∪ B|."""
        if self.num_perm != other.num_perm:
            raise ValueError("Different number of permutations")
        return np.mean(self.signature == other.signature)
    
    def __eq__(self, other: 'MinHash') -> bool:
        return np.array_equal(self.signature, other.signature)