"""Quotient Filter - компактная альтернатива Bloom Filter."""

import numpy as np
from dataclasses import dataclass
import hashlib


@dataclass
class QuotientConfig:
    q: int  # quotient bits
    r: int  # remainder bits
    
    @property
    def size(self) -> int:
        return 1 << self.q


class QuotientFilter:
    """Quotient Filter с динамическим разрешением коллизий."""
    
    def __init__(self, config: QuotientConfig):
        self.config = config
        self.size = config.size
        self.remainders = np.zeros(self.size, dtype=np.uint64)
        self.is_occupied = np.zeros(self.size, dtype=bool)
        self.is_continuation = np.zeros(self.size, dtype=bool)
        self.is_shifted = np.zeros(self.size, dtype=bool)
        self.n = 0
    
    def add(self, item: str) -> None:
        """Вставка элемента с linear probing."""
        h = int(hashlib.sha256(item.encode()).hexdigest()[:16], 16)
        q = (h >> self.config.r) & (self.size - 1)
        r = h & ((1 << self.config.r) - 1)
        
        if not self.is_occupied[q]:
            self.remainders[q] = r
            self.is_occupied[q] = True
            self.n += 1
            return
        
        # Linear probing для поиска свободного слота
        pos = q
        while self.is_occupied[pos]:
            if self.remainders[pos] == r:
                return  # уже есть
            pos = (pos + 1) % self.size
            if pos == q:
                raise MemoryError("Filter full")
        
        self.remainders[pos] = r
        self.is_occupied[pos] = True
        self.is_shifted[pos] = True
        self.n += 1
    
    def __contains__(self, item: str) -> bool:
        """Проверка принадлежности."""
        h = int(hashlib.sha256(item.encode()).hexdigest()[:16], 16)
        q = (h >> self.config.r) & (self.size - 1)
        r = h & ((1 << self.config.r) - 1)
        
        if not self.is_occupied[q]:
            return False
        
        pos = q
        while self.is_occupied[pos]:
            if self.remainders[pos] == r:
                return True
            pos = (pos + 1) % self.size
            if pos == q:
                break
        return False
    
    @property
    def load_factor(self) -> float:
        return self.n / self.size