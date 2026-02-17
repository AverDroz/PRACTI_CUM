"""HyperLogLog - кардинальность множества с O(log log n) памятью."""

import hashlib
import numpy as np


class HyperLogLog:
    """Оценка кардинальности с точностью ~1.04/sqrt(m)."""

    def __init__(self, precision: int = 14):
        self.p = precision
        self.m = 1 << precision
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.alpha = self._alpha_m(self.m)

    def add(self, item: str) -> None:
        h = int(hashlib.sha256(item.encode()).hexdigest()[:16], 16)
        j = h & (self.m - 1)           # первые p бит → номер регистра
        w = h >> self.p                 # оставшиеся (64-p) бит
        self.registers[j] = max(self.registers[j], self._clz(w) + 1)

    def count(self) -> int:
        indicator = float(np.sum(2.0 ** (-self.registers.astype(float))))
        raw = self.alpha * self.m ** 2 / indicator

        zeros = int(np.count_nonzero(self.registers == 0))

        if raw <= 2.5 * self.m and zeros > 0:
            return int(self.m * np.log(self.m / zeros))

        two32 = 1 << 32
        if raw > two32 / 30:
            return int(-two32 * np.log(1.0 - raw / two32))

        return int(raw)

    def _clz(self, w: int) -> int:
        """Ведущие нули в (64-p)-битном пространстве."""
        bits = 64 - self.p
        if w == 0:
            return bits
        return bits - w.bit_length()

    @staticmethod
    def _alpha_m(m: int) -> float:
        if m >= 128:
            return 0.7213 / (1 + 1.079 / m)
        if m >= 64:
            return 0.709
        if m >= 32:
            return 0.697
        return 0.673

    def __or__(self, other: 'HyperLogLog') -> 'HyperLogLog':
        if self.p != other.p:
            raise ValueError("Different precision")
        result = HyperLogLog(self.p)
        result.registers = np.maximum(self.registers, other.registers)
        return result