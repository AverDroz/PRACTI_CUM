import numpy as np
import hashlib


class HyperLogLog:
    """Оценка кардинальности с точностью ~2% используя log2(log2(n)) битов."""
    
    def __init__(self, precision: int = 14):
        self.p = precision
        self.m = 1 << precision  # 2^p buckets
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.alpha = self._alpha_m(self.m)
    
    def add(self, item: str) -> None:
        """Добавить элемент O(1)."""
        # Используем 64-битный хеш
        h = int(hashlib.sha256(item.encode()).hexdigest()[:16], 16)
        
        # Первые p бит для выбора регистра
        j = h & ((1 << self.p) - 1)
        
        # Остальные биты для подсчета leading zeros
        w = h >> self.p
        
        # Количество ведущих нулей + 1
        self.registers[j] = max(self.registers[j], self._leading_zero_count(w) + 1)
    
    def count(self) -> int:
        """Оценка количества уникальных элементов."""
        # Harmonic mean
        raw_estimate = self.alpha * self.m ** 2 / np.sum(2.0 ** (-self.registers))
        
        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            zeros = np.count_nonzero(self.registers == 0)
            if zeros != 0:
                return int(self.m * np.log(self.m / zeros))
        
        # No correction needed for medium range
        if raw_estimate <= (1 << 32) / 30:
            return int(raw_estimate)
        
        # Large range correction
        return int(-((1 << 32) * np.log(1 - raw_estimate / (1 << 32))))
    
    @staticmethod
    def _leading_zero_count(w: int) -> int:
        """Подсчет ведущих нулей в 64-битном числе."""
        if w == 0:
            return 64
        
        # Считаем позицию первого единичного бита
        # bit_length() возвращает количество бит для представления числа
        # 64 - bit_length = количество ведущих нулей
        return 64 - w.bit_length()
    
    @staticmethod
    def _alpha_m(m: int) -> float:
        """Bias correction constant."""
        if m >= 128:
            return 0.7213 / (1 + 1.079 / m)
        if m >= 64:
            return 0.709
        if m >= 32:
            return 0.697
        return 0.673
    
    def __or__(self, other: 'HyperLogLog') -> 'HyperLogLog':
        """Объединение HLL (берем максимум из регистров)."""
        if self.p != other.p:
            raise ValueError("Different precision")
        result = HyperLogLog(self.p)
        result.registers = np.maximum(self.registers, other.registers)
        return result