"""Логика времени суток — управляет self-modifying поведением классов."""

from enum import Enum, auto


class TimeOfDay(Enum):
    MORNING = auto()  # утро:    low,  Demi растёт,     Pauvre едят макс,   Malheureux активны
    DAY     = auto()  # день:    sun,  Lumiere растёт,  Pauvre активны,     Malheureux спят
    EVENING = auto()  # сумерки: low,  Demi растёт,     Pauvre мало едят,   Malheureux активны
    NIGHT   = auto()  # ночь:    moon, Obscurite растёт,все спят


# Освещённость для каждого времени суток.
# По заданию Demi растут при "low" — утром И в сумерках.
# Баланс удерживается через spread_chance в plants.py (Demi=0.35, остальные=0.6).
LIGHT = {
    TimeOfDay.MORNING: "low",
    TimeOfDay.DAY:     "sun",
    TimeOfDay.EVENING: "low",   # сумерки = низкая освещённость (по заданию)
    TimeOfDay.NIGHT:   "moon",
}

_CYCLE = [TimeOfDay.MORNING, TimeOfDay.DAY, TimeOfDay.EVENING, TimeOfDay.NIGHT]


class Clock:
    """Часы симуляции — управляют сменой времени."""

    def __init__(self, ticks_per_phase: int = 3):
        self.ticks_per_phase = ticks_per_phase
        self._phase_idx = 0
        self._tick = 0
        self.time = _CYCLE[0]
        self.light = LIGHT[self.time]

    def tick(self) -> bool:
        """Обновить часы. Возвращает True если сменилась фаза."""
        self._tick += 1
        if self._tick >= self.ticks_per_phase:
            self._tick = 0
            self._phase_idx = (self._phase_idx + 1) % len(_CYCLE)
            self.time = _CYCLE[self._phase_idx]
            self.light = LIGHT[self.time]
            return True
        return False

    def __repr__(self) -> str:
        return f"[{self.time.name}|{self.light}]"