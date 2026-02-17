"""Растения с self-modifying поведением через динамическую замену методов."""

import random
from time_cycle import TimeOfDay


class Plant:
    """Базовый класс растения. Метод spread() подменяется в зависимости от освещения."""

    symbol: str = "?"
    active_light: str = ""   # при каком освещении активно

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def is_active(self, light: str) -> bool:
        return light == self.active_light

    # spread() — self-modifying: подменяется через adapt_to_environment()
    def spread(self, world) -> list["Plant"]:
        return []  # пассивный режим по умолчанию

    def __repr__(self) -> str:
        return self.symbol


class Lumiere(Plant):
    symbol = "L"
    active_light = "sun"
    spread_chance = 0.6


class Obscurite(Plant):
    symbol = "O"
    active_light = "moon"
    spread_chance = 0.6


class Demi(Plant):
    symbol = "D"
    active_light = "low"
    spread_chance = 0.3   # активна 2 фазы из 4 — занижаем чтобы не доминировала


# ── Self-modifying механизм ────────────────────────────────────────────────

def _active_spread(self, world) -> list[Plant]:
    """Активное распространение — заменяет spread() когда освещение подходит."""
    if random.random() >= getattr(type(self), "spread_chance", 0.5):
        return []
    neighbors = world.get_empty_or_passive_neighbors(self.x, self.y, self)
    if not neighbors:
        return []
    x, y = random.choice(neighbors)
    return [type(self)(x, y)]


def _passive_spread(self, world) -> list[Plant]:
    """Пассивное распространение — вытесняется активными растениями."""
    return []


def adapt_plants_to_environment(plant_classes: list[type], light: str) -> None:
    """
    Self-modifying: подменяем метод spread() на уровне КЛАССА.
    Все экземпляры мгновенно получают новое поведение.
    """
    for cls in plant_classes:
        if cls.active_light == light:
            cls.spread = _active_spread   # активный режим
        else:
            cls.spread = _passive_spread  # пассивный режим