"""Животные с self-modifying поведением — методы eat/move/reproduce подменяются."""

import random
from time_cycle import TimeOfDay
from plants import Lumiere, Demi, Obscurite


class Animal:
    """Базовый класс животного."""

    symbol: str = "?"
    max_hunger: int = 10

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.hunger = 0
        self.alive = True
        self.group: list["Animal"] = [self]

    def move(self, world) -> None: ...
    def eat(self, world) -> None: ...
    def reproduce(self, world) -> list["Animal"]: return []

    def _random_step(self, world) -> None:
        neighbors = world.get_adjacent(self.x, self.y)
        if neighbors:
            self.x, self.y = random.choice(neighbors)

    def __repr__(self) -> str:
        return f"{self.symbol}(h={self.hunger})"


# ── Pauvre (травоядные) ────────────────────────────────────────────────────

class Pauvre(Animal):
    """
    Травоядные. Едят Lumiere.
    Self-modifying: при смене времени суток подменяются eat() и move().
    """
    symbol = "P"
    max_hunger = 25


# ── Malheureux (хищники) ──────────────────────────────────────────────────

class Malheureux(Animal):
    """
    Всеядные хищники. Едят Demi, Obscurite, Pauvre.
    Self-modifying: активны только утром и вечером.
    """
    symbol = "M"
    max_hunger = 30


# ── Self-modifying методы для Pauvre ──────────────────────────────────────

def _pauvre_eat_morning(self, world) -> None:
    """Утром едят максимально много."""
    cell = world.grid[self.x][self.y]
    if isinstance(cell.get("plant"), Lumiere):
        self.hunger = max(0, self.hunger - 3)
        cell["plant"] = None


def _pauvre_eat_default(self, world) -> None:
    """Вечером почти не едят."""
    cell = world.grid[self.x][self.y]
    if isinstance(cell.get("plant"), Lumiere) and random.random() < 0.2:
        self.hunger = max(0, self.hunger - 1)
        cell["plant"] = None


def _pauvre_eat_day(self, world) -> None:
    """Днём едят нормально."""
    cell = world.grid[self.x][self.y]
    if isinstance(cell.get("plant"), Lumiere):
        self.hunger = max(0, self.hunger - 2)
        cell["plant"] = None


def _pauvre_move_active(self, world) -> None:
    """Активное движение: убегает от хищников, ищет Lumiere, агрессия при перенаселении."""
    neighbors = world.get_adjacent(self.x, self.y)

    # Агрессия: перенаселение группы ИЛИ очень голодный
    if len(self.group) > 2 or self.hunger > 10:
        for nx, ny in neighbors:
            target = world.grid[nx][ny].get("animal")
            if isinstance(target, Pauvre) and target is not self:
                target.hunger += 2
                return

    # Побег от Malheureux в соседних клетках
    danger = {(nx, ny) for nx, ny in neighbors
              if isinstance(world.grid[nx][ny].get("animal"), Malheureux)}
    safe = [p for p in neighbors if p not in danger]
    pool = safe if (danger and safe) else neighbors

    # Предпочитаем клетку с Lumiere
    food = [(nx, ny) for nx, ny in pool
            if isinstance(world.grid[nx][ny].get("plant"), Lumiere)]
    if food:
        self.x, self.y = random.choice(food)
    elif pool:
        self.x, self.y = random.choice(pool)


def _pauvre_move_sleep(self, world) -> None:
    """Ночью спят — не двигаются."""
    pass


def _pauvre_reproduce(self, world) -> list[Animal]:
    """Размножение внутри группы при достаточном количестве еды."""
    if len(self.group) >= 1 and self.hunger < 12 and random.random() < 0.25:
        nx, ny = random.choice(world.get_adjacent(self.x, self.y) or [(self.x, self.y)])
        child = Pauvre(nx, ny)
        self.group.append(child)
        # Большие группы распадаются
        if len(self.group) > 5:
            half = len(self.group) // 2
            child.group = self.group[half:]
            self.group = self.group[:half]
        return [child]
    return []


# ── Self-modifying методы для Malheureux ──────────────────────────────────

def _malheureux_eat_active(self, world) -> None:
    """Активная охота — едят растения и Pauvre."""
    cell = world.grid[self.x][self.y]
    # Сначала пробуем съесть Pauvre
    prey = cell.get("animal")
    if isinstance(prey, Pauvre):
        prey.alive = False
        cell["animal"] = None
        self.hunger = max(0, self.hunger - 5)
        return
    # Иначе едим растения
    plant = cell.get("plant")
    if isinstance(plant, (Demi, Obscurite)):
        self.hunger = max(0, self.hunger - 2)
        cell["plant"] = None


def _malheureux_eat_sleep(self, world) -> None:
    """Спят — не едят."""
    pass


def _malheureux_move_active(self, world) -> None:
    """Активное движение — быстрее при сытости."""
    steps = 1 if self.hunger > 8 else 2  # голодные медленнее
    for _ in range(steps):
        self._random_step(world)


def _malheureux_move_sleep(self, world) -> None:
    """Спят."""
    pass


def _malheureux_reproduce(self, world) -> list[Animal]:
    """Размножение. Одиночный тоже может размножиться — минимальный размер стаи 1."""
    if len(self.group) >= 1 and self.hunger < 12 and random.random() < 0.08:
        nx, ny = random.choice(world.get_adjacent(self.x, self.y) or [(self.x, self.y)])
        child = Malheureux(nx, ny)
        self.group.append(child)
        return [child]
    return []


# ── Self-modifying механизм ────────────────────────────────────────────────

def adapt_animals_to_environment(time: TimeOfDay) -> None:
    """
    Self-modifying: подменяем методы eat/move/reproduce на уровне КЛАССА
    в зависимости от времени суток.
    """
    if time == TimeOfDay.MORNING:
        Pauvre.eat      = _pauvre_eat_morning
        Pauvre.move     = _pauvre_move_active
        Pauvre.reproduce = _pauvre_reproduce
        Malheureux.eat  = _malheureux_eat_active
        Malheureux.move = _malheureux_move_active
        Malheureux.reproduce = _malheureux_reproduce

    elif time == TimeOfDay.DAY:
        Pauvre.eat      = _pauvre_eat_day
        Pauvre.move     = _pauvre_move_active
        Pauvre.reproduce = _pauvre_reproduce
        Malheureux.eat  = _malheureux_eat_sleep   # спят днём
        Malheureux.move = _malheureux_move_sleep
        Malheureux.reproduce = lambda self, w: []

    elif time == TimeOfDay.EVENING:
        Pauvre.eat      = _pauvre_eat_default
        Pauvre.move     = _pauvre_move_active
        Pauvre.reproduce = lambda self, w: []
        Malheureux.eat  = _malheureux_eat_active  # активны вечером
        Malheureux.move = _malheureux_move_active
        Malheureux.reproduce = _malheureux_reproduce

    elif time == TimeOfDay.NIGHT:
        Pauvre.eat      = lambda self, w: None    # спят
        Pauvre.move     = _pauvre_move_sleep
        Pauvre.reproduce = lambda self, w: []
        Malheureux.eat  = _malheureux_eat_sleep
        Malheureux.move = _malheureux_move_sleep
        Malheureux.reproduce = lambda self, w: []