"""Мир симуляции — двумерная сетка, тики, взаимодействия."""

import random
from time_cycle import Clock
from plants import Plant, Lumiere, Obscurite, Demi, adapt_plants_to_environment
from animals import Animal, Pauvre, Malheureux, adapt_animals_to_environment

PLANT_CLASSES = [Lumiere, Obscurite, Demi]


class World:
    """
    Двумерная сетка m×n.
    Каждая клетка: {"plant": Plant | None, "animal": Animal | None}
    """

    def __init__(self, rows: int = 20, cols: int = 20,
                 plant_density: float = 0.3,
                 pauvre_count: int = 10,
                 malheureux_count: int = 5,
                 ticks_per_phase: int = 3):
        self.rows = rows
        self.cols = cols
        self.clock = Clock(ticks_per_phase)

        # Инициализация сетки
        self.grid: list[list[dict]] = [
            [{"plant": None, "animal": None} for _ in range(cols)]
            for _ in range(rows)
        ]

        self._spawn_plants(plant_density)
        self.animals: list[Animal] = []
        self._spawn_animals(Pauvre, pauvre_count)
        self._spawn_animals(Malheureux, malheureux_count)

        # Применяем начальное состояние
        adapt_plants_to_environment(PLANT_CLASSES, self.clock.light)
        adapt_animals_to_environment(self.clock.time)

    # ── Инициализация ──────────────────────────────────────────────────────

    def _spawn_plants(self, density: float) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                if random.random() < density:
                    cls = random.choice(PLANT_CLASSES)
                    self.grid[r][c]["plant"] = cls(r, c)

    def _spawn_animals(self, cls: type, count: int) -> None:
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if self.grid[r][c]["animal"] is None]
        random.shuffle(positions)
        group: list[Animal] = []
        for r, c in positions[:count]:
            a = cls(r, c)
            a.group = group
            group.append(a)
            self.grid[r][c]["animal"] = a
            self.animals.append(a)

    # ── Утилиты ────────────────────────────────────────────────────────────

    def get_adjacent(self, r: int, c: int) -> list[tuple[int, int]]:
        """Соседние клетки (4 стороны) в пределах сетки."""
        return [(r+dr, c+dc)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= r+dr < self.rows and 0 <= c+dc < self.cols]

    def get_empty_or_passive_neighbors(self, r: int, c: int,
                                        spreader: Plant) -> list[tuple[int, int]]:
        """Клетки куда растение может распространиться (пусто или пассивный конкурент)."""
        result = []
        for nr, nc in self.get_adjacent(r, c):
            existing = self.grid[nr][nc]["plant"]
            if existing is None:
                result.append((nr, nc))
            elif not existing.is_active(self.clock.light):
                # Вытесняем пассивное растение с вероятностью 0.4
                if random.random() < 0.2:
                    result.append((nr, nc))
        return result

    # ── Тик симуляции ──────────────────────────────────────────────────────

    def tick(self) -> None:
        """Один шаг симуляции."""
        phase_changed = self.clock.tick()

        # Self-modifying: подменяем методы при смене фазы
        if phase_changed:
            adapt_plants_to_environment(PLANT_CLASSES, self.clock.light)
            adapt_animals_to_environment(self.clock.time)

        # Растения распространяются
        new_plants: list[Plant] = []
        for r in range(self.rows):
            for c in range(self.cols):
                plant = self.grid[r][c]["plant"]
                if plant:
                    new_plants.extend(plant.spread(self))

        for p in new_plants:
            self.grid[p.x][p.y]["plant"] = p

        # Животные действуют
        new_animals: list[Animal] = []
        random.shuffle(self.animals)

        for animal in self.animals:
            if not animal.alive:
                continue
            # Убираем со старой позиции
            self.grid[animal.x][animal.y]["animal"] = None

            animal.eat(self)
            animal.move(self)
            animal.hunger += 1

            # Смерть от голода
            if animal.hunger >= animal.max_hunger:
                animal.alive = False
                continue

            # Размещаем на новой позиции
            self.grid[animal.x][animal.y]["animal"] = animal
            new_animals.extend(animal.reproduce(self))

        # Убираем мёртвых, добавляем новорождённых
        self.animals = [a for a in self.animals if a.alive] + new_animals
        for a in new_animals:
            self.grid[a.x][a.y]["animal"] = a

    def stats(self) -> dict:
        plants = {"L": 0, "O": 0, "D": 0}
        for r in range(self.rows):
            for c in range(self.cols):
                p = self.grid[r][c]["plant"]
                if p:
                    plants[p.symbol] = plants.get(p.symbol, 0) + 1
        pauvre    = sum(1 for a in self.animals if isinstance(a, Pauvre))
        malheureux = sum(1 for a in self.animals if isinstance(a, Malheureux))
        return {**plants, "P": pauvre, "M": malheureux}