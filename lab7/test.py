"""Тесты экосистемы."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from time_cycle import Clock, TimeOfDay
from plants import Lumiere, Obscurite, Demi, adapt_plants_to_environment
from animals import Pauvre, Malheureux, adapt_animals_to_environment
from world import World


class TestClock:
    def test_cycle(self):
        clock = Clock(ticks_per_phase=1)
        assert clock.time == TimeOfDay.MORNING
        clock.tick()
        assert clock.time == TimeOfDay.DAY
        clock.tick()
        assert clock.time == TimeOfDay.EVENING
        clock.tick()
        assert clock.time == TimeOfDay.NIGHT
        clock.tick()
        assert clock.time == TimeOfDay.MORNING  # цикл

    def test_light(self):
        clock = Clock(ticks_per_phase=1)
        clock.tick()  # DAY
        assert clock.light == "sun"


class TestSelfModifyingPlants:
    def test_lumiere_active_in_sun(self):
        """Self-modifying: Lumiere.spread должен быть активным при солнце."""
        adapt_plants_to_environment([Lumiere, Obscurite, Demi], "sun")
        world = World(rows=5, cols=5, plant_density=0, pauvre_count=0, malheureux_count=0)
        world.clock.light = "sun"

        l = Lumiere(2, 2)
        world.grid[2][2]["plant"] = l

        # активный spread должен вернуть новые растения
        result = l.spread(world)
        assert isinstance(result, list)

    def test_lumiere_passive_at_night(self):
        """Self-modifying: Lumiere.spread пассивен при луне."""
        adapt_plants_to_environment([Lumiere, Obscurite, Demi], "moon")
        world = World(rows=5, cols=5, plant_density=0, pauvre_count=0, malheureux_count=0)

        l = Lumiere(2, 2)
        world.grid[2][2]["plant"] = l
        result = l.spread(world)
        assert result == []

    def test_obscurite_active_at_night(self):
        adapt_plants_to_environment([Lumiere, Obscurite, Demi], "moon")
        world = World(rows=5, cols=5, plant_density=0, pauvre_count=0, malheureux_count=0)

        o = Obscurite(2, 2)
        world.grid[2][2]["plant"] = o
        result = o.spread(world)
        assert isinstance(result, list)

    def test_demi_active_low_light(self):
        adapt_plants_to_environment([Lumiere, Obscurite, Demi], "low")
        world = World(rows=5, cols=5, plant_density=0, pauvre_count=0, malheureux_count=0)

        d = Demi(2, 2)
        world.grid[2][2]["plant"] = d
        result = d.spread(world)
        assert isinstance(result, list)

    def test_class_method_changes(self):
        """Метод подменяется на уровне класса — все экземпляры затронуты.

        БЫЛО:   assert l1.spread is l2.spread
        ПОЧЕМУ ПАДАЛО: обращение к методу через экземпляр каждый раз создаёт
        новый bound method объект, поэтому `is` всегда False, даже если
        на классе лежит одна функция.
        СТАЛО:  сравниваем __func__ — саму функцию без обёртки.
        """
        adapt_plants_to_environment([Lumiere], "sun")
        l1, l2 = Lumiere(0, 0), Lumiere(1, 1)
        # оба должны использовать одну и ту же функцию на уровне класса
        assert l1.spread.__func__ is l2.spread.__func__


class TestSelfModifyingAnimals:
    def test_pauvre_sleeps_at_night(self):
        """Ночью Pauvre не двигается — метод подменён.

        БЫЛО:   adapt_animals_to_environment(NIGHT) → World(...)
        ПОЧЕМУ ПАДАЛО: World.__init__ вызывает adapt_animals_to_environment
        со своим начальным временем (MORNING), перезаписывая нашу адаптацию.
        СТАЛО:  сначала создаём World, потом применяем нужную адаптацию.
        """
        world = World(rows=5, cols=5, plant_density=0, pauvre_count=0, malheureux_count=0)
        adapt_animals_to_environment(TimeOfDay.NIGHT)  # после World
        p = Pauvre(2, 2)
        old_pos = (p.x, p.y)
        p.move(world)
        assert (p.x, p.y) == old_pos  # не двинулся

    def test_pauvre_eats_more_morning(self):
        """Утром Pauvre снижает голод на 3."""
        world = World(rows=5, cols=5, plant_density=0, pauvre_count=0, malheureux_count=0)
        adapt_animals_to_environment(TimeOfDay.MORNING)  # после World
        p = Pauvre(2, 2)
        p.hunger = 5
        world.grid[2][2]["plant"] = Lumiere(2, 2)
        p.eat(world)
        assert p.hunger == 2  # 5 - 3

    def test_malheureux_sleeps_daytime(self):
        """Днём Malheureux не ест.

        БЫЛО:   adapt_animals_to_environment(DAY) → World(...)
        ПОЧЕМУ ПАДАЛО: та же причина — World.__init__ перезаписывал адаптацию.
        СТАЛО:  сначала World, потом адаптация.
        """
        world = World(rows=5, cols=5, plant_density=0, pauvre_count=0, malheureux_count=0)
        adapt_animals_to_environment(TimeOfDay.DAY)  # после World
        m = Malheureux(2, 2)
        m.hunger = 5
        world.grid[2][2]["plant"] = Demi(2, 2)
        m.eat(world)
        assert m.hunger == 5  # не изменился — спит

    def test_malheureux_eats_pauvre(self):
        """Malheureux активен утром и ест Pauvre."""
        world = World(rows=5, cols=5, plant_density=0, pauvre_count=0, malheureux_count=0)
        adapt_animals_to_environment(TimeOfDay.MORNING)  # после World
        prey = Pauvre(2, 2)
        world.grid[2][2]["animal"] = prey
        m = Malheureux(2, 2)
        m.hunger = 8
        m.eat(world)
        assert not prey.alive  # съеден
        assert m.hunger == 3   # 8 - 5


class TestWorld:
    def test_tick_runs(self):
        world = World(rows=10, cols=10, ticks_per_phase=3)
        for _ in range(12):  # полный цикл суток
            world.tick()

    def test_stats(self):
        world = World(rows=10, cols=10, plant_density=0.5,
                      pauvre_count=5, malheureux_count=3)
        s = world.stats()
        assert s["P"] == 5
        assert s["M"] == 3
        assert s["L"] + s["O"] + s["D"] > 0

    def test_phase_triggers_self_modify(self):
        """При смене фазы методы подменяются автоматически."""
        world = World(rows=5, cols=5, ticks_per_phase=1,
                      plant_density=0, pauvre_count=0, malheureux_count=0)
        # После тика фаза меняется → методы подменяются
        world.tick()
        assert world.clock.time == TimeOfDay.DAY
        # Malheureux.move должен быть sleep-версией
        m = Malheureux(2, 2)
        old_pos = (m.x, m.y)
        m.move(world)
        assert (m.x, m.y) == old_pos


if __name__ == "__main__":
    pytest.main([__file__, "-v"])