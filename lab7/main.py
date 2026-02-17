"""Точка входа. Запуск: python main.py [--rows 20] [--cols 20] [--ticks 100]"""

import argparse
from visualizer import run


def main():
    p = argparse.ArgumentParser(description="Ecosystem simulation with self-modifying classes")
    p.add_argument("--rows",           type=int,   default=20,  help="Высота поля")
    p.add_argument("--cols",           type=int,   default=40,  help="Ширина поля")
    p.add_argument("--ticks",          type=int,   default=100, help="Количество тиков")
    p.add_argument("--phase-ticks",    type=int,   default=3,   help="Тиков на фазу суток")
    p.add_argument("--density",        type=float, default=0.3, help="Плотность растений 0-1")
    p.add_argument("--pauvre",         type=int,   default=15,  help="Начальное кол-во Pauvre")
    p.add_argument("--malheureux",     type=int,   default=3,   help="Начальное кол-во Malheureux")
    p.add_argument("--delay",          type=float, default=0.15,help="Задержка между тиками (сек)")
    args = p.parse_args()

    run(
        rows=args.rows,
        cols=args.cols,
        ticks=args.ticks,
        ticks_per_phase=args.phase_ticks,
        plant_density=args.density,
        pauvre_count=args.pauvre,
        malheureux_count=args.malheureux,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()