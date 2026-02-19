"""
Tank environment — состояние = рендер доски каналами (как в DeepMind Gridworld).
Никакого ручного feature engineering — сеть видит карту напрямую.
"""

import numpy as np
import random
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TankConfig:
    grid_size: int = 6
    max_steps: int = 50
    num_obstacles: int = 4
    fixed_map: bool = False


class TankEnv:
    """
    Состояние: 4 канала grid_size×grid_size, сплющенные в вектор.
        Канал 0: позиция танка
        Канал 1: позиция цели
        Канал 2: препятствия
        Канал 3: направление танка (нормализовано 0..1)

    Это то же что делает DeepMind Gridworld — сеть сама читает карту.

    Действия: 0=вперёд, 1=назад, 2=поворот влево, 3=поворот вправо, 4=выстрел
    """

    DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    def __init__(self, config: Optional[TankConfig] = None):
        self.config = config or TankConfig()
        self.size = self.config.grid_size
        self.tank_pos = (0, 0)
        self.tank_angle = 0
        self.goal_pos = (self.size - 1, self.size - 1)
        self.obstacles: List[Tuple[int, int]] = []
        self.steps = 0
        self.done = False
        self.last_shot_ray = None

        self._fixed_obstacles = None
        self._fixed_tank = None
        self._fixed_goal = None
        if self.config.fixed_map:
            self._fixed_tank = (0, 0)
            self._fixed_goal = (self.size - 1, self.size - 1)
            self._fixed_obstacles = self._gen_obstacles(self._fixed_tank, self._fixed_goal)

        self.reset()

    def _gen_obstacles(self, tank, goal):
        forbidden = set()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                forbidden.add((tank[0]+dx, tank[1]+dy))
                forbidden.add((goal[0]+dx, goal[1]+dy))
        cands = [(x, y) for x in range(self.size) for y in range(self.size)
                 if (x, y) not in forbidden]
        random.shuffle(cands)
        return cands[:self.config.num_obstacles]

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.done = False
        self.last_shot_ray = None

        if self.config.fixed_map:
            self.tank_pos = self._fixed_tank
            self.goal_pos = self._fixed_goal
            self.obstacles = list(self._fixed_obstacles)
            self.tank_angle = random.randint(0, 3)
        else:
            all_cells = [(x, y) for x in range(self.size) for y in range(self.size)]
            random.shuffle(all_cells)
            self.tank_pos = all_cells[0]
            self.goal_pos = next(
                (c for c in all_cells[1:] if abs(c[0]-all_cells[0][0]) + abs(c[1]-all_cells[0][1]) >= 3),
                all_cells[-1]
            )
            self.tank_angle = random.randint(0, 3)
            self.obstacles = self._gen_obstacles(self.tank_pos, self.goal_pos)

        return self._render()

    def _dist(self) -> float:
        return abs(self.tank_pos[0]-self.goal_pos[0]) + abs(self.tank_pos[1]-self.goal_pos[1])

    def _potential(self) -> float:
        """Потенциал состояния = -dist/max_dist. Чем ближе к цели — тем выше."""
        max_dist = (self.size - 1) * 2
        return -self._dist() / max_dist

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("done, call reset()")
        self.steps += 1
        self.last_shot_ray = None

        phi_before = self._potential()
        reward = -1.0  # базовый штраф за шаг

        if action == 0:
            reward += self._move(1)
        elif action == 1:
            reward += self._move(-1)
        elif action == 2:
            self.tank_angle = (self.tank_angle - 1) % 4
        elif action == 3:
            self.tank_angle = (self.tank_angle + 1) % 4
        elif action == 4:
            reward += self._shoot()

        # Potential-based reward shaping: F(s,s') = Φ(s') - Φ(s)  (γ=1 избегает discount artifact)
        # Применяем ТОЛЬКО при движении — при повороте phi не меняется, shaping = 0
        # Теорема Ng et al. 1999: оптимальная политика сохраняется
        if action in (0, 1):
            phi_after = self._potential()
            reward += phi_after - phi_before

        if self.tank_pos == self.goal_pos:
            reward += 10.0
            self.done = True
        if self.steps >= self.config.max_steps:
            self.done = True

        return self._render(), reward, self.done, {"steps": self.steps, "dist": self._dist()}

    def _move(self, direction: int) -> float:
        dx, dy = self.DIRS[self.tank_angle]
        nx = self.tank_pos[0] + dx * direction
        ny = self.tank_pos[1] + dy * direction
        if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in self.obstacles:
            self.tank_pos = (nx, ny)
            return 0.0
        return -5.0

    def _shoot(self) -> float:
        dx, dy = self.DIRS[self.tank_angle]
        x, y = self.tank_pos
        ray = []
        while True:
            x += dx; y += dy
            if not (0 <= x < self.size and 0 <= y < self.size):
                break
            if (x, y) in self.obstacles:
                ray.append((x, y))
                break
            ray.append((x, y))
            if (x, y) == self.goal_pos:
                self.last_shot_ray = (self.tank_pos, ray)
                self.done = True
                return 10.0
        self.last_shot_ray = (self.tank_pos, ray)
        return -2.0

    def _render(self) -> np.ndarray:
        """
        4 канала [танк, цель, стены, угол] -> flatten.
        + мелкий шум как в DeepMind для разбиения симметрии.
        """
        s = self.size
        board = np.zeros((4, s, s), dtype=np.float32)
        board[0, self.tank_pos[1], self.tank_pos[0]] = 1.0
        board[1, self.goal_pos[1], self.goal_pos[0]] = 1.0
        for ox, oy in self.obstacles:
            board[2, oy, ox] = 1.0
        board[3, :, :] = self.tank_angle / 3.0
        return board.flatten() + np.random.rand(4 * s * s).astype(np.float32) * 0.01

    def _goal_in_sight(self) -> bool:
        dx, dy = self.DIRS[self.tank_angle]
        x, y = self.tank_pos
        while True:
            x += dx; y += dy
            if not (0 <= x < self.size and 0 <= y < self.size): return False
            if (x, y) in self.obstacles: return False
            if (x, y) == self.goal_pos: return True

    @property
    def state_size(self) -> int:
        return 4 * self.size * self.size

    @property
    def action_size(self) -> int:
        return 5
