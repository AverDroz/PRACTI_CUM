# Lab 4: Reinforcement Learning — Tank Agent

DQN-агент обучается находить цель на 2D-карте с препятствиями и поддержкой выстрелов.

## Установка

```bash
pip install torch numpy pygame matplotlib tqdm
```

## Запуск

### Live-обучение с визуализацией (рекомендуется)
```bash
python live_train.py              # случайная карта каждый эпизод
python live_train.py --fixed      # фиксированная карта
python live_train.py --fixed --episodes 800
```

### Быстрое обучение без визуала
```bash
python train.py
python train.py --fixed --episodes 2000
```

## Структура

```
├── tank_env.py      # среда: стены, препятствия, выстрелы, fixed/random режим
├── dqn_agent.py     # Double DQN + Huber loss + gradient clipping
├── live_train.py    # обучение с визуализацией (танк, флаг, стены, путь, лучи)
├── train.py         # быстрое обучение без pygame
└── plot_results.py  # графики по training_log_*.json
```

## Среда

**Состояние (13 признаков):**
- Позиция танка (x/size, y/size)
- Угол поворота (sin, cos)
- Вектор к цели (dx, dy, dist)
- Угол к цели (sin, cos)
- Расстояние до стены/препятствия в 4 направлениях (ray casting)

**Действия (5):**
- 0: Вперёд, 1: Назад, 2: Поворот влево, 3: Поворот вправо, 4: Выстрел

**Награды:**
- +100: достижение цели движением
- +100: попадание выстрелом в цель
- -10: столкновение со стеной/препятствием
- -1: за каждый шаг (штраф за время)
- +10/-5: за приближение/удаление от цели (пропорционально)
- -2: пустой выстрел

## Режимы карты

| Режим | Описание |
|-------|----------|
| `random` (default) | Каждый эпизод — новая карта, позиции танка и цели рандомны. Агент обучается обобщённой стратегии. |
| `fixed` (`--fixed`) | Карта фиксирована. Быстрее сходится, проще задача. |

## DQN Architecture

```
State (13) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(5)
```

**Double DQN**: выбор действия через q_network, оценка через target_network — меньше overestimation.

**Гиперпараметры:**
- LR: 1e-3 (StepLR: ×0.5 каждые 500 эп.)
- Gamma: 0.97
- Epsilon: 1.0 → 0.05 (decay 0.995)
- Buffer: 20 000, Batch: 64
- Huber Loss + gradient clipping (norm 1.0)
- Soft target update: τ=0.01 каждые 10 эпизодов

## Результаты

- Fixed map: ~80% SR после 400–600 эпизодов
- Random map: ~60–70% SR после 1000–1500 эпизодов
