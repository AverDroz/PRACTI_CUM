"""
DQN агент — реализация по образцу PyTorch tutorial + NandaKishoreJoshi Gridworld.
Ключевое отличие от прошлой версии:
  - target network синхронизируется каждые N ШАГОВ (не эпизодов)
  - epsilon decay линейно по шагам
  - никакого Double DQN overhead — чистый ванильный DQN как в рабочих примерах
"""

import numpy as np
import random
from collections import deque
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Архитектура из PyTorch DQN tutorial — 128-128."""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 lr: float = 1e-3,
                 gamma: float = 0.9,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 2000,   # линейный decay по шагам
                 buffer_size: int = 1000,            # как у NandaKishore
                 batch_size: int = 200,              # как у NandaKishore
                 sync_freq: int = 500):              # sync target каждые 500 шагов

        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_freq = sync_freq

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.total_steps = 0  # глобальный счётчик шагов

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=buffer_size)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(t).argmax().item()

    def step(self, state, action, reward, next_state, done) -> Optional[float]:
        """Сохранить transition, обучить, обновить epsilon и target."""
        self.memory.append((state, action, reward, next_state, done))
        self.total_steps += 1

        # Линейный epsilon decay по шагам — как в PyTorch tutorial
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

        # Sync target network каждые sync_freq шагов
        if self.total_steps % self.sync_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return self._learn()

    def _learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        S  = torch.FloatTensor(np.array(s)).to(self.device)
        A  = torch.LongTensor(a).to(self.device)
        R  = torch.FloatTensor(r).to(self.device)
        S2 = torch.FloatTensor(np.array(s2)).to(self.device)
        D  = torch.FloatTensor(d).to(self.device)

        # Q(s, a)
        Q = self.q_net(S).gather(1, A.unsqueeze(1)).squeeze(1)

        # target = r + gamma * max Q_target(s') * (1 - done)
        with torch.no_grad():
            Q2 = self.target_net(S2).max(1)[0]
            target = R + self.gamma * Q2 * (1 - D)

        loss = self.loss_fn(Q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path: str):
        torch.save({
            'q': self.q_net.state_dict(),
            'target': self.target_net.state_dict(),
            'opt': self.optimizer.state_dict(),
            'eps': self.epsilon,
            'steps': self.total_steps,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ck['q'])
        self.target_net.load_state_dict(ck['target'])
        self.optimizer.load_state_dict(ck['opt'])
        self.epsilon = ck['eps']
        self.total_steps = ck.get('steps', 0)
