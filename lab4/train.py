"""
Быстрое обучение без визуала.
    python train.py               # random map
    python train.py --fixed       # fixed map
    python train.py --fixed --episodes 1000
"""

import argparse
import json
import numpy as np
from tqdm import tqdm
from tank_env import TankEnv, TankConfig
from dqn_agent import DQNAgent


def train(episodes=800, fixed_map=False, save_every=200):
    config = TankConfig(grid_size=7, max_steps=60, num_obstacles=5,
                        fixed_map=fixed_map, gamma=0.99)
    env = TankEnv(config)
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        lr=5e-4, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997,
        buffer_size=10000, batch_size=64, hidden=128
    )

    log = []
    best_reward = -float('inf')
    suffix = "fixed" if fixed_map else "random"

    for ep in tqdm(range(1, episodes+1), desc=f"[{suffix}]"):
        state = env.reset()
        total_r = 0.0
        steps = 0

        while True:
            a = agent.act(state, training=True)
            ns, r, done, info = env.step(a)
            agent.remember(state, a, r, ns, done)
            agent.replay()
            agent.update_target(tau=0.005)
            state = ns
            total_r += r
            steps += 1
            if done:
                break

        agent.decay_epsilon()
        success = info['dist'] == 0
        log.append({"ep": ep, "reward": float(total_r), "steps": steps,
                    "eps": float(agent.epsilon), "success": bool(success)})

        if total_r > best_reward:
            best_reward = total_r
            agent.save(f"best_{suffix}.pth")

        if ep % save_every == 0:
            agent.save(f"ckpt_{suffix}_{ep}.pth")
            with open(f"log_{suffix}.json", "w") as f:
                json.dump(log, f)
            recent = log[-100:]
            sr = np.mean([e["success"] for e in recent])
            ar = np.mean([e["reward"] for e in recent])
            tqdm.write(f"Ep {ep:5d} | R:{ar:8.2f} | SR:{sr:.0%} | ε:{agent.epsilon:.3f}")

    agent.save(f"final_{suffix}.pth")
    with open(f"log_{suffix}.json", "w") as f:
        json.dump(log, f)
    print(f"\n✓ Done. Best reward: {best_reward:.2f}")
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--episodes", type=int, default=800)
    args = p.parse_args()
    train(episodes=args.episodes, fixed_map=args.fixed)
