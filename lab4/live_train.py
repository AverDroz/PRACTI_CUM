"""
Live training с визуализацией.
    python live_train.py              # random map
    python live_train.py --fixed      # fixed map
"""

import sys
import math
import argparse
import numpy as np
from collections import deque
import pygame

from tank_env import TankEnv, TankConfig
from dqn_agent import DQNAgent

# ── Цвета ─────────────────────────────────────────────────────────────────
BG         = (12, 13, 20)
GRID_BG    = (16, 18, 28)
GRID_LINE  = (24, 28, 42)
PANEL_BG   = (18, 20, 32)
PANEL_EDGE = (40, 46, 72)
C_TANK     = (80, 160, 255)
C_CANNON   = (140, 200, 255)
C_GOAL     = (80, 220, 120)
C_WALL     = (180, 80, 80)
C_PATH     = (255, 190, 60)
C_SHOT     = (255, 80, 80)
C_TEXT     = (180, 188, 215)
C_SUB      = (85, 94, 130)
C_OK       = (80, 220, 120)
C_FAIL     = (255, 90, 90)


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def txt(surf, font, text, x, y, color=C_TEXT):
    surf.blit(font.render(text, True, color), (x, y))


def draw_tank(surf, cx, cy, angle_idx, radius):
    angle_deg = angle_idx * 90
    body_w = int(radius * 1.5)
    body_h = int(radius * 2.0)
    body_surf = pygame.Surface((body_w * 2 + 4, body_h * 2 + 4), pygame.SRCALPHA)
    bx, by = body_w + 2, body_h + 2

    track_col = (*tuple(max(0, c - 50) for c in C_TANK), 210)
    pygame.draw.rect(body_surf, track_col,
                     (bx - body_w//2, by - body_h//2 + 2, body_w//3, body_h - 4), border_radius=2)
    pygame.draw.rect(body_surf, track_col,
                     (bx + body_w//6, by - body_h//2 + 2, body_w//3, body_h - 4), border_radius=2)
    pygame.draw.rect(body_surf, (*C_TANK, 230),
                     (bx - body_w//2, by - body_h//2, body_w, body_h), border_radius=4)
    pygame.draw.circle(body_surf, (*C_CANNON, 245), (bx, by), int(radius * 0.52))
    pygame.draw.rect(body_surf, (*C_CANNON, 245),
                     (bx - 3, by - int(radius * 1.3), 6, int(radius * 0.9)), border_radius=3)

    rotated = pygame.transform.rotate(body_surf, -angle_deg)
    rr = rotated.get_rect(center=(cx, cy))
    surf.blit(rotated, rr.topleft)


def draw_flag(surf, cx, cy, radius):
    h = int(radius * 2.2)
    pygame.draw.rect(surf, C_GOAL, (cx - 2, cy - h, 3, h), border_radius=1)
    pts = [
        (cx + 1, cy - h),
        (cx + 1 + int(radius * 0.9), cy - h + int(radius * 0.5)),
        (cx + 1, cy - h + int(radius)),
    ]
    pygame.draw.polygon(surf, C_GOAL, pts)
    pygame.draw.rect(surf, tuple(max(0, c - 50) for c in C_GOAL),
                     (cx - int(radius * 0.6), cy - 3, int(radius * 1.2), 3), border_radius=2)


def draw_wall(surf, rx, ry, rw, rh):
    pygame.draw.rect(surf, C_WALL, (rx, ry, rw, rh), border_radius=4)
    light = tuple(min(255, c + 35) for c in C_WALL)
    dark  = tuple(max(0, c - 35) for c in C_WALL)
    pygame.draw.rect(surf, light, (rx+2, ry+2, rw-4, rh//2-2), border_radius=2)
    mid = ry + rh // 2
    pygame.draw.line(surf, dark, (rx, mid), (rx+rw, mid), 1)
    pygame.draw.line(surf, dark, (rx+rw//2, mid), (rx+rw//2, ry+rh), 1)
    pygame.draw.line(surf, dark, (rx+rw//4, ry), (rx+rw//4, mid), 1)
    pygame.draw.line(surf, dark, (rx+rw*3//4, ry), (rx+rw*3//4, mid), 1)


def render_grid(surf, env, gx, gy, gw, gh, path, shot_ray):
    cw = gw / env.size
    ch = gh / env.size
    pygame.draw.rect(surf, GRID_BG, (gx, gy, gw, gh), border_radius=10)

    for ox, oy in env.obstacles:
        draw_wall(surf,
                  int(gx + ox*cw + cw*0.05), int(gy + oy*ch + ch*0.05),
                  int(cw*0.9), int(ch*0.9))

    # Путь
    if len(path) > 1:
        for i in range(len(path)-1):
            alpha = int(50 + 205 * i / max(len(path)-1, 1))
            x1 = int(gx + path[i][0]*cw + cw/2)
            y1 = int(gy + path[i][1]*ch + ch/2)
            x2 = int(gx + path[i+1][0]*cw + cw/2)
            y2 = int(gy + path[i+1][1]*ch + ch/2)
            w = abs(x2-x1)+4; h = abs(y2-y1)+4
            s = pygame.Surface((max(w,1), max(h,1)), pygame.SRCALPHA)
            ox_ = min(x1,x2)-2; oy_ = min(y1,y2)-2
            pygame.draw.line(s, (*C_PATH, alpha), (x1-ox_, y1-oy_), (x2-ox_, y2-oy_), 2)
            surf.blit(s, (ox_, oy_))

    # Луч выстрела
    if shot_ray:
        origin, ray_cells = shot_ray
        if ray_cells:
            ox_ = int(gx + origin[0]*cw + cw/2)
            oy_ = int(gy + origin[1]*ch + ch/2)
            ex = int(gx + ray_cells[-1][0]*cw + cw/2)
            ey = int(gy + ray_cells[-1][1]*ch + ch/2)
            pygame.draw.line(surf, C_SHOT, (ox_, oy_), (ex, ey), 3)
            pygame.draw.circle(surf, C_SHOT, (ex, ey), 5)

    # Цель — флаг со свечением
    gcx = int(gx + env.goal_pos[0]*cw + cw/2)
    gcy = int(gy + env.goal_pos[1]*ch + ch/2)
    r = int(min(cw, ch) * 0.38)
    gs = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
    pygame.draw.circle(gs, (*C_GOAL, 20), (r*2, r*2), r*2)
    surf.blit(gs, (gcx-r*2, gcy-r*2))
    draw_flag(surf, gcx, gcy, r)

    # Танк со свечением
    tcx = int(gx + env.tank_pos[0]*cw + cw/2)
    tcy = int(gy + env.tank_pos[1]*ch + ch/2)
    tr = int(min(cw, ch) * 0.34)
    gs = pygame.Surface((tr*5, tr*5), pygame.SRCALPHA)
    pygame.draw.circle(gs, (*C_TANK, 25), (tr*2+tr//2, tr*2+tr//2), tr*2)
    surf.blit(gs, (tcx-tr*2, tcy-tr*2))
    draw_tank(surf, tcx, tcy, env.tank_angle, tr)

    # Прицел: если цель на линии огня — линия к ней
    if env._goal_in_sight():
        dx, dy = env.DIRS[env.tank_angle]
        ex = int(gx + env.goal_pos[0]*cw + cw/2)
        ey = int(gy + env.goal_pos[1]*ch + ch/2)
        s = pygame.Surface((abs(ex-tcx)+4, abs(ey-tcy)+4), pygame.SRCALPHA)
        pygame.draw.line(s, (*C_SHOT, 80),
                         (tcx-min(tcx,ex)+2, tcy-min(tcy,ey)+2),
                         (ex-min(tcx,ex)+2, ey-min(tcy,ey)+2), 2)
        surf.blit(s, (min(tcx,ex)-2, min(tcy,ey)-2))

    # Сетка
    for r in range(env.size + 1):
        yy = int(gy + r * ch)
        pygame.draw.line(surf, GRID_LINE, (gx, yy), (gx+gw, yy))
    for c in range(env.size + 1):
        xx = int(gx + c * cw)
        pygame.draw.line(surf, GRID_LINE, (xx, gy), (xx, gy+gh))
    pygame.draw.rect(surf, PANEL_EDGE, (gx, gy, gw, gh), 2, border_radius=10)


def render_chart(surf, fxs, history, x, y, w, h, title, color):
    pygame.draw.rect(surf, GRID_BG, (x, y, w, h), border_radius=6)
    pygame.draw.rect(surf, PANEL_EDGE, (x, y, w, h), 1, border_radius=6)
    txt(surf, fxs, title, x+8, y+5, C_SUB)
    px, py_ = x+6, y+20
    pw, ph = w-12, h-26
    vals = [v for v in history if v is not None]
    if len(vals) < 2:
        return
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    pts = []
    for i, v in enumerate(history):
        if v is None: continue
        sx = px + int(i * pw / max(len(history)-1, 1))
        sy = py_ + ph - int((v - mn) / rng * ph)
        pts.append((sx, max(py_, min(py_+ph, sy))))
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, 2)


def render_panel(surf, fonts, px, py, pw, ph, stats, reward_hist, success_hist, steps_hist, mode_label):
    font, fsm, fxs = fonts
    pygame.draw.rect(surf, PANEL_BG, (px, py, pw, ph), border_radius=10)
    pygame.draw.rect(surf, PANEL_EDGE, (px, py, pw, ph), 1, border_radius=10)

    cy = py + 14
    txt(surf, font, "Tank RL Training", px+12, cy)
    cy += 28
    txt(surf, fxs, f"Режим: {mode_label}", px+12, cy, C_OK if "fixed" in mode_label.lower() else C_TANK)
    cy += 20

    txt(surf, fsm, f"Episode:  {stats['ep']}", px+12, cy)
    cy += 20
    rc = lerp(C_FAIL, C_OK, min(1.0, (stats['reward'] + 5) / 15))
    txt(surf, fsm, f"Reward:   {stats['reward']:.1f}", px+12, cy, rc)
    cy += 20
    txt(surf, fsm, f"Steps:    {stats['steps']}/{stats['max_steps']}", px+12, cy, C_SUB)
    cy += 20
    txt(surf, fsm, f"Total steps: {stats['total_steps']}", px+12, cy, C_SUB)
    cy += 20

    eps = stats['epsilon']
    txt(surf, fxs, f"Explore: {eps:.1%}", px+12, cy, C_SUB)
    bw = pw - 24
    pygame.draw.rect(surf, GRID_LINE, (px+12, cy+14, bw, 5), border_radius=2)
    fw = int(bw * eps)
    if fw > 0:
        pygame.draw.rect(surf, lerp(C_OK, C_TANK, eps), (px+12, cy+14, fw, 5), border_radius=2)
    cy += 30

    if success_hist:
        recent = list(success_hist)[-50:]
        sr = np.mean(recent)
        txt(surf, fsm, f"Success:  {sr:.0%}", px+12, cy, lerp(C_FAIL, C_OK, sr))
    cy += 28

    cw_c = pw - 16
    render_chart(surf, fxs, list(reward_hist),  px+8, cy, cw_c, 72, "Reward / episode", C_TANK)
    cy += 82
    render_chart(surf, fxs, list(success_hist), px+8, cy, cw_c, 72, "Success rate (last 100)", C_OK)
    cy += 82
    render_chart(surf, fxs, list(steps_hist),   px+8, cy, cw_c, 72, "Steps / episode (меньше = лучше)", C_PATH)
    cy += 82

    acts = ["↑ Вперёд", "↓ Назад", "← Лево", "→ Право", "💥 Выстрел"]
    la = stats.get('last_action')
    if la is not None:
        txt(surf, fxs, f"Action: {acts[la]}", px+12, cy, C_SHOT if la == 4 else C_TEXT)
    cy += 18

    txt(surf, fxs, "ESC — выход", px+12, py+ph-16, C_SUB)


def live_train(fixed_map: bool = False, max_episodes: int = 500):
    pygame.init()
    pygame.display.set_caption("Tank RL")

    CELL  = 80
    PANEL = 270
    PAD   = 14

    config = TankConfig(grid_size=6, max_steps=50, num_obstacles=4, fixed_map=fixed_map)
    env = TankEnv(config)

    grid_w = env.size * CELL
    grid_h = env.size * CELL
    win_w  = grid_w + PANEL + PAD * 3
    win_h  = max(grid_h + PAD * 2, 760)

    screen = pygame.display.set_mode((win_w, win_h))
    clock  = pygame.time.Clock()

    try:
        font   = pygame.font.SysFont("segoeui", 16, bold=True)
        font_sm = pygame.font.SysFont("segoeui", 13)
        font_xs = pygame.font.SysFont("segoeui", 11)
    except Exception:
        font   = pygame.font.SysFont(None, 18, bold=True)
        font_sm = pygame.font.SysFont(None, 15)
        font_xs = pygame.font.SysFont(None, 13)

    # Гиперпараметры прямо из рабочего примера NandaKishore/PyTorch
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        lr=1e-3,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=3000,
        buffer_size=1000,
        batch_size=200,
        sync_freq=500,
    )

    reward_hist  = deque(maxlen=100)
    success_hist = deque(maxlen=100)
    steps_hist   = deque(maxlen=100)

    episode    = 0
    state      = env.reset()
    path       = [env.tank_pos]
    total_rew  = 0.0
    last_action = None

    mode_label = "Фиксированная" if fixed_map else "Случайная карта"

    while episode < max_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); return

        if not env.done:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            path.append(env.tank_pos)
            total_rew += reward
            last_action = action
        else:
            episode += 1
            # dist==0 ловит движение к цели; last_shot_ray ловит победный выстрел
            success = info['dist'] == 0 or (env.last_shot_ray is not None and info['dist'] > 0 and env.done)
            reward_hist.append(total_rew)
            success_hist.append(1.0 if success else 0.0)
            steps_hist.append(float(env.steps))

            if episode % 50 == 0:
                suffix = "fixed" if fixed_map else "random"
                agent.save(f"model_{suffix}_ep{episode}.pth")
                sr = np.mean(list(success_hist)[-50:])
                print(f"Ep {episode:4d} | SR: {sr:.1%} | ε={agent.epsilon:.3f} | steps={agent.total_steps}")

            state = env.reset()
            path = [env.tank_pos]
            total_rew = 0.0
            last_action = None

        # Рендер
        screen.fill(BG)
        render_grid(screen, env, PAD, PAD, grid_w, grid_h, path, env.last_shot_ray)

        stats = {
            "ep": episode, "reward": total_rew,
            "steps": env.steps, "max_steps": config.max_steps,
            "epsilon": agent.epsilon, "total_steps": agent.total_steps,
            "last_action": last_action,
        }
        render_panel(screen, (font, font_sm, font_xs),
                     PAD + grid_w + PAD, PAD, PANEL, win_h - PAD*2,
                     stats, reward_hist, success_hist, steps_hist, mode_label)

        pygame.display.flip()
        clock.tick(60)

    suffix = "fixed" if fixed_map else "random"
    agent.save(f"model_{suffix}_final.pth")
    sr = np.mean(list(success_hist)[-50:])
    print(f"\nДобучение завершено. SR={sr:.0%}")
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    live_train(fixed_map=args.fixed, max_episodes=args.episodes)
