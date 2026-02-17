"""
Pygame визуализация экосистемы.
Левая часть  — сетка мира с анимацией.
Правая часть — графики популяций + статистика в реальном времени.
"""

import pygame
import sys
import math
import time
from collections import deque
from world import World
from animals import Pauvre, Malheureux
from plants import Lumiere, Obscurite, Demi

# ── Цветовая палитра ──────────────────────────────────────────────────────────

BG           = (10, 11, 16)
GRID_BG      = (14, 16, 22)
GRID_LINE    = (22, 26, 36)
PANEL_BG     = (16, 18, 28)
PANEL_EDGE   = (32, 38, 60)

C_LUMIERE    = (80,  200, 100)
C_OBSCURITE  = (80,  120, 220)
C_DEMI       = (200, 170,  60)
C_PAUVRE     = (210, 210, 210)
C_MALHEUREUX = (220,  70,  70)

C_TEXT       = (180, 185, 210)
C_SUBTEXT    = ( 90,  96, 130)
C_ACCENT     = (100, 130, 255)

PHASE_COLORS = {
    "MORNING": (220, 160,  60),
    "DAY":     (240, 220, 100),
    "EVENING": (180, 100,  60),
    "NIGHT":   ( 60,  80, 160),
}

HISTORY_LEN = 200

CELL_COLORS = {
    Lumiere:    C_LUMIERE,
    Obscurite:  C_OBSCURITE,
    Demi:       C_DEMI,
    Pauvre:     C_PAUVRE,
    Malheureux: C_MALHEUREUX,
}


# ── Утилиты ───────────────────────────────────────────────────────────────────

def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_rounded_rect_alpha(surf, color, rect, r=6, alpha=60):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=r)
    surf.blit(s, (rect[0], rect[1]))


def draw_text(surf, font, text, x, y, color=C_TEXT):
    surf.blit(font.render(text, True, color), (x, y))


# ── Сетка ─────────────────────────────────────────────────────────────────────

def render_grid(surf, world, gx, gy, gw, gh, tick_count):
    rows, cols = world.rows, world.cols
    cw = gw / cols
    ch = gh / rows

    pygame.draw.rect(surf, GRID_BG, (gx, gy, gw, gh), border_radius=8)

    for r in range(rows):
        for c in range(cols):
            cell   = world.grid[r][c]
            animal = cell["animal"]
            plant  = cell["plant"]
            obj    = animal or plant
            if obj is None:
                continue

            color = CELL_COLORS.get(type(obj), C_TEXT)
            cx    = gx + c * cw + cw * 0.1
            cy    = gy + r * ch + ch * 0.1
            ew    = cw * 0.8
            eh    = ch * 0.8

            if animal:
                radius = min(ew, eh) / 2
                pulse  = 1.0 + 0.12 * math.sin(tick_count * 0.15 + r * 0.7 + c * 0.5)
                pr     = max(2, int(radius * pulse))
                center = (int(cx + ew / 2), int(cy + eh / 2))

                # Свечение
                gs = pygame.Surface((pr * 4, pr * 4), pygame.SRCALPHA)
                pygame.draw.circle(gs, (*color, 30), (pr * 2, pr * 2), pr * 2)
                surf.blit(gs, (center[0] - pr * 2, center[1] - pr * 2))

                pygame.draw.circle(surf, color, center, pr)
                hunger_t = min(1.0, animal.hunger / animal.max_hunger)
                inner_c  = lerp_color(color, (255, 60, 60), hunger_t)
                pygame.draw.circle(surf, inner_c, center, max(1, pr // 3))
            else:
                cx2 = int(cx + ew / 2)
                cy2 = int(cy + eh / 2)
                r2  = max(2, int(min(ew, eh) / 2))
                pts = [(cx2, cy2 - r2), (cx2 + r2, cy2),
                       (cx2, cy2 + r2), (cx2 - r2, cy2)]
                pygame.draw.polygon(surf, lerp_color(color, BG, 0.4), pts)
                pygame.draw.polygon(surf, color, pts, 1)

    for r in range(rows + 1):
        y = int(gy + r * ch)
        pygame.draw.line(surf, GRID_LINE, (gx, y), (gx + gw, y))
    for c in range(cols + 1):
        x = int(gx + c * cw)
        pygame.draw.line(surf, GRID_LINE, (x, gy), (x, gy + gh))

    pygame.draw.rect(surf, PANEL_EDGE, (gx, gy, gw, gh), 1, border_radius=8)


# ── График ────────────────────────────────────────────────────────────────────

def render_chart(surf, font_xs, histories, labels, colors, x, y, w, h, title):
    pygame.draw.rect(surf, PANEL_BG, (x, y, w, h), border_radius=6)
    pygame.draw.rect(surf, PANEL_EDGE, (x, y, w, h), 1, border_radius=6)

    draw_text(surf, font_xs, title, x + 10, y + 6, C_SUBTEXT)

    px, py = x + 32, y + 22
    pw, ph = w - 40, h - 34

    all_vals = [v for s in histories for v in s]
    max_val  = max(all_vals) if all_vals and max(all_vals) > 0 else 1

    for i in range(5):
        ly = py + ph - int(ph * i / 4)
        pygame.draw.line(surf, GRID_LINE, (px, ly), (px + pw, ly))
        draw_text(surf, font_xs, str(int(max_val * i / 4)), x + 2, ly - 6, C_SUBTEXT)

    n = len(histories[0]) if histories else 0
    if n < 2:
        return

    for series, color in zip(histories, colors):
        pts = []
        for i, v in enumerate(series):
            sx = px + int(i * pw / (HISTORY_LEN - 1))
            sy = py + ph - int(v / max_val * ph)
            pts.append((sx, sy))

        if len(pts) >= 2:
            fill_pts = [(px, py + ph)] + pts + [(pts[-1][0], py + ph)]
            fs = pygame.Surface((pw + 1, ph + 1), pygame.SRCALPHA)
            local = [(p[0] - px, p[1] - py) for p in fill_pts]
            pygame.draw.polygon(fs, (*color, 28), local)
            surf.blit(fs, (px, py))
            pygame.draw.lines(surf, color, False, pts, 2)

    # Легенда
    lx = px
    for label, color in zip(labels, colors):
        pygame.draw.rect(surf, color, (lx, y + h - 14, 8, 8), border_radius=2)
        draw_text(surf, font_xs, label, lx + 11, y + h - 16, C_TEXT)
        lx += 64


# ── Боковая панель ────────────────────────────────────────────────────────────

def render_panel(surf, world, fonts, px, py, pw, ph,
                 plant_hist, animal_hist, tick_num, fps):
    font, font_sm, font_xs = fonts

    pygame.draw.rect(surf, PANEL_BG, (px, py, pw, ph), border_radius=8)
    pygame.draw.rect(surf, PANEL_EDGE, (px, py, pw, ph), 1, border_radius=8)

    s          = world.stats()
    phase_name = world.clock.time.name
    phase_col  = PHASE_COLORS.get(phase_name, C_ACCENT)

    cy = py + 14

    # ── Фаза суток ────────────────────────────────────────────────────────────
    draw_rounded_rect_alpha(surf, phase_col, (px + 8, cy, pw - 16, 34), r=6, alpha=38)
    pygame.draw.rect(surf, phase_col, (px + 8, cy, 3, 34), border_radius=2)
    phase_label = {"MORNING": "Утро", "DAY": "День",
                   "EVENING": "Вечер", "NIGHT": "Ночь"}.get(phase_name, phase_name)
    draw_text(surf, font, phase_label, px + 18, cy + 8, phase_col)
    draw_text(surf, font_xs, f"свет: {world.clock.light}", px + pw - 74, cy + 10, C_SUBTEXT)
    cy += 44

    # ── Тик / FPS ─────────────────────────────────────────────────────────────
    draw_text(surf, font_sm, f"Тик: {tick_num}", px + 12, cy, C_SUBTEXT)
    draw_text(surf, font_sm, f"FPS: {fps:.0f}", px + pw - 58, cy, C_SUBTEXT)
    cy += 20

    # ── Полоски популяций ─────────────────────────────────────────────────────
    entries = [
        ("L  Lumiere",    s["L"], C_LUMIERE),
        ("O  Obscurite",  s["O"], C_OBSCURITE),
        ("D  Demi",       s["D"], C_DEMI),
        ("P  Pauvre",     s["P"], C_PAUVRE),
        ("M  Malheureux", s["M"], C_MALHEUREUX),
    ]
    max_count = max((e[1] for e in entries), default=1) or 1

    for label, count, color in entries:
        draw_text(surf, font_sm, label, px + 10, cy + 2, color)
        bx = px + 140
        bw = pw - 168
        pygame.draw.rect(surf, GRID_LINE, (bx, cy + 5, bw, 10), border_radius=4)
        fill = int(bw * count / max_count)
        if fill > 0:
            pygame.draw.rect(surf, color, (bx, cy + 5, fill, 10), border_radius=4)
        draw_text(surf, font_sm, str(count), px + pw - 26, cy + 2, C_TEXT)
        cy += 21

    cy += 8

    # ── Графики ───────────────────────────────────────────────────────────────
    chart_h = 114
    avail   = (py + ph) - cy - 10

    render_chart(surf, font_xs,
                 [list(plant_hist[k]) for k in ("L", "O", "D")],
                 ["Lumiere", "Obscurite", "Demi"],
                 [C_LUMIERE, C_OBSCURITE, C_DEMI],
                 px + 6, cy, pw - 12, chart_h, "Растения")
    cy += chart_h + 8

    render_chart(surf, font_xs,
                 [list(animal_hist[k]) for k in ("P", "M")],
                 ["Pauvre", "Malheureux"],
                 [C_PAUVRE, C_MALHEUREUX],
                 px + 6, cy, pw - 12, chart_h, "Животные")
    cy += chart_h + 10

    # ── Средний голод ─────────────────────────────────────────────────────────
    if cy + 60 < py + ph:
        draw_text(surf, font_sm, "Средний голод", px + 10, cy, C_SUBTEXT)
        cy += 18
        all_a   = world.animals
        pauvres = [a for a in all_a if isinstance(a, Pauvre)]
        malhs   = [a for a in all_a if isinstance(a, Malheureux)]

        for label, group, color, max_h in [
            ("Pauvre",     pauvres, C_PAUVRE,     8),
            ("Malheureux", malhs,   C_MALHEUREUX, 12),
        ]:
            avg = sum(a.hunger for a in group) / len(group) if group else 0
            draw_text(surf, font_sm, label, px + 10, cy + 2, color)
            bx  = px + 108
            bw  = pw - 136
            pygame.draw.rect(surf, GRID_LINE, (bx, cy + 5, bw, 10), border_radius=4)
            t     = min(1.0, avg / max_h)
            bar_c = lerp_color(color, (230, 55, 55), t)
            fill  = int(bw * t)
            if fill > 0:
                pygame.draw.rect(surf, bar_c, (bx, cy + 5, fill, 10), border_radius=4)
            draw_text(surf, font_sm, f"{avg:.1f}", px + pw - 32, cy + 2, C_TEXT)
            cy += 20

    # ── Подсказка управления ──────────────────────────────────────────────────
    bottom = py + ph - 18
    draw_text(surf, font_xs, "SPACE — пауза/продолжить    ESC — выход",
              px + 10, bottom, C_SUBTEXT)


# ── Запуск ────────────────────────────────────────────────────────────────────

def run(rows=20, cols=40, ticks=10000, ticks_per_phase=3,
        plant_density=0.3, pauvre_count=10, malheureux_count=5,
        delay=0.12):

    pygame.init()
    pygame.display.set_caption("Ecosystem Simulation")

    CELL  = 22
    PANEL = 280
    PAD   = 12

    grid_w = cols * CELL
    grid_h = rows * CELL
    win_w  = grid_w + PANEL + PAD * 3
    win_h  = max(grid_h + PAD * 2, 700)

    screen   = pygame.display.set_mode((win_w, win_h))
    pg_clock = pygame.time.Clock()

    try:
        font    = pygame.font.SysFont("segoeui", 16, bold=True)
        font_sm = pygame.font.SysFont("segoeui", 13)
        font_xs = pygame.font.SysFont("segoeui", 11)
    except Exception:
        font    = pygame.font.SysFont(None, 18, bold=True)
        font_sm = pygame.font.SysFont(None, 15)
        font_xs = pygame.font.SysFont(None, 13)

    world = World(
        rows=rows, cols=cols,
        plant_density=plant_density,
        pauvre_count=pauvre_count,
        malheureux_count=malheureux_count,
        ticks_per_phase=ticks_per_phase,
    )

    plant_hist  = {k: deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN) for k in ("L", "O", "D")}
    animal_hist = {k: deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN) for k in ("P", "M")}

    tick_num  = 0
    last_tick = time.time()
    paused    = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused

        now = time.time()
        if not paused and now - last_tick >= delay:
            world.tick()
            tick_num += 1
            last_tick = now
            s = world.stats()
            for k in plant_hist:
                plant_hist[k].append(s[k])
            for k in animal_hist:
                animal_hist[k].append(s[k])

        # ── Рендер ───────────────────────────────────────────────────────────
        screen.fill(BG)

        render_grid(screen, world, PAD, PAD, grid_w, grid_h, tick_num)

        px = PAD + grid_w + PAD
        render_panel(screen, world, (font, font_sm, font_xs),
                     px, PAD, PANEL, win_h - PAD * 2,
                     plant_hist, animal_hist, tick_num,
                     pg_clock.get_fps())

        if paused:
            ps = font.render("⏸  ПАУЗА", True, C_ACCENT)
            screen.blit(ps, (PAD + grid_w // 2 - ps.get_width() // 2,
                              PAD + grid_h // 2 - ps.get_height() // 2))

        pygame.display.flip()
        pg_clock.tick(60)

    pygame.quit()