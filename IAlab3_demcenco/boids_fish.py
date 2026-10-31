# -*- coding: utf-8 -*-
"""
Лабораторная работа №3. Вариант 2.
Симуляция косяка рыб с учётом препятствий: Boids Крейга Рейнольдса + правило избегания препятствий.

Запуск:
    python boids_fish.py                 # интерактивная визуализация (ползунки, клик мышью)
    python boids_fish.py --gif demo.gif  # сохранить анимацию в GIF без окна

Все параметры собраны в трёх местах и легко меняются:
    Env            - параметры среды (размер мира, препятствия)
    AgentParams    - параметры агентов (число, скорости, радиусы)
    DEFAULT_WEIGHTS - веса правил поведения
Чтобы добавить новое правило: написать функцию rule_xxx(sim) -> массив ускорений (N, 2),
добавить её в RULES и задать вес в словаре весов.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

# 1. Параметры
@dataclass
class Env:
    """Параметры среды."""
    width: float = 800.0
    height: float = 600.0
    wall_margin: float = 50.0                       # зона мягкого отталкивания от стен
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # строки (x, y, r)


@dataclass
class AgentParams:
    """Параметры агентов."""
    n: int = 60                # число рыб
    max_speed: float = 3.5     # максимальная скорость (обязательное ограничение)
    min_speed: float = 1.2     # минимальная скорость (рыба не зависает на месте)
    cruise_speed: float = 2.4  # комфортная скорость, к которой рыба стремится
    max_accel: float = 0.35    # максимальное ускорение
    r_interact: float = 70.0   # радиус взаимодействия (соседи для cohesion / alignment)
    r_sep: float = 22.0        # радиус разделения (слишком близкие соседи)
    r_obstacle: float = 90.0   # радиус обнаружения препятствий (от поверхности)
    body_radius: float = 2.5   # "размер тела" рыбы: минимально допустимый зазор до препятствия
    r_group: float = 45.0      # радиус связности для подсчёта групп (только для метрики, на поведение не влияет)


# Веса правил: итоговое ускорение a = sum(w_i * a_i)
DEFAULT_WEIGHTS = {
    "cohesion": 0.5,     # центрирование
    "alignment": 1.0,    # выравнивание
    "separation": 1.5,   # разделение
    "obstacle": 3.0,     # избегание препятствий
    "wall": 2.0,         # отталкивание от границ мира
    "cruise": 0.6,       # поддержание крейсерской скорости
}

# 2. Вспомогательные функции

def norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=-1)


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)[..., None]
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)


def limit(v: np.ndarray, max_len: float) -> np.ndarray:
    """Ограничить длину векторов значением max_len."""
    n = norm(v)[..., None]
    scale = np.minimum(1.0, np.divide(max_len, n, out=np.ones_like(n), where=n > 1e-9))
    return v * scale


def make_obstacles(count: int, r_range=(15.0, 35.0), env: Env | None = None, seed: int = 0,
                   keep_free_x: float = 260.0, gap: float = 45.0) -> np.ndarray:
    """Случайные непересекающиеся круглые препятствия (камни). Левая часть мира остаётся свободной (старт косяка)."""
    env = env or Env()
    rng = np.random.default_rng(seed)
    obs: list[tuple[float, float, float]] = []
    tries = 0
    while len(obs) < count and tries < 20000:
        tries += 1
        r = rng.uniform(*r_range)
        x = rng.uniform(keep_free_x + r, env.width - r - 60)
        y = rng.uniform(r + 60, env.height - r - 60)
        if all(np.hypot(x - ox, y - oy) > r + orr + gap for ox, oy, orr in obs):
            obs.append((x, y, r))
    return np.array(obs, dtype=float).reshape(-1, 3)


# 3. Правила поведения (каждое возвращает ускорения формы (N, 2))

def rule_cohesion(s: "Simulation") -> np.ndarray:
    """Центрирование: двигаться к центру масс соседей."""
    mask = s.dist < s.ap.r_interact
    cnt = mask.sum(axis=1)
    center = (mask @ s.pos) / np.maximum(cnt, 1)[:, None]
    a = s.steer_toward(center - s.pos)
    a[cnt == 0] = 0.0
    return a


def rule_alignment(s: "Simulation") -> np.ndarray:
    """Выравнивание: подстроить скорость под среднюю скорость соседей."""
    mask = s.dist < s.ap.r_interact
    cnt = mask.sum(axis=1)
    mean_v = (mask @ s.vel) / np.maximum(cnt, 1)[:, None]
    a = limit(mean_v - s.vel, s.ap.max_accel)
    a[cnt == 0] = 0.0
    return a


def rule_separation(s: "Simulation") -> np.ndarray:
    """Разделение: отойти от слишком близких соседей (чем ближе, тем сильнее)."""
    close = s.dist < s.ap.r_sep
    diff = s.pos[:, None, :] - s.pos[None, :, :]                 # от соседа j к агенту i
    w = np.where(close, 1.0 / np.maximum(s.dist, 1e-6) ** 2, 0.0)
    away = (diff * w[..., None]).sum(axis=1)
    a = s.steer_toward(away)
    a[norm(away) < 1e-12] = 0.0
    return a


def rule_obstacle(s: "Simulation") -> np.ndarray:
    """
    Избегание препятствий. Для каждого препятствия в радиусе обнаружения:
      - сила отталкивания от поверхности растёт при приближении;
      - если рыба летит "в лоб", добавляется боковая составляющая в сторону, куда рыба уже
        движется, поэтому рыба огибает камень, а не отскакивает назад.
    """
    obs = s.env.obstacles
    if len(obs) == 0:
        return np.zeros_like(s.pos)
    centers, radii = obs[:, :2], obs[:, 2]
    delta = s.pos[:, None, :] - centers[None, :, :]               # (N, M, 2)
    d = norm(delta)                                               # расстояние до центра
    surf = d - radii[None, :]                                     # расстояние до поверхности
    n = delta / np.maximum(d, 1e-9)[..., None]                    # направление от препятствия
    strength = np.clip((s.ap.r_obstacle - surf) / s.ap.r_obstacle, 0.0, 1.0) ** 2

    vhat = unit(s.vel)[:, None, :]
    head_on = np.clip(-(n * vhat).sum(axis=2), 0.0, 1.0)          # 1 - летит прямо в препятствие
    tang = np.stack([-n[..., 1], n[..., 0]], axis=2)              # перпендикуляр
    sign = np.sign((tang * vhat).sum(axis=2))
    sign[sign == 0] = 1.0
    tang = tang * sign[..., None]                                 # в сторону текущего движения
    direction = unit(n + 1.5 * head_on[..., None] * tang)
    return (strength[..., None] * direction).sum(axis=1) * s.ap.max_accel


def rule_wall(s: "Simulation") -> np.ndarray:
    """Мягкое отталкивание от границ мира."""
    m = s.env.wall_margin
    x, y = s.pos[:, 0], s.pos[:, 1]
    ax = np.clip((m - x) / m, 0, 1) - np.clip((x - (s.env.width - m)) / m, 0, 1)
    ay = np.clip((m - y) / m, 0, 1) - np.clip((y - (s.env.height - m)) / m, 0, 1)
    return np.stack([ax, ay], axis=1) * s.ap.max_accel


def rule_cruise(s: "Simulation") -> np.ndarray:
    """Поддержание крейсерской скорости: разгон/торможение вдоль текущего курса."""
    sp = norm(s.vel)[:, None]
    return limit(unit(s.vel) * (s.ap.cruise_speed - sp), s.ap.max_accel)


RULES = {
    "cohesion": rule_cohesion,
    "alignment": rule_alignment,
    "separation": rule_separation,
    "obstacle": rule_obstacle,
    "wall": rule_wall,
    "cruise": rule_cruise,
}

# 4. Симуляция


class Simulation:
    def __init__(self, env: Env, ap: AgentParams, weights: dict | None = None, seed: int = 0):
        self.env, self.ap = env, ap
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.seed = seed
        self.reset()

    # ----- инициализация ---------------------------------------------------------------
    def reset(self):
        rng = np.random.default_rng(self.seed + 1000)
        self.rng = rng
        n, ap, env = self.ap.n, self.ap, self.env
        pos = []
        while len(pos) < n:                       # старт: компактная группа в левой части, вне препятствий
            p = np.array([rng.uniform(40, 220), rng.uniform(env.height * 0.25, env.height * 0.75)])
            if self._free(p):
                pos.append(p)
        self.pos = np.array(pos)
        ang = rng.normal(0.0, 0.5, n)             # в основном вправо
        speed = rng.uniform(ap.min_speed, ap.max_speed, n)
        self.vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * speed[:, None]
        self.corrections = 0                      # сколько раз пришлось "вытолкнуть" рыбу из препятствия
        self.t = 0
        self._update_dist()

    def _free(self, p) -> bool:
        obs = self.env.obstacles
        if len(obs) == 0:
            return True
        return bool(np.all(np.hypot(p[0] - obs[:, 0], p[1] - obs[:, 1]) > obs[:, 2] + 20))

    def _update_dist(self):
        d = norm(self.pos[:, None, :] - self.pos[None, :, :])
        np.fill_diagonal(d, np.inf)
        self.dist = d

    def steer_toward(self, direction: np.ndarray) -> np.ndarray:
        """Рулевое ускорение к желаемому направлению: (желаемая скорость - текущая), с ограничением."""
        desired = unit(direction) * self.ap.max_speed
        return limit(desired - self.vel, self.ap.max_accel)

    # ----- шаг ----------------------------------------------------------------------------
    def step(self):
        ap = self.ap
        self._update_dist()
        a = np.zeros_like(self.pos)
        for name, rule in RULES.items():
            w = self.w.get(name, 0.0)
            if w != 0.0:
                a += w * rule(self)
        a = limit(a, ap.max_accel)                                  # ограничение ускорения
        v = limit(self.vel + a, ap.max_speed)                       # v <- ограничить(v + a, v_max)
        sp = norm(v)
        slow = sp < ap.min_speed
        if np.any(slow):                                            # не даём рыбе остановиться
            dirs = np.where(sp[:, None] > 1e-9, unit(v), unit(self.vel))
            v[slow] = dirs[slow] * ap.min_speed
        self.vel = v
        self.pos = self.pos + self.vel                              # p <- p + v
        self._hard_constraints()
        self.t += 1

    def _hard_constraints(self):
        """Жёсткие ограничения: рыба не может оказаться внутри препятствия и вне мира."""
        env, ap = self.env, self.ap
        # границы мира: отражение
        for k, size in ((0, env.width), (1, env.height)):
            lo = self.pos[:, k] < 0
            hi = self.pos[:, k] > size
            self.pos[lo, k] = 0.0
            self.pos[hi, k] = size
            self.vel[lo | hi, k] *= -1.0
        # препятствия: проекция на поверхность + удаление скорости, направленной внутрь
        for cx, cy, r in env.obstacles:
            delta = self.pos - np.array([cx, cy])
            d = norm(delta)
            inside = d < r + ap.body_radius
            if np.any(inside):
                self.corrections += int(inside.sum())
                nrm = delta[inside] / np.maximum(d[inside], 1e-9)[:, None]
                nrm[d[inside] < 1e-9] = [1.0, 0.0]
                self.pos[inside] = np.array([cx, cy]) + nrm * (r + ap.body_radius)
                vn = (self.vel[inside] * nrm).sum(axis=1)
                self.vel[inside] -= np.minimum(vn, 0.0)[:, None] * nrm

    # ----- метрики ---------------------------------------------------------------------------
    def polarization(self) -> float:
        """Степень согласованности движения: 1 - все плывут в одну сторону, 0 - хаос."""
        return float(norm(unit(self.vel).mean(axis=0)))

    def count_groups(self, link: float | None = None) -> int:
        """Число групп: компоненты связности графа "расстояние < link" (по умолчанию link = r_group)."""
        link = self.ap.r_group if link is None else link
        n = self.ap.n
        d = norm(self.pos[:, None, :] - self.pos[None, :, :])
        adj = d < link
        seen = np.zeros(n, dtype=bool)
        groups = 0
        for i in range(n):
            if seen[i]:
                continue
            groups += 1
            stack = [i]
            seen[i] = True
            while stack:
                u = stack.pop()
                for v in np.nonzero(adj[u] & ~seen)[0]:
                    seen[v] = True
                    stack.append(v)
        return groups

    def min_clearance(self) -> float:
        """Минимальный зазор между рыбами и поверхностью препятствий (должен быть >= 0)."""
        obs = self.env.obstacles
        if len(obs) == 0:
            return float("inf")
        d = norm(self.pos[:, None, :] - obs[None, :, :2]) - obs[None, :, 2]
        return float(d.min())


# 5. Визуализация (Matplotlib)


def _setup_axes(ax, env: Env):
    from matplotlib.patches import Circle
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#dff1fb")
    ax.set_xticks([])
    ax.set_yticks([])
    patches = []
    for x, y, r in env.obstacles:
        c = Circle((x, y), r, fc="#8a7f73", ec="#4b443d", lw=1.5, zorder=2)
        ax.add_patch(c)
        patches.append(c)
    return patches


def interactive(env: Env, ap: AgentParams, weights: dict, seed: int = 0):
    """Интерактивное окно: ползунки весов и радиуса, клик - добавить камень, правый клик - убрать."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle
    from matplotlib.widgets import Button, Slider

    sim = Simulation(env, ap, weights, seed)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([0.04, 0.30, 0.92, 0.66])
    _setup_axes(ax, env)
    ax.set_title("Косяк рыб (Boids + избегание препятствий)   |   клик: добавить камень, правый клик: убрать")

    trail_len = 18
    trail = np.repeat(sim.pos[None, :, :], trail_len, axis=0)
    lc = LineCollection([], colors="#1f77b4", linewidths=0.8, alpha=0.5, zorder=3)
    ax.add_collection(lc)
    quiv = ax.quiver(sim.pos[:, 0], sim.pos[:, 1], sim.vel[:, 0], sim.vel[:, 1], angles="xy",
                     scale_units="xy", scale=0.35, width=0.004, color="#d9480f", zorder=4)
    info = ax.text(8, env.height - 22, "", fontsize=10, zorder=5)
    patches = list(ax.patches)

    # ползунки
    sliders = {}
    specs = [("cohesion", 0.0, 3.0), ("alignment", 0.0, 3.0), ("separation", 0.0, 4.0),
             ("obstacle", 0.0, 8.0)]
    for i, (name, lo, hi) in enumerate(specs):
        sax = fig.add_axes([0.14, 0.22 - i * 0.045, 0.30, 0.025])
        sliders[name] = Slider(sax, name, lo, hi, valinit=sim.w[name])
        sliders[name].on_changed(lambda val, nm=name: sim.w.__setitem__(nm, val))
    sax = fig.add_axes([0.60, 0.22, 0.30, 0.025])
    s_rad = Slider(sax, "r_interact", 10, 200, valinit=ap.r_interact)
    s_rad.on_changed(lambda val: setattr(sim.ap, "r_interact", val))
    sax = fig.add_axes([0.60, 0.175, 0.30, 0.025])
    s_obs = Slider(sax, "r_obstacle", 10, 150, valinit=ap.r_obstacle)
    s_obs.on_changed(lambda val: setattr(sim.ap, "r_obstacle", val))
    sax = fig.add_axes([0.60, 0.13, 0.30, 0.025])
    s_spd = Slider(sax, "max_speed", 1.5, 7.0, valinit=ap.max_speed)
    s_spd.on_changed(lambda val: setattr(sim.ap, "max_speed", val))

    bax = fig.add_axes([0.60, 0.05, 0.12, 0.05])
    btn_reset = Button(bax, "Сброс")
    bax2 = fig.add_axes([0.76, 0.05, 0.14, 0.05])
    btn_clear = Button(bax2, "Убрать камни")

    def redraw_obstacles():
        for p in patches:
            p.remove()
        patches.clear()
        for x, y, r in sim.env.obstacles:
            c = Circle((x, y), r, fc="#8a7f73", ec="#4b443d", lw=1.5, zorder=2)
            ax.add_patch(c)
            patches.append(c)

    def reset(_=None):
        sim.reset()
        trail[:] = sim.pos[None, :, :]

    def clear(_=None):
        sim.env.obstacles = np.zeros((0, 3))
        redraw_obstacles()

    btn_reset.on_clicked(reset)
    btn_clear.on_clicked(clear)

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        if ev.button == 1:
            sim.env.obstacles = np.vstack([sim.env.obstacles, [ev.xdata, ev.ydata, 25.0]])
        elif ev.button == 3 and len(sim.env.obstacles):
            d = np.hypot(sim.env.obstacles[:, 0] - ev.xdata, sim.env.obstacles[:, 1] - ev.ydata)
            sim.env.obstacles = np.delete(sim.env.obstacles, int(d.argmin()), axis=0)
        redraw_obstacles()

    fig.canvas.mpl_connect("button_press_event", on_click)

    def update(_frame):
        for _ in range(2):                      # 2 шага симуляции на кадр
            sim.step()
            trail[:-1] = trail[1:]
            trail[-1] = sim.pos
        quiv.set_offsets(sim.pos)
        quiv.set_UVC(sim.vel[:, 0], sim.vel[:, 1])
        lc.set_segments([trail[:, i, :] for i in range(sim.ap.n)])
        info.set_text(f"t={sim.t}   поляризация={sim.polarization():.2f}   групп={sim.count_groups()}   "
                      f"выталкиваний из камней={sim.corrections}")
        return quiv, lc, info

    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()
    return anim


def save_gif(path: str, env: Env, ap: AgentParams, weights: dict, seed: int = 0,
             steps: int = 700, every: int = 4):
    """Сохранить анимацию в GIF (без окна)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    sim = Simulation(env, ap, weights, seed)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _setup_axes(ax, env)
    trail_len = 20
    trail = np.repeat(sim.pos[None, :, :], trail_len, axis=0)
    lc = LineCollection([], colors="#1f77b4", linewidths=0.8, alpha=0.5, zorder=3)
    ax.add_collection(lc)
    quiv = ax.quiver(sim.pos[:, 0], sim.pos[:, 1], sim.vel[:, 0], sim.vel[:, 1], angles="xy",
                     scale_units="xy", scale=0.35, width=0.004, color="#d9480f", zorder=4)
    txt = ax.text(8, env.height - 22, "", fontsize=10, zorder=5)
    fig.tight_layout()

    def update(_):
        for _ in range(every):
            sim.step()
            trail[:-1] = trail[1:]
            trail[-1] = sim.pos
        quiv.set_offsets(sim.pos)
        quiv.set_UVC(sim.vel[:, 0], sim.vel[:, 1])
        lc.set_segments([trail[:, i, :] for i in range(sim.ap.n)])
        txt.set_text(f"t = {sim.t}")
        return quiv, lc, txt

    anim = FuncAnimation(fig, update, frames=steps // every, blit=False)
    anim.save(path, writer=PillowWriter(fps=15), dpi=70)
    plt.close(fig)
    print("GIF сохранён:", path)


# 6. Запуск

def default_setup(seed: int = 0):
    env = Env()
    env.obstacles = make_obstacles(8, (15, 35), env, seed=seed)
    return env, AgentParams(), dict(DEFAULT_WEIGHTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", help="сохранить анимацию в GIF и выйти")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env, ap, weights = default_setup(args.seed)
    if args.gif:
        save_gif(args.gif, env, ap, weights, args.seed)
    else:
        interactive(env, ap, weights, args.seed)
