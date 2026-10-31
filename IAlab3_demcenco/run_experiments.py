# -*- coding: utf-8 -*-
"""
Лабораторная работа №3, вариант 2. Серия экспериментов с косяком рыб.
Запуск:  python run_experiments.py
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from boids_fish import (DEFAULT_WEIGHTS, AgentParams, Env, Simulation, make_obstacles, norm)

STEPS = 1200          # шагов на один запуск
WARMUP = 300          # первые шаги не учитываются в средних (косяк ещё собирается)
SEEDS = [0, 1, 2, 3, 4]

# (код, группа, описание, число камней, диапазон радиусов, изменения AgentParams, изменения весов)
EXPERIMENTS = [
    ("A0", "Препятствия", "без препятствий",            0,  (15, 35), {}, {}),
    ("A1", "Препятствия", "3 камня (r 15-35)",          3,  (15, 35), {}, {}),
    ("A2", "Препятствия", "8 камней (r 15-35), БАЗОВЫЙ", 8, (15, 35), {}, {}),
    ("A3", "Препятствия", "15 камней (r 15-35)",        15, (15, 35), {}, {}),
    ("A4", "Препятствия", "6 мелких камней (r 8-14)",   6,  (8, 14),  {}, {}),
    ("A5", "Препятствия", "5 крупных камней (r 40-60)", 5,  (40, 60), {}, {}),
    ("B1", "Радиус взаимодействия", "r_interact = 25",  8,  (15, 35), {"r_interact": 25.0}, {}),
    ("B2", "Радиус взаимодействия", "r_interact = 140", 8,  (15, 35), {"r_interact": 140.0}, {}),
    ("C1", "Веса Boids", "cohesion = 0",                8,  (15, 35), {}, {"cohesion": 0.0}),
    ("C2", "Веса Boids", "cohesion = 2.5",              8,  (15, 35), {}, {"cohesion": 2.5}),
    ("C3", "Веса Boids", "alignment = 0",               8,  (15, 35), {}, {"alignment": 0.0}),
    ("C4", "Веса Boids", "alignment = 3",               8,  (15, 35), {}, {"alignment": 3.0}),
    ("C5", "Веса Boids", "separation = 4",              8,  (15, 35), {}, {"separation": 4.0}),
    ("D1", "Вес избегания", "obstacle = 0",             8,  (15, 35), {}, {"obstacle": 0.0}),
    ("D2", "Вес избегания", "obstacle = 1",             8,  (15, 35), {}, {"obstacle": 1.0}),
    ("D3", "Вес избегания", "obstacle = 8",             8,  (15, 35), {}, {"obstacle": 8.0}),
]


def build(n_obs, r_range, ap_kw, w_kw, seed):
    env = Env()
    env.obstacles = make_obstacles(n_obs, r_range, env, seed=seed)
    ap = AgentParams(**ap_kw)
    w = {**DEFAULT_WEIGHTS, **w_kw}
    return Simulation(env, ap, w, seed)


def run_once(n_obs, r_range, ap_kw, w_kw, seed, record=False):
    sim = build(n_obs, r_range, ap_kw, w_kw, seed)
    pol, grp, speeds = [], [], []
    min_clear, max_speed = np.inf, 0.0
    traj, pol_t, grp_t = [], [], []
    for t in range(STEPS):
        sim.step()
        min_clear = min(min_clear, sim.min_clearance())
        max_speed = max(max_speed, float(norm(sim.vel).max()))
        if t % 10 == 0:
            p, g = sim.polarization(), sim.count_groups()
            if t >= WARMUP:
                pol.append(p)
                grp.append(g)
                speeds.append(float(norm(sim.vel).mean()))
            if record:
                pol_t.append(p)
                grp_t.append(g)
        if record:
            traj.append(sim.pos.copy())
    res = dict(corrections=sim.corrections, polarization=float(np.mean(pol)), groups=float(np.mean(grp)),
               groups_max=int(np.max(grp)), speed=float(np.mean(speeds)), max_speed=max_speed,
               min_clearance=float(min_clear), coverage=float(
                   np.pi * (sim.env.obstacles[:, 2] ** 2).sum() / (sim.env.width * sim.env.height) * 100)
               if len(sim.env.obstacles) else 0.0)
    if record:
        res.update(traj=np.array(traj), pol_t=pol_t, grp_t=grp_t, obstacles=sim.env.obstacles.copy())
    return res


def draw_panel(ax, res, title, tail=150):
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 600)
    ax.set_aspect("equal")
    ax.set_facecolor("#dff1fb")
    ax.set_xticks([])
    ax.set_yticks([])
    for x, y, r in res["obstacles"]:
        ax.add_patch(Circle((x, y), r, fc="#8a7f73", ec="#4b443d", lw=1.2, zorder=2))
    tr = res["traj"][-tail:]
    for i in range(tr.shape[1]):
        ax.plot(tr[:, i, 0], tr[:, i, 1], color="#1f77b4", lw=0.7, alpha=0.6, zorder=3)
    last = res["traj"][-1]
    prev = res["traj"][-2]
    ax.quiver(last[:, 0], last[:, 1], last[:, 0] - prev[:, 0], last[:, 1] - prev[:, 1], angles="xy",
              scale_units="xy", scale=0.35, width=0.006, color="#d9480f", zorder=4)
    ax.set_title(title, fontsize=10)


def main():
    rows, rec = [], {}
    for code, group, desc, n_obs, r_range, ap_kw, w_kw in EXPERIMENTS:
        runs = [run_once(n_obs, r_range, ap_kw, w_kw, s, record=(s == 0)) for s in SEEDS]
        rec[code] = runs[0]
        mean = lambda k: float(np.mean([r[k] for r in runs]))
        row = dict(code=code, group=group, description=desc,
                   corrections=mean("corrections"), polarization=mean("polarization"),
                   groups=mean("groups"), groups_std=float(np.std([r["groups"] for r in runs])),
                   polarization_std=float(np.std([r["polarization"] for r in runs])), groups_max=float(np.max([r["groups_max"] for r in runs])),
                   speed=mean("speed"), max_speed=float(np.max([r["max_speed"] for r in runs])),
                   min_clearance=float(np.min([r["min_clearance"] for r in runs])),
                   coverage=mean("coverage"))
        rows.append(row)
        print(f"{code} {desc:34s} выталк.={row['corrections']:8.1f}  поляризация={row['polarization']:.2f}  "
              f"групп={row['groups']:.1f}\u00b1{row['groups_std']:.1f} (макс {row['groups_max']:.0f})  v_ср={row['speed']:.2f}  "
              f"v_макс={row['max_speed']:.2f}  мин.зазор={row['min_clearance']:.2f}  покрытие={row['coverage']:.1f}%")

    with open("experiments_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]
    json.dump(clean, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # --- картинка 1: траектории в базовом эксперименте 
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_panel(ax, rec["A2"], "Базовый эксперимент (8 камней): траектории последних 150 шагов, seed 0", tail=150)
    fig.tight_layout()
    fig.savefig("fig_baseline.png", dpi=150)
    plt.close(fig)

    # --- картинка 2: сравнение сцен 
    panels = [("A2", "A2: базовый"), ("A3", "A3: 15 камней"), ("A5", "A5: 5 крупных камней"),
              ("B1", "B1: r_interact = 25"), ("D1", "D1: вес избегания = 0"), ("C5", "C5: separation = 4")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, (code, title) in zip(axes.ravel(), panels):
        draw_panel(ax, rec[code], title)
    fig.tight_layout()
    fig.savefig("fig_grid.png", dpi=110)
    plt.close(fig)

    # --- картинка 3: графики метрик 
    byc = {r["code"]: r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    cs = ["D1", "D2", "A2", "D3"]
    labels = ["0", "1", "3 (баз.)", "8"]
    vals = [byc[c]["corrections"] for c in cs]
    ax.bar(labels, vals, color="#d9480f")
    ax.set_yscale("symlog", linthresh=1)
    for i, v in enumerate(vals):
        ax.text(i, max(v, 0.3), f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Вес избегания препятствий и число выталкиваний\nрыб из камней (за 1200 шагов)")
    ax.set_xlabel("вес правила избегания")
    ax.set_ylabel("выталкиваний (лог. шкала)")

    ax = axes[1]
    cs = ["B1", "A2", "B2"]
    labels = ["25", "70 (баз.)", "140"]
    ax.bar(labels, [byc[c]["groups"] for c in cs], color="#2b8a3e")
    for i, c in enumerate(cs):
        ax.text(i, byc[c]["groups"], f"{byc[c]['groups']:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Радиус взаимодействия и среднее число групп")
    ax.set_xlabel("r_interact")
    ax.set_ylabel("групп")

    ax = axes[2]
    cs = ["A0", "A1", "A2", "A3"]
    labels = ["0", "3", "8 (баз.)", "15"]
    ax.bar(range(4), [byc[c]["groups"] for c in cs], 0.5, yerr=[byc[c]["groups_std"] for c in cs],
           capsize=4, color="#1f77b4", label="число групп (± разброс по seed)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("групп")
    ax.set_xlabel("число препятствий")
    ax.set_title("Число камней: группы и согласованность движения")
    ax.set_ylim(0, 4.6)
    ax2 = ax.twinx()
    ax2.plot(range(4), [byc[c]["polarization"] for c in cs], "o-", color="#f08c00", label="поляризация")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("поляризация")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("fig_metrics.png", dpi=130)
    plt.close(fig)

    # --- картинка 4: динамика группировки со временем 
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    t = np.arange(len(rec["A2"]["pol_t"])) * 10
    for code, label, col in [("A0", "без препятствий", "#2b8a3e"), ("A2", "8 камней (базовый)", "#1f77b4"),
                             ("A3", "15 камней", "#d9480f")]:
        axes[0].plot(t, rec[code]["grp_t"], label=label, color=col)
        axes[1].plot(t, rec[code]["pol_t"], label=label, color=col)
    axes[0].set_title("Число групп во времени (seed 0)")
    axes[1].set_title("Поляризация во времени (seed 0)")
    for ax in axes:
        ax.set_xlabel("шаг симуляции")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_timeseries.png", dpi=130)
    plt.close(fig)
    print("Готово: experiments_results.csv, fig_baseline.png, fig_grid.png, fig_metrics.png, fig_timeseries.png")


if __name__ == "__main__":
    main()
