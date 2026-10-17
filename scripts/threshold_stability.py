#!/usr/bin/env python3
"""Эксп. 26: как выбраны пороги и насколько они устойчивы (вопрос экспертов на консультации 16.10).

Для каждого критерия по вероятностям вне обучения (ось — после отделения сколиоза, как в бою): F1 при каждом пороге 0,01–0,99,
отрезок порогов с наибольшим F1 и его середина (так выбран боевой порог, train_criteria.best_f1), бутстреп по исследованиям
(1000 повторов) — где оказывается лучший порог на перевыборках и какой F1 даёт боевой порог.
    PYTHONPATH=src python scripts/threshold_stability.py   → presentation/fig/thresholds.png, data/work/threshold_stability.json
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.scoliosis import damp_axis, measure   # noqa: E402
WORK = os.path.join(ROOT, "data", "work")
NAMES = {"spine_positioning": "Укладка (позвоночник)", "spine_axis_tilt": "Ось позвоночника", "spine_artifact": "Посторонние предметы",
         "hip_positioning_rotation": "Укладка и ротация (бедро)", "hip_roi_field": "Поле сканирования (бедро)"}
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def f1_at(y, p, t):
    yh = p >= t; tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def best(y, p):
    f = np.array([f1_at(y, p, t) for t in GRID]); m = f.max()
    idx = np.where(np.isclose(f, m))[0]
    return f, m, GRID[idx.min()], GRID[idx.max()]


def main():
    oof = json.load(open(os.path.join(WORK, "criteria_oof.json"))); lm = json.load(open(os.path.join(WORK, "landmarks_oof_spine.json")))
    cj = json.load(open(os.path.join(ROOT, "weights", "criteria.json"), encoding="utf-8"))
    thr = {c: float(m["threshold"]) for c, m in cj["models"].items()}
    man = {}
    for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
        kind = "spine" if r["region"].startswith("Пояс") else "hip" + r["side"]
        man[f"{int(r['num']):03d}_{kind}"] = r
    out, rng = {}, np.random.default_rng(0)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=True)
    for ax, c in zip(axes, thr):
        ids = [i for i in oof if c in oof[i] and man.get(i, {}).get(c, "") != ""]
        p = []
        for i in ids:
            probs = {k: v for k, v in oof[i].items()}
            if c == "spine_axis_tilt" and i in lm:
                probs, _ = damp_axis(probs, measure(lm[i])["grade"])
            p.append(probs[c])
        p = np.array(p); y = np.array([int(man[i][c]) for i in ids]); g = np.array([man[i]["study_dir"] for i in ids])
        f, m, lo, hi = best(y, p)
        ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; bt, bf = [], []
        for _ in range(1000):
            sel = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
            if y[sel].sum() == 0:
                continue
            _, _, l2, h2 = best(y[sel], p[sel]); bt.append((l2 + h2) / 2); bf.append(f1_at(y[sel], p[sel], thr[c]))
        t_lo, t_hi = np.percentile(bt, [2.5, 97.5]); f_lo, f_hi = np.percentile(bf, [2.5, 97.5])
        near = GRID[f >= m - 0.05]                                            # пороги, где F1 не хуже максимума на 0,05
        out[c] = {"n": int(len(y)), "pos": int(y.sum()), "threshold": thr[c], "f1_max": round(float(m), 3), "plateau": [float(lo), float(hi)],
                  "within_005": [float(near.min()), float(near.max())], "boot_best_ci": [round(float(t_lo), 2), round(float(t_hi), 2)],
                  "f1_at_threshold_ci": [round(float(f_lo), 3), round(float(f_hi), 3)]}
        ax.plot(GRID, f, color="#D55E00", lw=2.2)
        ax.axvspan(near.min(), near.max(), color="#E69F00", alpha=0.18, lw=0)
        ax.axvspan(t_lo, t_hi, ymin=0, ymax=0.06, color="#0072B2", alpha=0.6, lw=0)
        ax.axvline(thr[c], color="black", lw=1.2)
        ax.set_title(f"{NAMES[c]}\n{int(y.sum())} нарушений из {len(y)}", fontsize=12, loc="left")
        ax.text(thr[c] + 0.02, 0.93, f"порог {thr[c]:.2f}".replace(".", ","), fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("порог вероятности", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("F1 вне обучения", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.join(ROOT, "presentation", "fig"), exist_ok=True)
    plt.savefig(os.path.join(ROOT, "presentation", "fig", "thresholds.png"), dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    json.dump(out, open(os.path.join(WORK, "threshold_stability.json"), "w"), ensure_ascii=False, indent=1)
    for c, v in out.items():
        print(f"{c:26s} порог {v['threshold']:.2f} | плато макс. F1 {v['plateau']} | F1 ≥ макс−0,05: {v['within_005']} | "
              f"лучший порог на бутстрепе {v['boot_best_ci']} | F1 при боевом пороге {v['f1_max']:.2f} [{v['f1_at_threshold_ci'][0]}; {v['f1_at_threshold_ci'][1]}]")


if __name__ == "__main__":
    main()
