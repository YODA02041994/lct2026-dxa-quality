#!/usr/bin/env python3
"""Режимы порогов в weights/criteria.json (запускать после train_criteria.py).

competition — пороги по максимуму F1 на снимках вне обучения (их подбирает train_criteria.py);
soft        — наибольший порог с чувствительностью вне обучения не ниже 0,8: зона «проверить укладку» (predict.aggregate);
sensitive   — по каждому критерию наибольший порог, при котором чувствительность вне обучения не ниже 0,9.
    PYTHONPATH=src python scripts/make_modes.py [критерий ...]   # без аргументов — пересчитать все критерии
"""
import csv, json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "data", "work")
TARGET = 0.9
SOFT_TARGET = 0.8


def sensitive_threshold(y: np.ndarray, p: np.ndarray, target: float = TARGET) -> float:
    best = 0.01
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        if ((p >= t) & (y == 1)).sum() / max(1, (y == 1).sum()) >= target:
            best = float(t)
    return best


def main(criteria: list[str] | None = None) -> dict:
    """criteria — для каких критериев пересчитать порог sensitive (по умолчанию для всех); остальные сохраняются."""
    path = os.path.join(ROOT, "weights", "criteria.json")
    cj = json.load(open(path, encoding="utf-8"))
    oof = json.load(open(os.path.join(WORK, "criteria_oof.json")))
    truth = {}
    for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
        kind = "spine" if r["region"].startswith("Пояс") else "hip"
        truth[f"{int(r['num']):03d}_{'spine' if kind == 'spine' else 'hip' + r['side']}"] = r
    old = cj.get("modes") or {}
    modes = {"competition": {c: round(float(m["threshold"]), 2) for c, m in cj["models"].items()}, "soft": {}, "sensitive": {}}
    for c in cj["models"]:
        if criteria and c not in criteria and c in old.get("sensitive", {}) and c in old.get("soft", {}):
            modes["sensitive"][c] = old["sensitive"][c]; modes["soft"][c] = old["soft"][c]
            continue
        ids = [i for i in oof if c in oof[i] and truth.get(i, {}).get(c, "") != ""]
        y = np.array([int(truth[i][c]) for i in ids]); p = np.array([oof[i][c] for i in ids])
        sens = sensitive_threshold(y, p)
        if sens < 0.05 and c in old.get("sensitive", {}):          # recall 0,9 недостижим без порога «всё подряд» (мало примеров) — оставляем прежний
            sens = old["sensitive"][c]
        modes["sensitive"][c] = sens
        # мягкий порог обязан быть ниже основного, иначе зоны «проверить» нет
        modes["soft"][c] = min(sensitive_threshold(y, p, SOFT_TARGET), round(modes["competition"][c] - 0.05, 2))
    modes["note"] = ("competition — пороги по максимуму F1 на OOF (официальный ответ); soft — recall ≥ 0,8 по критерию, зона "
                     "«проверить укладку» в трёхуровневой выдаче; sensitive — recall ≥ 0,9 для клиники, где пропуск дороже перепроверки")
    cj["modes"] = modes
    json.dump(cj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in ("competition", "soft", "sensitive"):
        print(k, modes[k])
    return modes


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
