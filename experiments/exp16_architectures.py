#!/usr/bin/env python3
"""Эксп. 16: крупные и трансформерные сети против ResNet18. Сети обучаются scripts/train_cnn_criterion.py с ключом --arch
(команды — в docs/08, раздел 4), здесь сравнение по предсказаниям вне обучения: сеть одна, сеть с измерениями по снимку,
сеть добавкой к боевому ансамблю. Скрипт только читает data/work/cnn_oof_*.json и ничего не записывает.

Запуск: PYTHONPATH=src python experiments/exp16_architectures.py
"""
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
import train_criteria as tc
from dxaqc.cnn import CNN_SOURCES
from dxaqc.criteria import CRITERION_FEATURES
WORK = tc.WORK
man = tc.manifest()

NEW = {
    "hip_positioning_rotation": ("hip", "cnn_hip_pos", [
        "hip_positioning_rotation_lt100_eff", "hip_positioning_rotation_lt100_r50_in", "hip_positioning_rotation_lt100_effv2s",
        "hip_positioning_rotation_lt100_effv2s_e40", "hip_positioning_rotation_lt100_sx50", "hip_positioning_rotation_lt100_regy",
        "hip_positioning_rotation_lt100_dn121", "hip_positioning_rotation_lt100_xrv", "hip_positioning_rotation_lt100_cnxt",
        "hip_positioning_rotation_lt100_dinov2s", "hip_positioning_rotation_lt100_dinov2b",
        "hip_any_r50_320_in", "hip_any_effv2s_320", "hip_any_dinov2s_322"]),
    "spine_artifact": ("spine", "cnn_artifact", [
        "spine_artifact_eff_320", "spine_artifact_r50_320_in", "spine_artifact_dn121_320", "spine_artifact_xrv_320",
        "spine_artifact_effv2s_512", "spine_artifact_cnxt_384", "spine_artifact_dinov2s_448"]),
}


def load(name):
    p = os.path.join(WORK, f"cnn_oof_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def run(crit, kind, feat, tags):
    net = tc.features_for(kind, json.load(open(os.path.join(WORK, f"landmarks_oof_{kind}.json"))), man)
    ids = sorted(i for i in net if man[i]["labeled"] and man[i][crit] is not None)
    y = np.array([man[i][crit] for i in ids], int)
    g = np.array([man[i]["study"] for i in ids])
    base = CNN_SOURCES[feat]
    other = [f for f in CRITERION_FEATURES[crit] if f != feat]
    src = {n: load(n) for n in base + tags}

    def ens(names):
        return np.array([np.mean([src[n][i] for n in names if src[n] and i in src[n]]) for i in ids])

    def criterion(cnn_vec):
        X = np.column_stack([np.array([[net[i][f] for f in other] for i in ids], float), cnn_vec])
        p = tc.oof_probs(X, y, g)
        lo, hi = tc.ci(roc_auc_score, y, p, g, n=500)
        return roc_auc_score(y, p), lo, hi, tc.best_f1(y, p)[0]

    print(f"\n{crit}: снимков {len(ids)}, с нарушением {int(y.sum())}")
    a, lo, hi, f1 = criterion(ens(base))
    print(f"  {'боевой ансамбль (' + str(len(base)) + ' файлов весов)':46} сеть {roc_auc_score(y, ens(base)):.3f} | критерий {a:.3f} [{lo:.3f}; {hi:.3f}] F1 {f1:.3f}")
    for n in base:
        print(f"    {n:44} сеть {roc_auc_score(y, ens([n])):.3f}")
    print(f"  {'кандидат':46} {'сеть':>6} {'сеть+измерения':>15} {'в ансамбле: критерий [95 % ДИ]':>34} {'F1':>6}")
    for t in tags:
        if not src[t]:
            continue
        a1 = roc_auc_score(y, ens([t]))
        a2 = criterion(ens([t]))[0]
        a3, lo3, hi3, f3 = criterion(ens(base + [t]))
        print(f"  {t:46} {a1:6.3f} {a2:15.3f} {a3:12.3f} [{lo3:.3f}; {hi3:.3f}] {f3:12.3f}")


if __name__ == "__main__":
    for crit, (kind, feat, tags) in NEW.items():
        run(crit, kind, feat, tags)
