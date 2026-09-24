"""Почему логрегрессия по 5 признакам (OOF 0,78) хуже одного сырого признака (0,91)? Перебор наборов и
преобразований (лог для тяжёлых хвостов) на честном OOF — для предметов и укладки бедра."""
import json, os, sys, itertools
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_criteria import manifest, human_landmarks, features_for, oof_probs, WORK
from dxaqc.criteria import SPINE_FEATURES, HIP_FEATURES
from sklearn.metrics import roc_auc_score, f1_score

man = manifest()
def best_f1(y, p):
    return max(f1_score(y, p >= t, zero_division=0) for t in np.linspace(0.05, 0.95, 91))

for kind, crit, cands, feat_names in (
    ("spine", "spine_artifact", ["th_frac40", "th_p999", "th_lines", "th_linearea", "bg_median", "cnn_artifact"], SPINE_FEATURES),
    ("hip", "hip_positioning_rotation", ["shaft_img_deg", "shaft_deg", "ischium_conf", "ischium_cut", "lt_mm", "lt_conf", "n_missing", "base_conf_min", "cnn_hip_pos", "lowest_gap_cm", "head_gt_dy_mm", "neck_mm"], HIP_FEATURES),
):
    net = features_for(kind, json.load(open(os.path.join(WORK, f"landmarks_oof_{kind}.json"))), man)
    ids = sorted(i for i in net if man[i]["labeled"] and man[i][crit] is not None)
    y = np.array([man[i][crit] for i in ids]); g = np.array([man[i]["study"] for i in ids])
    X = {f: np.array([net[i][f] for i in ids], float) for f in cands}
    heavy = {f for f in cands if f.startswith(("th_", "lt_mm")) }
    def mat(fs, log):
        cols = []
        for f in fs:
            v = X[f]
            if log and f in heavy:
                v = np.log1p(v / max(1e-9, np.percentile(v[v > 0], 50) if (v > 0).any() else 1.0))
            cols.append(v)
        return np.stack(cols, 1)
    print(f"\n=== {crit}: n={len(y)} +{y.sum()}")
    print("  сырые AUC:", ", ".join(f"{f} {roc_auc_score(y, X[f]):.2f}" for f in cands))
    res = []
    for k in (1, 2, 3):
        for fs in itertools.combinations(cands, k):
            for log in (False, True):
                if log and not (set(fs) & heavy): continue
                p = oof_probs(mat(fs, log), y, g)
                res.append((roc_auc_score(y, p), best_f1(y, p), fs, log))
    res.sort(key=lambda r: -r[0])
    for a, f1, fs, log in res[:12]:
        print(f"  AUC {a:.3f}  F1 {f1:.3f}  {'log ' if log else '    '}{'+'.join(fs)}")
    full = oof_probs(mat(cands, True), y, g)
    print(f"  все {len(cands)} с логом: AUC {roc_auc_score(y, full):.3f} F1 {best_f1(y, full):.3f}")
