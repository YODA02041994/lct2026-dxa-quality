#!/usr/bin/env python3
"""Сверка разметки ориентиров с метками экспертов + согласованность разметчиков.

Зачем: убедиться ПО ХОДУ разметки, что признаки, которые мы считаем по точкам, действительно отличают снимки,
помеченные экспертами как нарушение, от нормальных. Если признак не различает — меняем протокол сейчас, а не 28.09.

    python scripts/label_report.py            # забрать разметку с сервера и показать отчёт
    python scripts/label_report.py --local    # только локальная папка data/work/labeler/ann
"""
from __future__ import annotations

import collections, csv, glob, json, math, os, subprocess, sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "data", "work", "labeler", "ann")
PNG = os.path.join(ROOT, "data", "work", "labeler", "png")
SP = "Поясничный отдел позвоночника"
MM = {300: 0.60, 280: 0.65, 248: 0.61}


def pull():
    os.makedirs(ANN, exist_ok=True)
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "vps",
                          "cd /var/lib/lct_labeler/ann && for f in *.json; do printf '\\n@@@%s\\n' \"$f\"; cat \"$f\"; done"],
                         capture_output=True, text=True, check=True).stdout
    n = 0
    for block in out.split("\n@@@")[1:]:
        name, body = block.split("\n", 1)
        open(os.path.join(ANN, name.strip()), "w", encoding="utf-8").write(body)
        n += 1
    return n


def near_mask(img: np.ndarray, x: float, y: float, reach: int = 6) -> bool:
    """Точка стоит у границы полностью чёрной области (маска лаборанта или край поля) — структура обрезана."""
    h, w = img.shape
    x0, x1 = max(0, int(x) - 4), min(w, int(x) + 5)
    y1 = min(h, int(y) + reach + 1)
    below = img[min(h - 1, int(y) + 2):y1, x0:x1]
    return below.size > 0 and (below == 0).mean() > 0.8 or int(y) + reach >= h


def features(a: dict, img: np.ndarray) -> dict:
    P = collections.defaultdict(list)
    for p in a["points"]:
        P[p["label"]].append((p["x"], p["y"]))
    absent, fuzzy = set(a.get("absent", [])), set(a.get("fuzzy", []))
    h, w = img.shape
    mm = MM.get(w, 0.6)
    f: dict = {}
    if a["kind"] == "spine":
        vb = sorted(P["vb"] + [P[k][0] for k in ("L1", "L2", "L3", "L4", "L5") if P[k]], key=lambda t: t[1])   # старый и новый формат
        f["levels"] = "".join(k[1] if P[k] else ("-" if k in absent else "?") for k in ("L1", "L2", "L3", "L4", "L5")) if not P["vb"] else "без уровней"
        if len(vb) >= 2:
            f["axis_deg"] = round(abs(math.degrees(math.atan(np.polyfit([v[1] for v in vb], [v[0] for v in vb], 1)[0]))), 1)
            f["axis_end2end_deg"] = round(abs(math.degrees(math.atan2(vb[-1][0] - vb[0][0], vb[-1][1] - vb[0][1]))), 1)
        f["n_vb"] = len(vb)
        f["th12_seen"] = int("th12" not in absent)
        f["crests_seen"] = int("crest_l" not in absent) + int("crest_r" not in absent)
        f["artifact"] = str(len(a.get("boxes", []))) if a.get("boxes") else ("нет" if "artifact" in absent else "?")
    else:
        one = lambda k: P[k][0] if P[k] else None
        gt, gl, lt, isch = one("gt_top"), one("gt_lat"), one("lt_tip"), one("ischium")
        side = a["image_id"][-1]
        if gt: f["top_margin_cm"] = round(gt[1] * mm / 10, 1)
        if gl: f["lat_margin_cm"] = round((gl[0] if side == "R" else w - gl[0]) * mm / 10, 1)
        low = one("lt_down") or lt
        if low: f["bottom_margin_cm"] = round((h - low[1]) * mm / 10, 1)
        f["lt_state"] = "не видно" if "lt_tip" in absent else ("нечётко" if "lt_tip" in fuzzy else "видно")
        if lt and one("lt_up") and one("lt_down"):
            (x1, y1), (x2, y2) = one("lt_up"), one("lt_down")
            f["lt_prominence_mm"] = round(abs((y2 - y1) * lt[0] - (x2 - x1) * lt[1] + x2 * y1 - y2 * x1) / math.hypot(x2 - x1, y2 - y1) * mm, 1)
        f["ischium"] = "не видно" if "ischium" in absent else ("обрезана" if isch and near_mask(img, *isch) else "видна")
        st, sb = one("shaft_top"), one("shaft_bot")
        if st and sb: f["shaft_deg"] = round(abs(math.degrees(math.atan2(sb[0] - st[0], sb[1] - st[1]))), 1)
        f["artifact"] = str(len(a.get("boxes", []))) if a.get("boxes") else ("нет" if "artifact" in absent else "?")
    return f


def main() -> int:
    if "--local" not in sys.argv:
        print("забрано файлов с сервера:", pull())
    man = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "data/work/manifest.csv"), encoding="utf-8-sig")):
        iid = f"{int(r['num']):03d}_{'spine' if r['region'] == SP else 'hip' + r['side']}"
        man[iid] = r
    anns = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(ANN, "*.json")))]
    done = [a for a in anns if a.get("done")]
    who = collections.Counter(a["annotator"] for a in done)
    print(f"готовых разметок: {len(done)} (черновиков {len(anns) - len(done)}) | по людям: {dict(who)} | уникальных снимков: {len({a['image_id'] for a in done})}")
    secs = [a["seconds"] for a in done if a.get("seconds")]
    if secs: print(f"время на снимок: медиана {int(np.median(secs))} с")

    rows = []
    for a in done:
        img = np.array(Image.open(os.path.join(PNG, a["image_id"] + ".png")))
        rows.append((a, features(a, img), man[a["image_id"]]))

    print("\n=== ПОЗВОНОЧНИК: признаки по точкам ↔ метки экспертов")
    print(f"{'снимок':11}{'кто':11}{'ось°':>6}{'Th12':>6}{'уровни':>13}{'гребни':>8}{'предметы':>10}   эксперт: укладка ось предметы   комментарий")
    for a, f, m in rows:
        if a["kind"] != "spine": continue
        print(f"{a['image_id']:11}{a['annotator'][:10]:11}{f.get('axis_deg', ''):>6}{f['th12_seen']:>6}{f.get('levels', ''):>13}{f['crests_seen']:>8}{f['artifact']:>10}"
              f"            {m['spine_positioning'] or '-':>5} {m['spine_axis_tilt'] or '-':>5} {m['spine_artifact'] or '-':>6}      {m['comment']}")
    print("\n=== БЕДРО: признаки по точкам ↔ метки экспертов")
    print(f"{'снимок':11}{'кто':11}{'сверху':>7}{'сбоку':>7}{'снизу':>7}  {'малый вертел':14}{'выст.мм':>8}  {'седалищная':11}{'диафиз°':>8}   эксперт: укладка поле   комментарий")
    for a, f, m in rows:
        if a["kind"] != "hip": continue
        print(f"{a['image_id']:11}{a['annotator'][:10]:11}{f.get('top_margin_cm', ''):>7}{f.get('lat_margin_cm', ''):>7}{f.get('bottom_margin_cm', ''):>7}  "
              f"{f['lt_state']:14}{f.get('lt_prominence_mm', ''):>8}  {f['ischium']:11}{f.get('shaft_deg', ''):>8}            {m['hip_positioning_rotation'] or '-':>5} {m['hip_roi_field'] or '-':>5}     {m['comment']}")

    print("\n=== СОГЛАСОВАННОСТЬ: один снимок — два разметчика")
    by = collections.defaultdict(list)
    for a, f, m in rows: by[a["image_id"]].append(a)
    for iid, lst in sorted(by.items()):
        if len(lst) < 2: continue
        a, b = lst[0], lst[1]
        mm = MM.get(int(man[iid]["cols"]), 0.6)
        pa = {p["label"]: (p["x"], p["y"]) for p in a["points"] if p["label"] != "vb" and not p["label"].startswith("L")}
        pb = {p["label"]: (p["x"], p["y"]) for p in b["points"] if p["label"] != "vb" and not p["label"].startswith("L")}
        d = {k: round(math.hypot(pa[k][0] - pb[k][0], pa[k][1] - pb[k][1]) * mm, 1) for k in pa if k in pb}
        isv = lambda p: p["label"] == "vb" or p["label"] in ("L1", "L2", "L3", "L4", "L5")
        va, vbb = sorted([(p["x"], p["y"]) for p in a["points"] if isv(p)], key=lambda t: t[1]), sorted([(p["x"], p["y"]) for p in b["points"] if isv(p)], key=lambda t: t[1])
        if va and len(va) == len(vbb): d["vb(средн.)"] = round(float(np.mean([math.hypot(x[0] - y[0], x[1] - y[1]) for x, y in zip(va, vbb)])) * mm, 1)
        dis = sorted(set(a.get("absent", [])) ^ set(b.get("absent", [])))
        print(f"{iid}  {a['annotator']} ↔ {b['annotator']}: расхождение, мм: {d}" + (f"  | РАЗНЫЙ ответ «не видно»: {dis}" if dis else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
