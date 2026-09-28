#!/usr/bin/env python3
"""Рисунки для презентации (science-deck: Arial, палитра Okabe–Ito, значения подписаны у элементов).
Все числа — из docs/metrics.md, weights/criteria.json, docs/05, docs/08 и рефератов PubMed (PMID в подписях слайдов)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
plt.rcParams.update({"font.family": "Arial", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0})
OURS, LIT, SECOND, SKY, NEUT, GREEN = "#D55E00", "#0072B2", "#E69F00", "#56B4E9", "#999999", "#009E73"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig"); os.makedirs(OUT, exist_ok=True)
def save(name): plt.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor="white"); plt.close()

# 1. литература: ошибки DXA
fig, ax = plt.subplots(1, 2, figsize=(12.6, 3.9), gridspec_kw={"wspace": 0.75, "width_ratios": [1.15, 1]})
lab = ["технические ошибки\nсъёмки и анализа", "ошибки, повлиявшие\nна интерпретацию", "ошибки заключения", "из них значимые\nдля лечения"]
val = [90, 13, 80, 42]
ax[0].barh(lab[::-1], val[::-1], color=LIT, height=0.62)
for i, v in enumerate(val[::-1]): ax[0].text(v + 1.5, i, f"{v} %", va="center")
ax[0].set_xlim(0, 105); ax[0].set_xlabel("доля пациентов, %"); ax[0].set_title("Krueger 2018: 345 направленных\nв клинику остеопороза", fontsize=15, loc="left")
lab2 = ["позвоночник: неверное\nвыделение кости", "бедро: неверная зона\nшейки", "бедро 10° внутр. ротации:\nзначимый сдвиг МПК шейки", "бедро 10° наруж. ротации:\nзначимый сдвиг МПК шейки"]
val2 = [51.9, 37.2, 12, 8]; col2 = [LIT, LIT, SECOND, SECOND]
ax[1].barh(lab2[::-1], val2[::-1], color=col2[::-1], height=0.62)
for i, v in enumerate(val2[::-1]): ax[1].text(v + 1.2, i, f"{str(v).replace('.', ',')} %", va="center")
ax[1].set_xlim(0, 70); ax[1].set_xlabel("доля снимков / пациенток, %"); ax[1].set_title("Jung 2020: 320 снимков до обучения персонала\nLekamwasam 2003: 50 женщин", fontsize=15, loc="left")
save("lit_errors.png")

# 2. данные
fig, ax = plt.subplots(figsize=(11.5, 3.4))
names = ["укладка\nпозвоночника", "ось\nпозвоночника", "посторонние\nпредметы", "укладка/ротация\nбедра", "поле ROI\nбедра"]
pos = [6, 10, 17, 36, 7]; tot = [99, 99, 99, 150, 150]
b = ax.bar(names, pos, color=OURS, width=0.6)
for r, p, t in zip(b, pos, tot): ax.text(r.get_x() + r.get_width() / 2, p + 0.8, f"{p} из {t}\n({p / t * 100:.0f} %)".replace(".", ","), ha="center", va="bottom", fontsize=14)
ax.set_ylim(0, 48); ax.set_ylabel("снимков с нарушением")
save("data.png")

# 3. сравнение вариантов модели
fig, ax = plt.subplots(figsize=(12.2, 3.9))
st = ["ориентиры +\nгеометрия\n+ 2 CNN", "+ признаки\nпо картинке", "+ ансамбль\nсетей,\nвырезки", "+ пред-\nобучение\nobject-CXR", "+ яркость\nмягких тканей", "+ карта\nпредметов", "+ DenseNet121\nс рентгено-\nграмм", "+ общая\nшкала\nкритериев"]
auc = [0.737, 0.821, 0.822, 0.830, 0.841, 0.845, 0.849, 0.857]; f1 = [0.583, 0.719, 0.727, 0.740, 0.752, 0.752, 0.762, 0.762]; mf = [0.535, 0.610, 0.608, 0.627, 0.649, 0.649, 0.653, 0.653]
x = np.arange(len(st))
for v, c, n in ((auc, OURS, "ROC-AUC"), (f1, LIT, "F1"), (mf, GREEN, "macro-F1 по 5 типам")):
    ax.plot(x, v, "-o", color=c, lw=2.5, ms=8, label=n)
    ax.text(x[-1] + 0.12, v[-1], f"{v[-1]:.2f}".replace(".", ","), color=c, va="center", fontsize=15, fontweight="bold")
    ax.text(x[0] - 0.12, v[0], f"{v[0]:.2f}".replace(".", ","), color=c, va="center", ha="right", fontsize=14)
ax.set_xticks(x); ax.set_xticklabels(st, fontsize=11.5); ax.set_ylim(0.5, 0.9); ax.set_xlim(-0.6, 7.7); ax.legend(frameon=False, loc="lower right", fontsize=14, ncol=3)
ax.set_ylabel("значение метрики")
save("progress.png")

# 4. по критериям, 95 % ДИ
fig, ax = plt.subplots(figsize=(11.8, 3.6))
cr = ["Некорректная укладка (позвоночник), 6 из 99", "Не выровнена ось позвоночника, 10 из 99", "Присутствуют посторонние предметы, 17 из 99", "Некорректная укладка (бедро), 36 из 150", "Некорректная область интереса (бедро), 7 из 150"]
a = [0.901, 0.799, 0.922, 0.873, 0.827]; lo = [0.778, 0.661, 0.834, 0.785, 0.501]; hi = [0.990, 0.915, 0.985, 0.940, 0.998]
y = np.arange(len(cr))[::-1]
ax.errorbar(a, y, xerr=[np.array(a) - np.array(lo), np.array(hi) - np.array(a)], fmt="o", color=OURS, ecolor=OURS, elinewidth=2.5, capsize=5, ms=9)
for yi, ai, l, h in zip(y, a, lo, hi): ax.text(1.02, yi, f"{ai:.2f} [{l:.2f}; {h:.2f}]".replace(".", ","), va="center", fontsize=14)
ax.axvline(0.5, color=NEUT, ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels(cr, fontsize=14); ax.set_xlim(0.45, 1.0); ax.set_xlabel("ROC-AUC на снимках вне обучения")
save("forest.png")

# 5. бинарные метрики
fig, ax = plt.subplots(figsize=(11.8, 3.9))
m = ["ROC-AUC", "PR-AUC", "F1", "Accuracy", "Balanced accuracy", "Чувствительность", "Специфичность"]
v = [0.857, 0.718, 0.762, 0.859, 0.835, 0.778, 0.893]; lo = [0.802, 0.598, 0.680, 0.820, 0.786, 0.690, 0.847]; hi = [0.907, 0.843, 0.832, 0.899, 0.884, 0.869, 0.935]
y = np.arange(len(m))[::-1]
ax.errorbar(v, y, xerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], fmt="s", color=OURS, ecolor=OURS, elinewidth=2.5, capsize=5, ms=9)
for yi, ai, l, h in zip(y, v, lo, hi): ax.text(1.01, yi, f"{ai:.3f} [{l:.3f}; {h:.3f}]".replace(".", ","), va="center", fontsize=14)
ax.set_yticks(y); ax.set_yticklabels(m); ax.set_xlim(0.5, 1.0); ax.set_xlabel("значение [95 % ДИ, бутстреп по исследованиям], 249 снимков")
save("binary.png")

# 6. перенос с рентгена грудной клетки
fig, ax = plt.subplots(figsize=(11.6, 3.6))
groups = [("RadImageNet\n(КТ, МРТ, УЗИ)", [0.549], NEUT), ("ImageNet", [0.763, 0.781, 0.805], LIT), ("object-CXR,\n900 снимков", [0.778, 0.833], SECOND), ("object-CXR,\n9000 снимков", [0.857, 0.842, 0.867], OURS)]
for i, (n, vals, c) in enumerate(groups):
    ax.scatter([i + d for d in np.linspace(-0.12, 0.12, len(vals))], vals, s=110, color=c, zorder=3)
    ax.hlines(np.mean(vals), i - 0.3, i + 0.3, color=c, lw=3)
    ax.text(i + 0.34, np.mean(vals), f"{np.mean(vals):.2f}".replace(".", ","), va="center", color=c, fontsize=16, fontweight="bold")
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0.5, 0.92); ax.set_xlim(-0.6, 3.8)
ax.set_ylabel("ROC-AUC сети по предметам\n(точка = одно разбиение)", fontsize=14)
save("transfer.png")

# 7. что проверено: две панели
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.3), gridspec_kw={"wspace": 0.95})
hp = [("угол диафиза по 2 точкам сети", 0.60, 0), ("атлас нормального бедра", 0.67, 0), ("CNN, вырезка шейки", 0.72, 0), ("CNN по всему кадру", 0.76, 1), ("CNN, вырезка седалищной кости", 0.83, 1), ("CNN, вырезка малого вертела", 0.86, 1), ("DenseNet121 с рентгенограмм, вертел", 0.86, 1), ("ансамбль 10 сетей", 0.89, 1)]
ar = [("насыщенные яркие пятна", 0.55, 0), ("RadImageNet + CNN", 0.55, 0), ("аномалия без учителя (PatchCore)", 0.71, 0), ("RAD-DINO замороженный", 0.78, 0), ("CNN, ImageNet", 0.78, 0), ("карта предметов object-CXR", 0.84, 1), ("CNN, предобучение object-CXR", 0.87, 1), ("тонкие яркие линии (top-hat)", 0.91, 1), ("всё вместе", 0.92, 1)]
for axx, data, t in ((ax[0], hp, "Укладка и ротация бедра"), (ax[1], ar, "Посторонние предметы")):
    n = [d[0] for d in data]; v = [d[1] for d in data]; c = [OURS if d[2] else NEUT for d in data]
    axx.barh(n, v, color=c, height=0.66)
    for i, vv in enumerate(v): axx.text(vv + 0.01, i, f"{vv:.2f}".replace(".", ","), va="center", fontsize=13)
    axx.set_xlim(0.4, 1.0); axx.axvline(0.5, color="black", lw=0.8, ls=":"); axx.set_title(t, loc="left", fontsize=15); axx.tick_params(axis="y", labelsize=13); axx.set_xlabel("ROC-AUC вне обучения")
save("tried.png")

# 8. порог
fig, ax = plt.subplots(figsize=(11.6, 3.7))
k = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5]; sens = [0.92, 0.86, 0.82, 0.81, 0.79, 0.78, 0.61, 0.53, 0.47]; spec = [0.47, 0.60, 0.76, 0.81, 0.85, 0.89, 0.92, 0.93, 0.96]; fl = [0.64, 0.53, 0.41, 0.37, 0.34, 0.30, 0.23, 0.20, 0.16]; f1 = [0.569, 0.605, 0.678, 0.712, 0.731, 0.762, 0.677, 0.623, 0.602]
ax.plot(k, sens, "-o", color=OURS, lw=2.5, label="чувствительность"); ax.plot(k, spec, "-o", color=LIT, lw=2.5, label="специфичность")
ax.plot(k, f1, "-s", color=GREEN, lw=2.5, label="F1"); ax.plot(k, fl, "--", color=NEUT, lw=2, label="доля помеченных снимков")
ax.axvline(1.0, color="black", lw=1); ax.text(1.01, 0.12, "порог сдачи:\nмаксимум F1", fontsize=13, va="bottom")
ax.axhline(0.29, color=NEUT, lw=0.8, ls=":"); ax.text(1.52, 0.29, "нарушений в данных 29 %", fontsize=12, va="center", color="#555555")
ax.set_xlabel("множитель порогов критериев (меньше — строже к пропускам)"); ax.set_ylim(0.1, 1.0); ax.set_xlim(0.45, 1.95); ax.legend(frameon=False, fontsize=13, loc="center right")
save("tradeoff.png")
# 9. архитектуры: число параметров и ROC-AUC вне обучения (эксп. 16); None — не обучалась на этой задаче
ARCH = [  # имя, млн параметров, бедро (вырезка малого вертела, 150 снимков), предметы (позвоночник, 99 снимков), цвет
    ("DenseNet121, рентгенограммы", 7.0, 0.860, None, OURS), ("EfficientNet-B0", 5.3, 0.804, 0.801, LIT), ("DenseNet121, ImageNet", 8.0, 0.815, None, LIT),
    ("ResNet18, ImageNet", 11.7, 0.824, 0.783, OURS), ("ResNet18, object-CXR", 11.7, None, 0.855, OURS),
    ("RegNetY-032", 17.9, 0.795, None, LIT), ("EfficientNetV2-S", 20.2, 0.819, 0.766, LIT), ("DINOv2 ViT-S/14", 22.1, 0.702, 0.649, LIT),
    ("SE-ResNeXt50", 25.5, 0.807, None, LIT), ("ResNet50", 25.6, 0.821, 0.848, LIT), ("ConvNeXt-Tiny", 27.8, 0.802, 0.836, LIT),
]
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.6), gridspec_kw={"wspace": 0.95})
for axx, col, t, base in ((ax[0], 2, "Ротация бедра, вырезка малого вертела\n150 снимков, 36 с нарушением", 0.824),
                          (ax[1], 3, "Посторонние предметы, позвоночник\n99 снимков, 17 с нарушением", 0.855)):
    pts = sorted([(a[1], a[col], a[0], a[4]) for a in ARCH if a[col] is not None], key=lambda q: q[1])
    axx.axvline(base, color=OURS, lw=1.2, ls=":")
    for i, (par, auc, name, c) in enumerate(pts):
        axx.barh(i, auc - 0.5, left=0.5, color=c, height=0.62)
        axx.text(auc - 0.008, i, f"{auc:.2f}".replace(".", ","), va="center", ha="right", fontsize=13, color="white", fontweight="bold")
    axx.set_yticks(range(len(pts))); axx.set_yticklabels([f"{q[2]} · {str(q[0]).replace('.', ',')} млн" for q in pts], fontsize=12.5)
    axx.set_xlim(0.5, 0.95); axx.set_xlabel("ROC-AUC сети вне обучения"); axx.set_title(t, loc="left", fontsize=14)
    axx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
save("arch.png")
print("рисунки:", sorted(os.listdir(OUT)))
