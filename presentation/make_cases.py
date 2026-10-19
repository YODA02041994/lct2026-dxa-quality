#!/usr/bin/env python3
"""Монтаж кейсов для слайда «Кейсы»: 6 снимков с разметкой конвейера (overlays) и русскими подписями
«система → эксперт». Снимки — обучающий набор (вердикты получены конвейером 16.10 после доработки).
    python presentation/make_cases.py   → presentation/img/cases2.jpg
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "img", "cases_src")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
CASES = [
    ("052_spine", "Ось отклонена", "система: не выровнена ось (наклон 6,1°) → переснять", "эксперт: нарушение оси"),
    ("005_spine", "Предметы при сколиозе", "система: предметы → переснять; деформация около 15° отдельно", "эксперт: предметы; «сколиоз»"),
    ("011_spine", "Сколиоз", "система: деформация 28°, укладка в норме (на стенде с одной сетью — «проверить»)", "эксперт: норма; «сколиоз, не верная разметка»"),
    ("023_spine", "Укладка у порога", "система: проверить укладку (0,55 при пороге 0,76)", "эксперт: нарушение укладки"),
    ("059_hipL", "Бедро: расхождение", "система: укладка бедра → переснять; поле у порога", "эксперт: поле сканирования"),
    ("001_spine", "Норма", "система: в норме", "эксперт: норма"),
]
CARD_W, PAD, LINE = 760, 14, 30
F_TITLE, F_TEXT = ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 24)


def wrap(text, font, width):
    """Перенос по словам: подпись не обрезается по краю карточки."""
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if font.getlength(t) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def caption_h(sys_line, exp_line):
    w = CARD_W - 2 * PAD
    return 50 + LINE * (len(wrap(sys_line, F_TEXT, w)) + len(wrap(exp_line, F_TEXT, w))) + 8


CAP_H = max(caption_h(c[2], c[3]) for c in CASES)          # одна высота подписи у всех карточек — ряды ровные


def card(stem, title, sys_line, exp_line):
    im = Image.open(os.path.join(SRC, f"{stem}.png")).convert("RGB")
    im = im.resize((CARD_W, int(im.height * CARD_W / im.width)), Image.LANCZOS)
    out = Image.new("RGB", (CARD_W, im.height + CAP_H), "white")
    out.paste(im, (0, CAP_H))
    d = ImageDraw.Draw(out)
    d.text((PAD, 10), title, font=F_TITLE, fill=(0, 0, 0))
    color = (0x99, 0x3C, 0x1D) if "переснять" in sys_line else (0x08, 0x50, 0x41) if "в норме" in sys_line else (0x85, 0x4F, 0x0B)
    y = 50
    for ln in wrap(sys_line, F_TEXT, CARD_W - 2 * PAD):
        d.text((PAD, y), ln, font=F_TEXT, fill=color); y += LINE
    for ln in wrap(exp_line, F_TEXT, CARD_W - 2 * PAD):
        d.text((PAD, y), ln, font=F_TEXT, fill=(90, 90, 90)); y += LINE
    return out


cards = [card(*c) for c in CASES]
gap = 24
for name, grp in (("cases2a.jpg", cards[:3]), ("cases2b.jpg", cards[3:])):     # по три карточки в ряд — читается с экрана
    h = max(c.height for c in grp)
    sheet = Image.new("RGB", (3 * CARD_W + 2 * gap, h), "white")
    for i, c in enumerate(grp):
        sheet.paste(c, (i * (CARD_W + gap), 0))
    sheet.save(os.path.join(HERE, "img", name), quality=88)
    print(name, sheet.size)
