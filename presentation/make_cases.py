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
    ("052_spine", "Ось отклонена на 9°", "система: не выровнена ось → переснять", "эксперт: нарушение оси"),
    ("079_spine", "Предметы и укладка", "система: предметы, укладка → переснять", "эксперт: предметы, укладка"),
    ("011_spine", "Сколиоз", "система: деформация 28°, укладка в норме", "эксперт: «сколиоз, не верная разметка»"),
    ("023_spine", "Укладка у порога", "система: проверить укладку (0,55 при пороге 0,76)", "эксперт: нарушение укладки"),
    ("059_hipL", "Бедро", "система: укладка бедра → переснять; поле ROI у порога", "эксперт: поле ROI"),
    ("001_spine", "Норма", "система: в норме; пограничная асимметрия 8°", "эксперт: норма"),
]
CARD_W, CAP_H, PAD = 760, 118, 14


def card(stem, title, sys_line, exp_line):
    im = Image.open(os.path.join(SRC, f"{stem}.png")).convert("RGB")
    im = im.resize((CARD_W, int(im.height * CARD_W / im.width)), Image.LANCZOS)
    out = Image.new("RGB", (CARD_W, im.height + CAP_H), "white")
    out.paste(im, (0, CAP_H))
    d = ImageDraw.Draw(out)
    d.text((PAD, 10), title, font=ImageFont.truetype(FONT_B, 30), fill=(0, 0, 0))
    d.text((PAD, 50), sys_line, font=ImageFont.truetype(FONT, 24), fill=(0x99, 0x3C, 0x1D) if "переснять" in sys_line else (0x08, 0x50, 0x41) if "в норме" in sys_line else (0x85, 0x4F, 0x0B))
    d.text((PAD, 82), exp_line, font=ImageFont.truetype(FONT, 24), fill=(90, 90, 90))
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
