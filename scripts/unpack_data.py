#!/usr/bin/env python3
"""Распаковать data/raw/*.zip в data/work/ с правильными кириллическими именами.

Архивы организатора собраны без флага UTF-8, поэтому обычный `unzip` даёт `????`.
Запуск:  python scripts/unpack_data.py
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
WORK = os.path.join(ROOT, "data", "work")


def _decode(name: str) -> str:
    for enc in ("utf-8", "cp866"):
        try:
            return name.encode("cp437").decode(enc)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return name


def unpack(zip_name: str, subdir: str) -> int:
    src = os.path.join(RAW, zip_name)
    dst = os.path.join(WORK, subdir)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    n = 0
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            name = _decode(info.filename)
            target = os.path.join(dst, name)
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(zf.read(info))
            n += 1
    return n


def main() -> int:
    os.makedirs(WORK, exist_ok=True)
    total = 0
    for zip_name, subdir in (("train.zip", "train"), ("test.zip", "test")):
        if not os.path.exists(os.path.join(RAW, zip_name)):
            print(f"нет {zip_name} в data/raw — пропускаю", file=sys.stderr)
            continue
        n = unpack(zip_name, subdir)
        print(f"{zip_name} → data/work/{subdir}: {n} файлов")
        total += n
    xlsx = os.path.join(RAW, "разметка.xlsx")
    if os.path.exists(xlsx):
        shutil.copy(xlsx, os.path.join(WORK, "разметка.xlsx"))
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
