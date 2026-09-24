#!/usr/bin/env python3
"""На vps: распаковать object-CXR zip и уменьшить снимки до 320 px по длинной стороне (JPEG q90) — чтобы перенести на Мак.
    python3 tools/objectcxr_prepare.py dev.zip dev320 [size]
"""
import io, os, sys, zipfile
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]; size = int(sys.argv[3]) if len(sys.argv) > 3 else 320
os.makedirs(dst, exist_ok=True); n = 0
with zipfile.ZipFile(src) as zf:
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith((".jpg", ".jpeg", ".png")): continue
        im = Image.open(io.BytesIO(zf.read(info))).convert("L")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(os.path.join(dst, os.path.splitext(os.path.basename(info.filename))[0] + ".jpg"), quality=90); n += 1
print("готово:", n, "снимков →", dst)
