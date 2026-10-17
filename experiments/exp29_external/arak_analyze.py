import csv, json, os, sys, random, collections
from PIL import Image, ImageDraw
res, png_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out_dir, exist_ok=True)
rows = list(csv.DictReader(open(res, encoding="utf-8-sig"), delimiter=";"))
def kind(r):
    if r["processing_status"] != "Success": return "не поддерживается"
    return "позвоночник" if r["anatomical_region"].startswith("Пояс") else "бедро"
def suffix(name):
    base = os.path.splitext(name)[0]; return base.rsplit("-", 1)[-1] if "-" in base else "без суффикса"
def shape(name):
    im = Image.open(os.path.join(png_dir, os.path.splitext(name)[0] + ".png")); w, h = im.size
    return "высокий кадр (боковой)" if h > 2 * w else "обычный"
tab = collections.Counter(); kinds = collections.Counter(); shp = collections.Counter(); groups = collections.defaultdict(list)
err = collections.Counter()
for r in rows:
    k = kind(r); kinds[k] += 1; s = shape(r["path_to_study"]); shp[(s, k)] += 1
    tab[(suffix(r["path_to_study"]) if suffix(r["path_to_study"]) in ("1", "2", "3", "4") else "прочие", k)] += 1
    groups[k].append(r["path_to_study"])
    if k == "не поддерживается": err[(r["error_message"] or "")[:60]] += 1
print("всего:", len(rows), "| итог:", dict(kinds))
print("по суффиксу имени × решение:", {f"{a}:{b}": c for (a, b), c in sorted(tab.items())})
print("по форме кадра × решение:", {f"{a}:{b}": c for (a, b), c in sorted(shp.items())})
print("причины «не поддерживается»:", err.most_common(3))
lv = collections.Counter((kind(r), r["level"]) for r in rows if r["processing_status"] == "Success" and not r["duplicate_of"])
print("уровни:", {f"{a}:{b}": c for (a, b), c in sorted(lv.items())})
ty = collections.Counter(t for r in rows if r["processing_status"] == "Success" and not r["duplicate_of"] for t in (r["violation_type"] or "").split(";") if t)
print("типы нарушений:", dict(ty))
random.seed(29)
for k, names in groups.items():
    pick = random.sample(names, min(40, len(names))); W, H = 110, 140
    sheet = Image.new("RGB", (10 * W, 4 * H), "white"); dr = ImageDraw.Draw(sheet)
    for i, f in enumerate(pick):
        im = Image.open(os.path.join(png_dir, os.path.splitext(f)[0] + ".png")).convert("RGB"); im.thumbnail((W - 4, H - 4))
        sheet.paste(im, ((i % 10) * W + 2, (i // 10) * H + 2))
    name = {"позвоночник": "spine", "бедро": "hip", "не поддерживается": "unsupported"}[k]
    sheet.save(os.path.join(out_dir, f"sheet_{name}.jpg"), quality=55); print("лист:", name, len(pick))
