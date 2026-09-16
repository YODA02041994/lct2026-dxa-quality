#!/usr/bin/env bash
# Скачать материалы организатора (ТЗ, датасеты, шаблон презентации) с mostech-cloud.mos.ru.
# Пароль к материалам — в чате задачи в Telegram (сообщение «ТЗ И ДАТАСЕТЫ // СТАРТ РАБОТЫ», 14.09).
#
#   LCT_PASSWORD='...' scripts/download_data.sh          # или без переменной — спросит
#   python scripts/unpack_data.py                          # затем распаковать в data/work/
#
# В git эти файлы не хранятся: репозиторий к сдаче публичный, а материалы выданы под пароль.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
MAT="$ROOT/docs/materials"
BASE="https://mostech-cloud.mos.ru/public.php/webdav"

if [ -z "${LCT_PASSWORD:-}" ]; then
  read -r -s -p "Пароль к материалам (из чата задачи): " LCT_PASSWORD; echo
fi

mkdir -p "$RAW" "$MAT"

# токен шары | имя файла на сервере (url-encoded) | куда положить
ITEMS=(
  "JTeXCzE3MH35ga9|%D0%9D%D0%94_%D0%B4%D0%BB%D1%8F_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F.zip|$RAW/train.zip"
  "JTeXCzE3MH35ga9|%D0%94%D0%BB%D1%8F%20%D1%82%D0%B5%D1%81%D1%82%D0%B0.zip|$RAW/test.zip"
  "TAQmECyfqBpAjxH|4.%20%D0%94%D0%B5%D0%BF%D0%97%D0%B4%D1%80%D0%B0%D0%B2.pdf|$MAT/ТЗ_Задача4_ДепЗдрав.pdf"
  "bnxfqRnTp3c7eke|%D0%9B%D0%A6%D0%A22026%20%D0%A8%D0%B0%D0%B1%D0%BB%D0%BE%D0%BD%20%D0%BF%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D0%B8.pptx|$MAT/ЛЦТ2026_Шаблон_презентации.pptx"
)

for item in "${ITEMS[@]}"; do
  IFS='|' read -r token remote dest <<<"$item"
  if [ -s "$dest" ]; then
    echo "есть: $(basename "$dest")"
    continue
  fi
  echo "качаю: $(basename "$dest")"
  curl -fsSL --retry 3 --max-time 900 -u "$token:$LCT_PASSWORD" "$BASE/$remote" -o "$dest.part" \
    || { echo "ОШИБКА: $(basename "$dest") — проверьте пароль/сеть"; rm -f "$dest.part"; exit 1; }
  mv "$dest.part" "$dest"
done

# разметка отдельно, чтобы не лазить в архив
if [ ! -s "$RAW/разметка.xlsx" ]; then
  python3 - "$RAW" <<'EOF'
import sys, zipfile, os
raw = sys.argv[1]
with zipfile.ZipFile(os.path.join(raw, "train.zip")) as z:
    for info in z.infolist():
        if info.filename.lower().endswith(".xlsx"):
            open(os.path.join(raw, "разметка.xlsx"), "wb").write(z.read(info))
            print("извлечена разметка.xlsx")
            break
EOF
fi

# текст ТЗ для grep/чтения (если есть pdftotext)
if command -v pdftotext >/dev/null 2>&1 && [ ! -s "$ROOT/docs/TZ_full.txt" ]; then
  pdftotext -layout "$MAT/ТЗ_Задача4_ДепЗдрав.pdf" "$ROOT/docs/TZ_full.txt" && echo "docs/TZ_full.txt готов"
fi

echo "готово: $(du -sh "$RAW" | cut -f1) в data/raw, $(du -sh "$MAT" | cut -f1) в docs/materials"
