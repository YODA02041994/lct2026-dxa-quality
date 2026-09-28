# Обучение и дообучение модели

Все шаги воспроизводимы скриптами; случайные числа зафиксированы. Проверка — кросс-валидация по исследованиям
(снимки одного пациента не попадают в обучение и проверку одновременно), 95 % ДИ — бутстреп по исследованиям.

## 1. Данные

| Источник | Объём | Назначение |
|---|---|---|
| Набор постановщика | 100 исследований, 499 файлов, 252 уникальных снимка | обучение и проверка |
| Экспертная разметка постановщика | 5 критериев, истина — колонки критериев | метки нарушений |
| Разметка ориентиров командой | 252 снимка, 8 и 11 ориентиров, согласованность врачей 2,9 мм | обучение локализатора |
| object-CXR (CC BY-NC 4.0) | 9000 рентгенограмм грудной клетки, половина с предметами | предобучение сетей по предметам |

```bash
LCT_PASSWORD='…' scripts/download_data.sh && python scripts/unpack_data.py
PYTHONPATH=src python scripts/make_manifest.py            # data/work/manifest.csv: область, сторона, метки
```

Разметка ориентиров выполнялась в веб-инструменте `tools/labeler/` (протокол — `docs/06_Протокол_разметки.md`).

## 2. Порядок обучения

```bash
# 1. локализатор ориентиров (5 блоков + финальная сеть), ~20 мин на GPU
PYTHONPATH=src python scripts/train_landmarks.py --kind spine --epochs 80
PYTHONPATH=src python scripts/train_landmarks.py --kind hip --epochs 80

# 2. предобучение на object-CXR (торрент academictorrents, уменьшение до 320 px — tools/objectcxr_prepare.py)
PYTHONPATH=src python experiments/exp12_objectcxr_pretrain.py --data train320,dev320 --csv train.csv,dev.csv --epochs 6
PYTHONPATH=src python experiments/exp13_objectcxr_heatmap.py --epochs 6          # карта предметов

# 3. свёрточные сети по критериям (каждая команда даёт веса и вероятности вне обучения)
P=weights/pretrain_objectcxr_full320_r18.pt
for s in 0 1 2; do PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion spine_artifact --init $P --seed $s --tag r18_320_ocxrF_s$s; done
PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion hip_positioning_rotation --arch efficientnet_b0 --tag eff_320
PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion hip_any --tag r18_320
PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion hip_positioning_rotation --crop lt --half 50 --size 224 --tag lt100
PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion hip_positioning_rotation --crop lt --half 50 --size 224 --epochs 40 --tag lt100_e40
PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion hip_positioning_rotation --crop isch --half 50 --size 224 --tag isch100
PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion hip_positioning_rotation --crop prox --half 85 --size 288 --tag prox170
PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion hip_any --crop lt --half 50 --size 224 --tag any_lt100

# 4. логистические регрессии по критериям, пороги, отчёт
PYTHONPATH=src python scripts/train_criteria.py           # → weights/criteria.json
PYTHONPATH=src python scripts/make_report.py              # → docs/metrics.md
```

Состав ансамблей задаётся в одном месте — `src/dxaqc/cnn.py: CNN_SOURCES`; его читают и обучение, и инференс.

## 3. Правила, нарушение которых портит результат

- без поворотов в аугментации сетей для критерия оси; отражение по горизонтали — только со сменой стороны бедра;
- чёрные маски по краям бедра ставит лаборант, признаком они не являются;
- вероятности сетей подаются в регрессию только как вероятности вне обучения (иначе утечка);
- истина — колонки критериев; сколиоз, перелом, эндопротез — норма.

## 4. Дообучение на новых данных

1. Положить новые исследования и разметку в `data/work/`, перестроить манифест.
2. При смене аппарата разметить ориентиры на 50–100 снимках в `tools/labeler/` и дообучить локализатор.
3. Повторить шаги 3–4; сравнить `docs/metrics.md` до и после.
4. Снимки, где модель и эксперт расходятся (`experiments/exp10_label_audit.py`), передать на сверку.

## 5. Что проверено и отклонено

PatchCore, RAD-DINO, RadImageNet, атлас нормального бедра, индекс запирательного отверстия, мета-классификатор,
признак контралатерального бедра, вырезка шейки, сети с входом 512 px для предметов — цифры в `docs/08_Рисерч_что_ещё.md`.
