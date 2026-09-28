# Руководство по развёртыванию

## 1. Контейнер (основной способ)

```bash
git clone https://github.com/YODA02041994/lct2026-dxa-quality && cd lct2026-dxa-quality
scripts/download_weights.sh          # 15 файлов, 0,8 ГБ, нужен интернет; curl или авторизованный gh
docker build -t dxaqc .
docker run --rm dxaqc python -c "from dxaqc.predict import load_default_predictor as l; print(l().name)"   # → landmarks
docker run --rm -v "$PWD/in:/in" -v "$PWD/out:/out" dxaqc run -i /in -o /out
docker run -d --name dxaqc -p 8000:8000 --restart unless-stopped dxaqc       # сервис
```

Проверка: `curl http://localhost:8000/health` → `{"status":"ok","model":"landmarks","weights_loaded":true}`.

## 2. Машина без интернета

На машине с интернетом: `docker save dxaqc | gzip > dxaqc.tar.gz`. На целевой: `docker load < dxaqc.tar.gz`.
Образ содержит код, зависимости и веса; при работе сеть не используется (можно запускать с `--network none`).

## 3. Сборка под другую архитектуру

`docker build --platform linux/amd64 -t dxaqc:amd64 .` — сборка x86-64 на ARM-машине. GPU не требуется; для CUDA замените в
`Dockerfile` индекс колёс PyTorch на `https://download.pytorch.org/whl/cu121` и запускайте с `--gpus all`.

## 4. Сервер без Docker

```bash
python3.12 -m venv /opt/dxaqc/venv
/opt/dxaqc/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
/opt/dxaqc/venv/bin/pip install -r requirements-docker.txt
# код → /opt/dxaqc/src, веса → /opt/dxaqc/weights
PYTHONPATH=/opt/dxaqc/src /opt/dxaqc/venv/bin/uvicorn dxaqc.api:app --host 127.0.0.1 --port 8890
```

Служба systemd: `WorkingDirectory=/opt/dxaqc`, `Environment=PYTHONPATH=/opt/dxaqc/src`, `Restart=on-failure`, `MemoryMax=5G`.

## 5. Обратный прокси с префиксом

```nginx
location ^~ /dxa/ {
    proxy_pass http://127.0.0.1:8890/;      # со слэшем: префикс снимается
    proxy_read_timeout 900s;
    client_max_body_size 300m;
}
```

Сервис запускается с `--root-path /dxa`; страница обращается к API по относительным адресам.

## 6. Настройки

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `DXAQC_WEIGHTS` | `<проект>/weights` | папка весов |
| `DXAQC_MODE` | `competition` | пороги: `competition` или `sensitive` |
| `DXAQC_NETS_PER_FILE` | все | `1` — экономный режим для 2 ядер и 4 ГБ на процесс |
| `DXAQC_RUNS` | временная папка | где хранить загруженные файлы и результаты |
| `DXAQC_RUN_TTL_MIN` | 180 | срок хранения прогона, минут |
| `DXAQC_MAX_UPLOAD_MB` | 500 | предел размера загрузки |
| `OMP_NUM_THREADS` | 4 | число потоков вычислений |

## 7. Проверка после установки

```bash
python -m pytest -q                                           # 13 тестов
docker run --rm -v "$PWD/data/work/test:/in" -v "$PWD/out:/out" dxaqc run -i /in -o /out    # 3 файла, 0 Failure
```

## 8. Частые проблемы

| Симптом | Причина и решение |
|---|---|
| `/health` → `weights_loaded: false` | нет файлов в `weights/`; выполнить `scripts/download_weights.sh` и пересобрать образ |
| служба завершается, в журнале `oom-kill` | мало памяти; `DXAQC_NETS_PER_FILE=1` или поднять `MemoryMax` |
| 404 на всех адресах за прокси | префикс передан дважды; в `proxy_pass` нужен завершающий слэш |
| порт занят | сменить `--port`, проверить `ss -ltnp` |
