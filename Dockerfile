# Образ для проверки организатором: CPU-only, без сети при работе. Веса кладутся в образ на этапе сборки.
#   docker build -t dxaqc .                              (веса должны лежать в ./weights — scripts/download_weights.sh)
#   docker run --rm -v "$PWD/in:/in" -v "$PWD/out:/out" dxaqc run -i /in -o /out
#   docker run --rm -p 8000:8000 dxaqc serve             → http://localhost:8000 (страница) и /docs (Swagger)
#   docker save dxaqc | gzip > dxaqc.tar.gz              (для передачи без реестра)
# базовый образ зафиксирован по контрольной сумме (python 3.12-slim, Debian bookworm)
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 DXAQC_WEIGHTS=/app/weights PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# точные версии всех библиотек — requirements.lock; PyTorch для CPU ставится из своего индекса
COPY requirements.lock ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 \
 && pip install -r requirements.lock
COPY src ./src
COPY weights ./weights
COPY README.md ./
COPY docker/entrypoint.sh /usr/local/bin/dxaqc
RUN chmod +x /usr/local/bin/dxaqc
EXPOSE 8000
ENTRYPOINT ["dxaqc"]
CMD ["serve"]
