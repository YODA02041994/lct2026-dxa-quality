# Образ для проверки организатором: CPU-only, без сети при работе. Веса кладутся в образ на этапе сборки.
#   docker build -t dxaqc .                              (веса должны лежать в ./weights — scripts/download_weights.sh)
#   docker run --rm -v "$PWD/in:/in" -v "$PWD/out:/out" dxaqc run -i /in -o /out
#   docker run --rm -p 8000:8000 dxaqc serve             → http://localhost:8000 (страница) и /docs (Swagger)
#   docker save dxaqc | gzip > dxaqc.tar.gz              (для передачи без реестра)
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 DXAQC_WEIGHTS=/app/weights PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-docker.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 \
 && pip install -r requirements-docker.txt
COPY src ./src
COPY weights ./weights
COPY README.md ./
COPY docker/entrypoint.sh /usr/local/bin/dxaqc
RUN chmod +x /usr/local/bin/dxaqc
EXPOSE 8000
ENTRYPOINT ["dxaqc"]
CMD ["serve"]
