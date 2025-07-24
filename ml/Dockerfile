FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

RUN --mount=type=secret,id=envfile \
    sh -c 'set -e && set -a && . /run/secrets/envfile && set +a && python -m scripts.init'

ENV GUNICORN_WORKERS=4

CMD sh -c "gunicorn app:app -k uvicorn.workers.UvicornWorker --workers $GUNICORN_WORKERS --bind 0.0.0.0:8000"
