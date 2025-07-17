FROM python:3.9
RUN apt-get update && apt-get install -y git libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .
COPY external/ external/

RUN pip install torch torchvision --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install pillow==9.5.0 numpy==1.26 --no-cache-dir

COPY . .

RUN --mount=type=secret,id=envfile \
    sh -c 'set -e && set -a && . /run/secrets/envfile && set +a && python -m scripts.init'

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]