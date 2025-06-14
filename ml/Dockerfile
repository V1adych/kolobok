FROM python:3.9-slim

WORKDIR /app

COPY build.sh .
RUN ./build.sh

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]