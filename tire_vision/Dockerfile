FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install fastapi uvicorn pillow

COPY . .

CMD ["uvicorn", "tire_vision.app:app", "--host", "0.0.0.0", "--port", "8000"]