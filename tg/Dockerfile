FROM python:3.9-slim

WORKDIR /app

RUN pip install telegram-bot-api

COPY . .

CMD ["python", "logic.py"]
