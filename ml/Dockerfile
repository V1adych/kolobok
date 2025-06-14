FROM python:3.9-slim
RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY build.sh .
RUN ./build.sh

COPY . .

RUN echo "Sanity check"
RUN python run.py test_input.png

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]