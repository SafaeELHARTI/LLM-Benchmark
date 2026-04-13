FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir "numpy<2"

RUN pip install --no-cache-dir torch==2.4.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 6006

CMD ["python", "benchmark/run_benchmark.py"]