FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8005

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models reports data/processed scripts \
    && chmod +x /app/scripts/entrypoint.sh

EXPOSE 8005

ENTRYPOINT ["/app/scripts/entrypoint.sh"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]