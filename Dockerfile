FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src:/app PYTHONUNBUFFERED=1

# LightGBM needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY app ./app

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
