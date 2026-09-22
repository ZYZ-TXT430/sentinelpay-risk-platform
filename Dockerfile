FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m sentinelpay.pipeline --n-transactions 20000
EXPOSE 8000
CMD ["uvicorn", "sentinelpay.api:app", "--host", "0.0.0.0", "--port", "8000"]
