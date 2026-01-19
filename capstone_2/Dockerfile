FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY predict.py .
COPY serve.py .
COPY astana_price_model.pkl .
COPY form.html .

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:9696/health')" || exit 1

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]
