FROM python:3.11-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY model.py .
COPY main.py .
COPY onnx_session.py .
RUN mkdir /app/train_model
COPY train_model/room_classifier_final.onnx/ /app/train_model/room_classifier_final.onnx
COPY train_model/room_classifier_final.onnx.data/ /app/train_model/room_classifier_final.onnx.data

# Expose port
EXPOSE 9696

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]
