FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY predict.py .
COPY serve.py .
COPY midterm_model.bin/ /app/midterm_model.bin
COPY form.html/ /app/form.html

# Expose port
EXPOSE 9696

# Run the application
CMD ["python", "serve.py"]
