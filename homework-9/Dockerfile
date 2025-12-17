FROM agrigorev/model-2025-hairstyle:v1

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./predict_hair_type.py .

# Syntax: [filename].[handler_function_name]
CMD ["predict_hair_type.lambda_handler"]