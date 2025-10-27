FROM agrigorev/zoomcamp-model:2025
WORKDIR /app

COPY ./predict.py /app/predict.py
COPY ./pipeline_v1.bin /app/pipeline_v1.bin

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv lock
RUN uv sync --locked

EXPOSE 9797
WORKDIR /app
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9797"]