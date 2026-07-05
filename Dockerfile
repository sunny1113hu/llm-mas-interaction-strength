FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv ${UV_PROJECT_ENVIRONMENT}
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

COPY pyproject.toml uv.lock* /app/
RUN uv sync --frozen || uv sync

COPY . /app

ENV PYTHONPATH=/app

CMD ["python", "-m", "scripts.run_one", "--config", "configs/default.yaml"]
