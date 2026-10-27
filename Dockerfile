FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml config.yaml ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]" && pytest -q
ENTRYPOINT ["python", "-m", "ppg"]
CMD ["demo"]
