# Serves the MCP server over streamable HTTP (Cloud Run sets $PORT).
# Build AFTER running the pipeline locally: the DuckDB file is baked into the image read-only.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VOC_EMBED_BACKEND=tfidf \
    VOC_DB=data/voc.duckdb

COPY pyproject.toml ./
COPY voc ./voc
RUN pip install --no-cache-dir .

COPY data/products.yaml data/voc.duckdb ./data/

EXPOSE 8080
CMD ["sh", "-c", "voc serve mcp --transport http --host 0.0.0.0 --port ${PORT:-8080}"]
