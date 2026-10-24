FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY immicheye ./immicheye
COPY scripts ./scripts
RUN pip install --no-cache-dir . && useradd --system --uid 1000 immicheye && mkdir /data && chown immicheye /data

USER immicheye
ENV IMMICHEYE_DATA=/data PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/immicheye/status')"
CMD ["uvicorn", "immicheye.server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
