FROM python:3.11-slim

#explain here
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* README.md /app/

ARG INSTALL_DEV=true

RUN if [ "$INSTALL_DEV" = "true" ]; then \
    uv pip install --system -e .; \
    else \
    uv pip install --system --no-deps -e .; \
    fi

RUN python -m nltk.downloader punkt punkt_tab

COPY . /app/

EXPOSE 8501

CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
