# ── Base image ─────────────────────────────────────────
# Official Python 3.12 on lightweight Linux (not Windows)
FROM python:3.11-slim

# ── Set working directory inside container ──────────────
WORKDIR /app

# ── Install system dependencies ─────────────────────────
# These are needed by PyMuPDF (PDF reader)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (Docker cache trick) ────────
# If requirements.txt didn't change, Docker skips reinstall
# This makes rebuilds much faster
COPY requirements.txt .

# ── Install Python packages ─────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy rest of project ─────────────────────────────────
COPY . .

# ── Create data directory ────────────────────────────────
RUN mkdir -p data

# ── Expose port ──────────────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]