FROM python:3.8-slim

WORKDIR /app

# System libraries required by RDKit + git for pip GitHub installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 1.8.1 CPU-only (no conda, avoids OOM during build)
RUN pip install --no-cache-dir \
    torch==1.8.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# RDKit via pip wheel (no conda needed)
RUN pip install --no-cache-dir rdkit-pypi==2022.9.3

# Remaining Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code and Chemformer assets from repo
COPY . /app

ENV PORT=7860
ENV PROJECT_DIR=/app
EXPOSE 7860

CMD ["python", "app.py"]
