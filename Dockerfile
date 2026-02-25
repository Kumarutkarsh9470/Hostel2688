FROM python:3.11-slim

# System deps for scipy / torch CPU
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the backend needs
COPY backend/        backend/
COPY training/bdh.py training/bdh.py
COPY checkpoints/    checkpoints/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
