FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create directories for models and configs
RUN mkdir -p /app/models /app/configs /app/logs \
    /app/models/breast_cancer \
    /app/models/parkinsons \
    /app/models/reinopath \
    /app/global_models \
    /app/model_backups

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python code
COPY *.py .
COPY models/ models/
COPY training/ training/
COPY utils/ utils/

# Copy configuration files
COPY *.json .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the server
CMD ["python", "main.py", "--mode", "server", "--config", "/app/configs/server_config.json"]