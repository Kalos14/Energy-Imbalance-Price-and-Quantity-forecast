FROM python:3.11-slim

# Do not write .pyc files and disable stdout buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install minimal system dependencies needed for xgboost, catboost and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Ensure input/output folders exist inside the container
RUN mkdir -p Datasets output

# Default command: run the main forecasting pipeline
CMD ["python", "Main.py"]
