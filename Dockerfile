# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create Python 3.11 virtual environment
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/venv"

# Upgrade pip in venv
RUN pip install --upgrade pip setuptools wheel

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Make docker-entrypoint.sh executable
RUN chmod +x /app/docker-entrypoint.sh

# Create a non-root user for running the application (optional but recommended)
RUN useradd -m -u 1000 sbtruser && chown -R sbtruser:sbtruser /app
USER sbtruser

# Set entrypoint script for proper environment setup
CMD ["python", "--version"]
