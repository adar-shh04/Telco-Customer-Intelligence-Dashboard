# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose ports for both Streamlit and FastAPI
EXPOSE 8501 8000

# Default command (can be overridden in docker-compose)
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
