# Credit Risk Scoring System
# Multi-service Docker image for FastAPI backend and Streamlit frontend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and ML pipeline
COPY app/ ./app/
COPY ml/ ./ml/
COPY streamlit_app.py .

# Create data directory for SQLite database
RUN mkdir -p ./data

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Copy and configure startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Run the startup script
CMD ["/start.sh"]