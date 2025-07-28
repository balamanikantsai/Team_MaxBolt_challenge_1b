FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (for local LLM)
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models to ensure offline operation
# Note: In a real scenario, these models would be pre-downloaded to meet the size constraint
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY main.py .
COPY challenge-1b.py .

# Make main.py executable
RUN chmod +x main.py

# Start Ollama service and download model (in practice, this would be pre-done)
# RUN ollama serve & sleep 5 && ollama pull llama2:7b-chat

# Set the entry point
CMD ["python", "main.py"]

