FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TA-Lib (requires special handling)
RUN pip install --no-cache-dir talib-binary

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p data/logs data/memory data/templates data/reports

# Pull the DeepSeek-R1 model (this will happen at build time)
RUN ollama serve & sleep 5 && ollama pull deepseek-r1:7b

# Expose the ports used by Streamlit and the MCP server
EXPOSE 8000 8501

# Set the entrypoint
CMD ["sh", "-c", "ollama serve & sleep 5 && streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 & python server.py"]
