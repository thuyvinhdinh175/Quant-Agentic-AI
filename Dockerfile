FROM python:3.12-slim

WORKDIR /app

# Install Ollama
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && \
    uv pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Pull the DeepSeek-R1 model (this will happen at build time)
RUN ollama serve & sleep 5 && ollama pull deepseek-r1:7b

# Expose the port used by the MCP server
EXPOSE 8000

# Set the entrypoint
CMD ["sh", "-c", "ollama serve & sleep 5 && python server.py"]
