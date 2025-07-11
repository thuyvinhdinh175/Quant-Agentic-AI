# Financial Analyst DeepSeek

A powerful AI-driven financial analysis system that processes natural language queries about stocks and generates detailed visualizations and insights.

## Architecture Overview

```mermaid
graph TD
    A[User] -->|Natural Language Query| B[MCP Server]
    B -->|Parse Query| C[Query Parser Agent]
    C -->|Extracted Data| D[Code Writer Agent]
    D -->|Python Code| E[Code Execution Agent]
    E -->|Execute & Validate| F[Stock Data Visualization]
    F -->|Result| B
    B -->|Final Result| A
    
    G[yfinance API] -->|Stock Data| E
    H[Ollama - DeepSeek-R1] -->|LLM Capabilities| C
    H -->|LLM Capabilities| D
    H -->|LLM Capabilities| E
```

## Features

- **Natural Language Interface**: Ask questions about stocks in plain English
- **Multi-Agent Architecture**: Specialized AI agents for different tasks
- **Local LLM Integration**: Powered by DeepSeek-R1 running via Ollama
- **Automated Code Generation**: Generates Python code for stock analysis
- **Data Visualization**: Creates charts and graphs for stock performance
- **Flexible Analysis**: Handles various timeframes and comparison scenarios

## Prerequisites

- Python 3.12 or later
- Ollama (for running DeepSeek-R1 locally)
- Docker and Docker Compose (for containerized deployment)

## Quick Start (Docker)

The easiest way to run the project is using Docker:

```bash
# Clone the repository
git clone <repository-url>
cd financial-analyst-deepseek

# Build and start the Docker container
docker-compose up --build -d

# Check logs
docker-compose logs -f
```

## Manual Setup

If you prefer to run the project without Docker:

1. **Install Ollama**

```bash
# For Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the DeepSeek-R1 model
ollama pull deepseek-r1:7b
```

2. **Install Dependencies**

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Configure MCP Server in Cursor IDE**

- Go to Cursor settings
- Select MCP 
- Add new global MCP server with the following configuration:

```json
{
    "mcpServers": {
        "financial-analyst": {
            "command": "uv",
            "args": [
                "--directory",
                "absolute/path/to/project_root",
                "run",
                "server.py"
            ]
        }
    }
}
```

4. **Run the Server**

```bash
python server.py
```

## Usage Examples

Ask the system financial questions like:

- "Show me Tesla's stock performance over the last 3 months"
- "Compare Apple and Microsoft stocks for the past year"
- "Analyze the trading volume of Amazon stock for the last month"
- "Plot YTD stock gain of Tesla"
- "Show me the price-to-earnings ratio for NVIDIA"

## How It Works

1. **Query Parsing**: The system interprets your natural language query to extract stock symbols, timeframes, and analysis actions.
2. **Code Generation**: Based on the parsed query, it generates Python code using yfinance, pandas, and matplotlib.
3. **Execution & Validation**: The generated code is executed and validated to ensure it works correctly.
4. **Visualization**: The system creates charts and graphs to visualize the requested financial data.

## Project Structure

- `server.py`: MCP server implementation that handles incoming queries
- `finance_crew.py`: Multi-agent system using CrewAI for financial analysis
- `Dockerfile` & `docker-compose.yml`: Containerization configuration
- `pyproject.toml`: Project dependencies and configuration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contribution

Contributions are welcome! Please open an issue or submit a pull request with your improvements.
