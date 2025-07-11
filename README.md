# Quant Agentic AI

A powerful agentic AI system for quantitative financial analysis that processes natural language queries about stocks and generates detailed visualizations and insights using a multi-agent architecture powered by DeepSeek-R1.


**Key Capabilities:**

* Transform natural language questions into comprehensive financial analyses
* Generate and execute Python code tailored to specific financial queries
* Create interactive visualizations of stock performance and technical indicators
* Provide technical analysis using industry-standard indicators and patterns
* Run completely locally, ensuring data privacy and eliminating API costs
* Process and analyze multiple data sources for holistic market insights

Built on DeepSeek-R1 and running via Ollama, this project demonstrates how agentic AI systems can deliver specialized domain expertise while maintaining full local execution. The system's modular design allows for continuous enhancement with new capabilities like sentiment analysis, risk assessment, and PDF report generation.

## 🏗️ System Architecture

```
┌────────────────────────┐
│ Natural Language Query │
└───────────┬────────────┘
            ↓
┌───────────────────────┐
│ MCP Server & Parsing  │ <──── (CrewAI + Query Parsing)
└────┬──────────┬───────┘
     ↓          ↓
┌────────────┐  ┌────────────────────┐
│ Code Gen   │  │ Multi-Agent System │ <──── (CrewAI + DeepSeek-R1)
└────┬───────┘  └────┬───────────────┘
     ↓               ↓
┌────────────┐  ┌────────────────────┐
│ Execution  │  │ Technical Analysis │
└────┬───────┘  └────┬───────────────┘
     ↓               ↓
┌───────────────────────────┐
│ Visualization & Reporting │
└───────────────────────────┘
```

## 🛠️ Core Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit / Cursor IDE |
| Agent Orchestration | CrewAI |
| LLM Reasoning | DeepSeek-R1 via Ollama |
| Data Retrieval | yfinance / News APIs / SEC Filings |
| Technical Analysis | TA-Lib / Pandas / Matplotlib |
| Memory System | LangChain / Custom Persistence |
| Containerization | Docker / Docker Compose |

## 🔍 Key Features

### ✅ Natural Language Processing
* Transform plain English queries into structured analysis
* Extract stock symbols, timeframes, and analysis type
* Support for comparative analysis across multiple stocks

### ✅ Automated Code Generation
* Dynamic Python code creation for specific analysis needs
* Auto-adaptation to different stock metrics and timeframes
* Intelligent error correction and validation

### ✅ Technical Analysis
* Over 15 common technical indicators (RSI, MACD, Bollinger Bands)
* Pattern recognition and trend analysis
* Volatility and momentum measurements

### ✅ Multi-Agent Collaboration
* Specialized agents for different analysis aspects
* Agent memory for consistent conversation context
* Hierarchical task delegation and result synthesis

### ✅ Visualization & Reporting
* Interactive stock charts and indicator plots
* Comparative multi-stock visualization
* PDF report generation with analysis summaries

## 🚀 Advanced Ideas

| Feature | Description |
|---------|-------------|
| 🧠 Portfolio Analysis | Multi-stock correlation and risk assessment across portfolios |
| 🔁 Backtesting | Test trading strategies against historical market data |
| 📈 Sentiment Integration | News and social media sentiment impact on stock movements |
| 🔮 Market Prediction | Probabilistic forecasting of potential price movements |

## 🗂️ Project Folder Structure

```
Quant-Agentic-AI/
│
├── agents/                # Specialized agent implementations
│   ├── query_parser.py    # Query parsing and extraction
│   ├── code_writer.py     # Python code generation
│   ├── execution.py       # Code execution and validation
│   ├── technical_analysis.py # Technical indicator analysis
│   ├── risk_analysis.py   # Risk assessment and evaluation
│   └── sentiment.py       # News and social sentiment analysis
│
├── utils/                 # Utility modules
│   ├── memory.py          # Agent memory implementation
│   ├── logging.py         # Logging utilities
│   ├── templates.py       # Query templates
│   └── pdf_generator.py   # PDF report generation
│
├── data/                  # Data storage and caching
│   ├── logs/              # System and agent logs
│   └── reports/           # Generated reports
│
├── server.py              # MCP server implementation
├── finance_crew.py        # CrewAI orchestration logic
├── streamlit_app.py       # Streamlit web interface
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container setup
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/thuyvinhdinh175/Quant-Agentic-AI.git
cd Quant-Agentic-AI

# Option 1: Using Docker (recommended)
docker-compose up --build -d

# Option 2: Manual setup
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the DeepSeek-R1 model
ollama pull deepseek-r1:7b

# Install Python dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

## 🚀 Usage

### Command Line / IDE Integration

1. Configure the MCP server in Cursor IDE settings
2. Send natural language queries directly in the IDE
3. View results and generated code inline

### Web Interface

1. Start the Streamlit app: `streamlit run streamlit_app.py`
2. Access the web interface at http://localhost:8501
3. Enter your financial query in the input field
4. Review the generated analysis and visualizations

### Example Queries

```
"Compare the performance of AAPL and MSFT over the past year"
"Show me the RSI indicator for NVDA over the last 3 months"
"Calculate the volatility of TSLA stock since January"
"Plot the moving averages for SPY with 50 and 200 day periods"
```

## 🔧 Configuration

Configuration options can be set in the following locations:

- LLM settings: Configure Ollama parameters in `finance_crew.py`
- Agent settings: Modify agent roles and goals in `finance_crew.py`
- UI customization: Adjust Streamlit settings in `streamlit_app.py`
