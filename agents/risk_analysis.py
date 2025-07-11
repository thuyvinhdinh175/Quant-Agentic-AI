from crewai import Agent
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np

class RiskMetrics(BaseModel):
    """Model for risk analysis results."""
    symbol: str = Field(..., description="Stock ticker symbol")
    volatility: float = Field(..., description="Historical volatility (standard deviation of returns)")
    beta: Optional[float] = Field(None, description="Beta relative to market index")
    var_95: float = Field(..., description="Value at Risk (95% confidence)")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    risk_assessment: str = Field(..., description="Overall risk assessment")

class RiskAnalysisAgent:
    """
    Agent responsible for analyzing risk metrics for stocks and portfolios.
    Provides risk assessment and recommendations based on volatility, drawdowns, and other metrics.
    """
    
    def __init__(self, llm):
        """Initialize the risk analysis agent."""
        self.agent = Agent(
            role="Risk Analysis Specialist",
            goal="Analyze stock and portfolio risk metrics and provide risk assessments",
            backstory="""You are a risk analysis expert with extensive experience in financial markets.
                       You specialize in identifying and quantifying investment risks through various
                       statistical methods and financial models. Your expertise helps investors
                       understand their risk exposure and make informed decisions.""",
            llm=llm,
            verbose=True,
            memory=True,
        )
    
    def generate_risk_analysis_code(self, symbol: str, period: str, benchmark_symbol: str = "SPY") -> str:
        """
        Generate Python code for risk analysis.
        
        Args:
            symbol: The stock ticker symbol
            period: The time period for analysis
            benchmark_symbol: The ticker symbol for the benchmark index (default: SPY)
            
        Returns:
            Python code string for the risk analysis
        """
        return f"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')
benchmark_data = yf.download('{benchmark_symbol}', period='{period}', interval='1d')

# Calculate daily returns
stock_data['Daily Return'] = stock_data['Close'].pct_change()
benchmark_data['Daily Return'] = benchmark_data['Close'].pct_change()

# Align the two DataFrames (handling missing dates)
aligned_data = pd.concat([stock_data['Daily Return'], benchmark_data['Daily Return']], axis=1)
aligned_data.columns = ['{symbol} Returns', 'Market Returns']
aligned_data = aligned_data.dropna()

# ----- Risk Metrics Calculation -----

# Historical Volatility (annualized)
trading_days = 252
daily_volatility = aligned_data['{symbol} Returns'].std()
annual_volatility = daily_volatility * np.sqrt(trading_days)

# Beta calculation
covariance = aligned_data.cov()
beta = covariance.loc['{symbol} Returns', 'Market Returns'] / covariance.loc['Market Returns', 'Market Returns']

# Value at Risk (95% confidence)
var_95 = np.percentile(aligned_data['{symbol} Returns'], 5)
dollar_var_95 = stock_data['Close'].iloc[-1] * abs(var_95)

# Maximum Drawdown
def calculate_max_drawdown(prices):
    max_so_far = prices[0]
    max_drawdown = 0
    for price in prices:
        if price > max_so_far:
            max_so_far = price
        drawdown = (max_so_far - price) / max_so_far
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown

max_drawdown = calculate_max_drawdown(stock_data['Close'].values)

# Sharpe Ratio (annualized, using risk-free rate of 2%)
risk_free_rate = 0.02
annual_return = aligned_data['{symbol} Returns'].mean() * trading_days
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan

# ----- Visualization -----

# Create figure with multiple subplots
fig = plt.figure(figsize=(12, 15))
plt.suptitle(f'Risk Analysis for {symbol} ({period})', fontsize=16)

# 1. Stock Price Plot
ax1 = plt.subplot(4, 1, 1)
ax1.plot(stock_data.index, stock_data['Close'], 'b-')
ax1.set_title(f'{symbol} Stock Price')
ax1.set_ylabel('Price ($)')
ax1.grid(True)

# 2. Daily Returns Histogram with VaR
ax2 = plt.subplot(4, 1, 2)
ax2.hist(aligned_data['{symbol} Returns'], bins=50, alpha=0.75, color='blue')
ax2.axvline(var_95, color='r', linestyle='dashed', linewidth=2, label=f'VaR (95%): {{var_95:.2%}}')
ax2.set_title('Daily Returns Distribution with VaR')
ax2.set_xlabel('Daily Returns')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True)

# 3. Drawdown Plot
stock_data['Peak'] = stock_data['Close'].cummax()
stock_data['Drawdown'] = (stock_data['Close'] - stock_data['Peak']) / stock_data['Peak']
ax3 = plt.subplot(4, 1, 3)
ax3.fill_between(stock_data.index, 0, stock_data['Drawdown'], color='red', alpha=0.5)
ax3.set_title('Drawdown Over Time')
ax3.set_ylabel('Drawdown (%)')
ax3.set_ylim(min(stock_data['Drawdown'])*1.1, 0.01)
ax3.grid(True)

# 4. Beta / Scatter Plot
ax4 = plt.subplot(4, 1, 4)
ax4.scatter(aligned_data['Market Returns'], aligned_data['{symbol} Returns'], alpha=0.5)
ax4.set_title(f'Beta: {{beta:.2f}} - Stock Returns vs Market Returns')
ax4.set_xlabel('Market Returns')
ax4.set_ylabel('Stock Returns')

# Add regression line
m, b = np.polyfit(aligned_data['Market Returns'], aligned_data['{symbol} Returns'], 1)
ax4.plot(aligned_data['Market Returns'], m*aligned_data['Market Returns'] + b, 'r-')
ax4.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# ----- Risk Assessment -----
# Determine risk level based on metrics
def get_risk_level(volatility, beta, max_drawdown):
    vol_score = 0
    beta_score = 0
    drawdown_score = 0
    
    # Volatility scoring
    if volatility < 0.15:  # less than 15% annualized
        vol_score = 1  # Low
    elif volatility < 0.25:
        vol_score = 2  # Medium
    else:
        vol_score = 3  # High
        
    # Beta scoring
    if beta < 0.8:
        beta_score = 1  # Low
    elif beta < 1.2:
        beta_score = 2  # Medium
    else:
        beta_score = 3  # High
    
    # Drawdown scoring
    if max_drawdown < 0.15:
        drawdown_score = 1  # Low
    elif max_drawdown < 0.30:
        drawdown_score = 2  # Medium
    else:
        drawdown_score = 3  # High
    
    # Average score
    avg_score = (vol_score + beta_score + drawdown_score) / 3
    
    if avg_score < 1.5:
        return "Low"
    elif avg_score < 2.5:
        return "Medium"
    else:
        return "High"

risk_level = get_risk_level(annual_volatility, beta, max_drawdown)

# Print results
print("===== RISK ANALYSIS SUMMARY =====")
print(f"Stock Symbol: {symbol}")
print(f"Analysis Period: {period}")
print(f"Current Price: ${stock_data['Close'].iloc[-1]:.2f}")
print("\\n--- RISK METRICS ---")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Beta (vs {benchmark_symbol}): {beta:.2f}")
print(f"Value at Risk (95%): {var_95:.2%} (${dollar_var_95:.2f} per $100 investment)")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Overall Risk Level: {risk_level}")

# Detailed risk interpretation
print("\\n--- RISK INTERPRETATION ---")
if annual_volatility > 0.25:
    print("HIGH VOLATILITY: This stock exhibits significant price swings, indicating higher risk.")
elif annual_volatility > 0.15:
    print("MODERATE VOLATILITY: This stock shows average price fluctuations compared to the broader market.")
else:
    print("LOW VOLATILITY: This stock shows relatively stable price movements, indicating lower risk.")

if beta > 1.2:
    print(f"HIGH BETA ({beta:.2f}): This stock tends to move more dramatically than the overall market, amplifying both gains and losses.")
elif beta > 0.8:
    print(f"MODERATE BETA ({beta:.2f}): This stock tends to move relatively in line with the overall market.")
else:
    print(f"LOW BETA ({beta:.2f}): This stock tends to move less dramatically than the overall market, potentially providing some insulation from market downturns.")

if max_drawdown > 0.30:
    print(f"LARGE DRAWDOWN RISK: Historical maximum drawdown of {max_drawdown:.2%} indicates significant potential for capital loss during market downturns.")
elif max_drawdown > 0.15:
    print(f"MODERATE DRAWDOWN RISK: Historical maximum drawdown of {max_drawdown:.2%} indicates moderate potential for capital loss during market downturns.")
else:
    print(f"LOW DRAWDOWN RISK: Historical maximum drawdown of {max_drawdown:.2%} indicates relatively lower potential for capital loss during market downturns.")

if sharpe_ratio > 1:
    print(f"FAVORABLE RISK-ADJUSTED RETURN: Sharpe ratio of {sharpe_ratio:.2f} suggests good returns relative to risk taken.")
elif sharpe_ratio > 0:
    print(f"MODERATE RISK-ADJUSTED RETURN: Sharpe ratio of {sharpe_ratio:.2f} suggests returns are proportional to risk taken.")
else:
    print(f"POOR RISK-ADJUSTED RETURN: Sharpe ratio of {sharpe_ratio:.2f} suggests returns do not adequately compensate for the risk taken.")
"""
    
    def analyze_risk(self, symbol: str, period: str, benchmark_symbol: str = "SPY") -> RiskMetrics:
        """
        Perform risk analysis and return structured results.
        
        Args:
            symbol: The stock ticker symbol
            period: The time period for analysis
            benchmark_symbol: The ticker symbol for the benchmark index
            
        Returns:
            RiskMetrics: Structured result of the risk analysis
        """
        # This would typically involve executing the code and parsing results
        # Placeholder implementation
        return RiskMetrics(
            symbol=symbol,
            volatility=0.0,
            beta=0.0,
            var_95=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            risk_assessment="Analysis not implemented yet"
        )
