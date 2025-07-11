from crewai import Agent
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import numpy as np

class TechnicalIndicatorResult(BaseModel):
    """Model for technical indicator analysis results."""
    indicator_name: str = Field(..., description="Name of the technical indicator")
    values: List[float] = Field(..., description="List of calculated indicator values")
    dates: List[str] = Field(..., description="List of dates corresponding to the values")
    interpretation: str = Field(..., description="Interpretation of the indicator values")

class TechnicalAnalysisAgent:
    """
    Agent responsible for technical analysis of stock data using various indicators.
    Integrates with TA-Lib for calculating technical indicators.
    """
    
    def __init__(self, llm):
        """Initialize the technical analysis agent."""
        self.agent = Agent(
            role="Technical Analysis Specialist",
            goal="Analyze stock data using technical indicators and provide insightful interpretations",
            backstory="""You are an expert in technical analysis with years of experience in 
                        financial markets. You have a deep understanding of various technical 
                        indicators and how they can be used to identify market trends and potential 
                        trading opportunities.""",
            llm=llm,
            verbose=True,
            memory=True,
        )
    
    def generate_technical_analysis_code(self, symbol: str, indicator: str, period: str) -> str:
        """
        Generate Python code for technical analysis.
        
        Args:
            symbol: The stock ticker symbol
            indicator: The technical indicator to calculate
            period: The time period for analysis
            
        Returns:
            Python code string for the technical analysis
        """
        # Mapping of indicator names to TA-Lib functions
        indicator_mapping = {
            "RSI": self._generate_rsi_code,
            "MACD": self._generate_macd_code,
            "BOLLINGER": self._generate_bollinger_bands_code,
            "SMA": self._generate_sma_code,
            "EMA": self._generate_ema_code,
            "ADX": self._generate_adx_code,
            "ATR": self._generate_atr_code,
            "OBV": self._generate_obv_code,
            "STOCH": self._generate_stochastic_code,
            "ICHIMOKU": self._generate_ichimoku_code,
        }
        
        # Get the appropriate code generation function
        normalized_indicator = indicator.strip().upper()
        code_generator = indicator_mapping.get(normalized_indicator)
        
        if code_generator:
            return code_generator(symbol, period)
        else:
            # Default to a basic RSI implementation if indicator not found
            return self._generate_rsi_code(symbol, period)
    
    def _generate_rsi_code(self, symbol: str, period: str) -> str:
        """Generate code for Relative Strength Index (RSI) analysis."""
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')

# Calculate RSI
stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={{'height_ratios': [3, 1]}})

# Plot price on top subplot
ax1.plot(stock_data.index, stock_data['Close'], 'b-', label='{symbol} Price')
ax1.set_title(f'{symbol} Stock Price and RSI ({period})')
ax1.set_ylabel('Price ($)')
ax1.grid(True)
ax1.legend()

# Plot RSI on bottom subplot
ax2.plot(stock_data.index, stock_data['RSI'], 'r-', label='RSI')
ax2.axhline(y=70, color='g', linestyle='-', label='Overbought (70)')
ax2.axhline(y=30, color='r', linestyle='-', label='Oversold (30)')
ax2.fill_between(stock_data.index, y1=70, y2=stock_data['RSI'].where(stock_data['RSI'] >= 70), color='g', alpha=0.3)
ax2.fill_between(stock_data.index, y1=30, y2=stock_data['RSI'].where(stock_data['RSI'] <= 30), color='r', alpha=0.3)
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.grid(True)
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Interpretation
latest_rsi = stock_data['RSI'].iloc[-1]
print(f"Latest RSI value: {latest_rsi:.2f}")

if latest_rsi > 70:
    print("INTERPRETATION: The stock is currently OVERBOUGHT. This could indicate a potential sell signal or price pullback.")
elif latest_rsi < 30:
    print("INTERPRETATION: The stock is currently OVERSOLD. This could indicate a potential buy signal or price bounce.")
else:
    print(f"INTERPRETATION: The RSI value of {latest_rsi:.2f} indicates that the stock is in a NEUTRAL territory, neither overbought nor oversold.")
"""

    def _generate_macd_code(self, symbol: str, period: str) -> str:
        """Generate code for Moving Average Convergence Divergence (MACD) analysis."""
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')

# Calculate MACD
macd, signal, hist = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
stock_data['MACD'] = macd
stock_data['Signal'] = signal
stock_data['Histogram'] = hist

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={{'height_ratios': [3, 1]}})

# Plot price on top subplot
ax1.plot(stock_data.index, stock_data['Close'], 'b-', label='{symbol} Price')
ax1.set_title(f'{symbol} Stock Price and MACD ({period})')
ax1.set_ylabel('Price ($)')
ax1.grid(True)
ax1.legend()

# Plot MACD on bottom subplot
ax2.plot(stock_data.index, stock_data['MACD'], 'b-', label='MACD')
ax2.plot(stock_data.index, stock_data['Signal'], 'r-', label='Signal')
ax2.bar(stock_data.index, stock_data['Histogram'], color=stock_data['Histogram'].apply(lambda x: 'g' if x > 0 else 'r'), alpha=0.5, label='Histogram')
ax2.set_ylabel('MACD')
ax2.grid(True)
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Interpretation
latest_macd = stock_data['MACD'].iloc[-1]
latest_signal = stock_data['Signal'].iloc[-1]
latest_hist = stock_data['Histogram'].iloc[-1]
print(f"Latest MACD: {latest_macd:.2f}, Signal: {latest_signal:.2f}, Histogram: {latest_hist:.2f}")

# Check for crossovers and trends
if latest_macd > latest_signal:
    if stock_data['MACD'].iloc[-2] < stock_data['Signal'].iloc[-2]:
        print("INTERPRETATION: BULLISH SIGNAL - MACD just crossed above the signal line, which could indicate a good buying opportunity.")
    else:
        print("INTERPRETATION: BULLISH TREND - MACD is above the signal line, suggesting an upward momentum.")
else:
    if stock_data['MACD'].iloc[-2] > stock_data['Signal'].iloc[-2]:
        print("INTERPRETATION: BEARISH SIGNAL - MACD just crossed below the signal line, which could indicate a good selling opportunity.")
    else:
        print("INTERPRETATION: BEARISH TREND - MACD is below the signal line, suggesting a downward momentum.")

if latest_hist > 0 and latest_hist > stock_data['Histogram'].iloc[-2]:
    print("INTERPRETATION: The histogram is positive and increasing, indicating strengthening bullish momentum.")
elif latest_hist > 0 and latest_hist < stock_data['Histogram'].iloc[-2]:
    print("INTERPRETATION: The histogram is positive but decreasing, which might suggest weakening bullish momentum.")
elif latest_hist < 0 and latest_hist < stock_data['Histogram'].iloc[-2]:
    print("INTERPRETATION: The histogram is negative and decreasing, indicating strengthening bearish momentum.")
elif latest_hist < 0 and latest_hist > stock_data['Histogram'].iloc[-2]:
    print("INTERPRETATION: The histogram is negative but increasing, which might suggest weakening bearish momentum.")
"""

    def _generate_bollinger_bands_code(self, symbol: str, period: str) -> str:
        """Generate code for Bollinger Bands analysis."""
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')

# Calculate Bollinger Bands (20, 2)
stock_data['Upper'], stock_data['Middle'], stock_data['Lower'] = talib.BBANDS(stock_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# Calculate Bollinger Bands Width (volatility indicator)
stock_data['BBWidth'] = (stock_data['Upper'] - stock_data['Lower']) / stock_data['Middle']

# Calculate %B (position of price relative to bands)
stock_data['%B'] = (stock_data['Close'] - stock_data['Lower']) / (stock_data['Upper'] - stock_data['Lower'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={{'height_ratios': [3, 1]}})

# Plot price and Bollinger Bands on top subplot
ax1.plot(stock_data.index, stock_data['Close'], 'b-', label='{symbol} Price')
ax1.plot(stock_data.index, stock_data['Upper'], 'g--', label='Upper Band (2σ)')
ax1.plot(stock_data.index, stock_data['Middle'], 'r--', label='Middle Band (SMA 20)')
ax1.plot(stock_data.index, stock_data['Lower'], 'g--', label='Lower Band (2σ)')
ax1.fill_between(stock_data.index, stock_data['Upper'], stock_data['Lower'], alpha=0.1, color='gray')
ax1.set_title(f'{symbol} Stock Price and Bollinger Bands ({period})')
ax1.set_ylabel('Price ($)')
ax1.grid(True)
ax1.legend()

# Plot Bollinger Band Width on bottom subplot
ax2.plot(stock_data.index, stock_data['BBWidth'], 'b-', label='BB Width')
ax2.set_ylabel('BB Width')
ax2.grid(True)
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Interpretation
latest_close = stock_data['Close'].iloc[-1]
latest_upper = stock_data['Upper'].iloc[-1]
latest_middle = stock_data['Middle'].iloc[-1]
latest_lower = stock_data['Lower'].iloc[-1]
latest_bbwidth = stock_data['BBWidth'].iloc[-1]
latest_b_percent = stock_data['%B'].iloc[-1]

print(f"Latest Close: ${latest_close:.2f}")
print(f"Latest Upper Band: ${latest_upper:.2f}")
print(f"Latest Middle Band: ${latest_middle:.2f}")
print(f"Latest Lower Band: ${latest_lower:.2f}")
print(f"Latest BB Width: {latest_bbwidth:.4f}")
print(f"Latest %B: {latest_b_percent:.4f}")

# Interpretation based on price position
if latest_close > latest_upper:
    print("INTERPRETATION: Price is ABOVE the upper band, indicating strong upward momentum but potentially overbought conditions.")
elif latest_close < latest_lower:
    print("INTERPRETATION: Price is BELOW the lower band, indicating strong downward momentum but potentially oversold conditions.")
else:
    distance_from_middle = abs(latest_close - latest_middle) / (latest_upper - latest_middle) * 100
    if latest_close > latest_middle:
        print(f"INTERPRETATION: Price is {distance_from_middle:.1f}% of the way from the middle to the upper band, suggesting positive momentum within normal volatility.")
    else:
        print(f"INTERPRETATION: Price is {distance_from_middle:.1f}% of the way from the middle to the lower band, suggesting negative momentum within normal volatility.")

# Interpretation based on BB Width (volatility)
avg_bbwidth = stock_data['BBWidth'].mean()
if latest_bbwidth > avg_bbwidth * 1.5:
    print(f"INTERPRETATION: BB Width is high ({latest_bbwidth:.4f} vs average {avg_bbwidth:.4f}), indicating increased volatility and potential for major price movements.")
elif latest_bbwidth < avg_bbwidth * 0.5:
    print(f"INTERPRETATION: BB Width is low ({latest_bbwidth:.4f} vs average {avg_bbwidth:.4f}), indicating decreased volatility and potential for a breakout soon.")
else:
    print(f"INTERPRETATION: BB Width is near average ({latest_bbwidth:.4f} vs average {avg_bbwidth:.4f}), suggesting normal volatility levels.")
"""

    def _generate_sma_code(self, symbol: str, period: str) -> str:
        """Generate code for Simple Moving Average (SMA) analysis."""
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import talib

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')

# Calculate Simple Moving Averages
stock_data['SMA_20'] = talib.SMA(stock_data['Close'], timeperiod=20)
stock_data['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
stock_data['SMA_200'] = talib.SMA(stock_data['Close'], timeperiod=200)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], 'b-', label='{symbol} Price')
plt.plot(stock_data.index, stock_data['SMA_20'], 'r--', label='20-day SMA')
plt.plot(stock_data.index, stock_data['SMA_50'], 'g--', label='50-day SMA')
plt.plot(stock_data.index, stock_data['SMA_200'], 'y--', label='200-day SMA')

plt.title(f'{symbol} Stock Price with Simple Moving Averages ({period})')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Interpretation
latest_close = stock_data['Close'].iloc[-1]
latest_sma20 = stock_data['SMA_20'].iloc[-1]
latest_sma50 = stock_data['SMA_50'].iloc[-1]
latest_sma200 = stock_data['SMA_200'].iloc[-1]

print(f"Latest Close: ${latest_close:.2f}")
print(f"20-day SMA: ${latest_sma20:.2f}")
print(f"50-day SMA: ${latest_sma50:.2f}")
print(f"200-day SMA: ${latest_sma200:.2f}")

# Golden Cross / Death Cross detection
sma50_prev = stock_data['SMA_50'].iloc[-2]
sma200_prev = stock_data['SMA_200'].iloc[-2]

if latest_sma50 > latest_sma200 and sma50_prev <= sma200_prev:
    print("INTERPRETATION: GOLDEN CROSS DETECTED - The 50-day SMA has crossed above the 200-day SMA, which is typically a strong bullish signal.")
elif latest_sma50 < latest_sma200 and sma50_prev >= sma200_prev:
    print("INTERPRETATION: DEATH CROSS DETECTED - The 50-day SMA has crossed below the 200-day SMA, which is typically a strong bearish signal.")

# Price relative to moving averages
if latest_close > latest_sma20 and latest_close > latest_sma50 and latest_close > latest_sma200:
    print("INTERPRETATION: STRONG UPTREND - Price is above all major moving averages, indicating a strong bullish trend.")
elif latest_close < latest_sma20 and latest_close < latest_sma50 and latest_close < latest_sma200:
    print("INTERPRETATION: STRONG DOWNTREND - Price is below all major moving averages, indicating a strong bearish trend.")
elif latest_close > latest_sma200:
    print("INTERPRETATION: LONG-TERM UPTREND - Price is above the 200-day SMA, suggesting a positive long-term trend.")
else:
    print("INTERPRETATION: MIXED SIGNALS - The price is showing mixed signals relative to moving averages.")
"""

    def _generate_ema_code(self, symbol: str, period: str) -> str:
        """Generate code for Exponential Moving Average (EMA) analysis."""
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import talib

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')

# Calculate Exponential Moving Averages
stock_data['EMA_12'] = talib.EMA(stock_data['Close'], timeperiod=12)
stock_data['EMA_26'] = talib.EMA(stock_data['Close'], timeperiod=26)
stock_data['EMA_50'] = talib.EMA(stock_data['Close'], timeperiod=50)
stock_data['EMA_200'] = talib.EMA(stock_data['Close'], timeperiod=200)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], 'b-', label='{symbol} Price')
plt.plot(stock_data.index, stock_data['EMA_12'], 'r--', label='12-day EMA')
plt.plot(stock_data.index, stock_data['EMA_26'], 'g--', label='26-day EMA')
plt.plot(stock_data.index, stock_data['EMA_50'], 'm--', label='50-day EMA')
plt.plot(stock_data.index, stock_data['EMA_200'], 'y--', label='200-day EMA')

plt.title(f'{symbol} Stock Price with Exponential Moving Averages ({period})')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Interpretation
latest_close = stock_data['Close'].iloc[-1]
latest_ema12 = stock_data['EMA_12'].iloc[-1]
latest_ema26 = stock_data['EMA_26'].iloc[-1]
latest_ema50 = stock_data['EMA_50'].iloc[-1]
latest_ema200 = stock_data['EMA_200'].iloc[-1]

print(f"Latest Close: ${latest_close:.2f}")
print(f"12-day EMA: ${latest_ema12:.2f}")
print(f"26-day EMA: ${latest_ema26:.2f}")
print(f"50-day EMA: ${latest_ema50:.2f}")
print(f"200-day EMA: ${latest_ema200:.2f}")

# MACD Crossover (simplified)
if latest_ema12 > latest_ema26 and stock_data['EMA_12'].iloc[-2] <= stock_data['EMA_26'].iloc[-2]:
    print("INTERPRETATION: BULLISH SIGNAL - The 12-day EMA has crossed above the 26-day EMA, which is a bullish signal.")
elif latest_ema12 < latest_ema26 and stock_data['EMA_12'].iloc[-2] >= stock_data['EMA_26'].iloc[-2]:
    print("INTERPRETATION: BEARISH SIGNAL - The 12-day EMA has crossed below the 26-day EMA, which is a bearish signal.")

# Price relative to EMAs
if latest_close > latest_ema50 and latest_close > latest_ema200:
    if latest_ema50 > latest_ema200:
        print("INTERPRETATION: STRONG UPTREND - Price and shorter-term EMAs are above longer-term EMAs, indicating a strong bullish trend.")
    else:
        print("INTERPRETATION: POTENTIAL TREND CHANGE - Price is above all EMAs but shorter-term EMAs haven't crossed above longer-term EMAs yet.")
elif latest_close < latest_ema50 and latest_close < latest_ema200:
    if latest_ema50 < latest_ema200:
        print("INTERPRETATION: STRONG DOWNTREND - Price and shorter-term EMAs are below longer-term EMAs, indicating a strong bearish trend.")
    else:
        print("INTERPRETATION: POTENTIAL TREND CHANGE - Price is below all EMAs but shorter-term EMAs haven't crossed below longer-term EMAs yet.")
else:
    print("INTERPRETATION: MIXED SIGNALS - The price is showing mixed signals relative to EMAs, suggesting a possible consolidation or transition phase.")
"""

    def _generate_obv_code(self, symbol: str, period: str) -> str:
        """Generate code for On-Balance Volume (OBV) analysis."""
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')

# Calculate On-Balance Volume (OBV)
stock_data['OBV'] = talib.OBV(stock_data['Close'], stock_data['Volume'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={{'height_ratios': [3, 1]}})

# Plot price on top subplot
ax1.plot(stock_data.index, stock_data['Close'], 'b-', label='{symbol} Price')
ax1.set_title(f'{symbol} Stock Price and On-Balance Volume ({period})')
ax1.set_ylabel('Price ($)')
ax1.grid(True)
ax1.legend()

# Plot OBV on bottom subplot
ax2.plot(stock_data.index, stock_data['OBV'], 'g-', label='OBV')
ax2.set_ylabel('OBV')
ax2.grid(True)
ax2.legend()

# Add a trendline to OBV
days = np.arange(len(stock_data))
obv_polyfit = np.polyfit(days, stock_data['OBV'], 1)
obv_val_line = np.polyval(obv_polyfit, days)
ax2.plot(stock_data.index, obv_val_line, 'r--', alpha=0.7, label='OBV Trend')
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Interpretation
price_change_pct = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0] * 100
obv_change_pct = (stock_data['OBV'].iloc[-1] - stock_data['OBV'].iloc[0]) / abs(stock_data['OBV'].iloc[0]) * 100 if stock_data['OBV'].iloc[0] != 0 else np.inf

print(f"Price change over period: {price_change_pct:.2f}%")
print(f"OBV change over period: {obv_change_pct:.2f}%")

# Divergence check
if price_change_pct > 0 and obv_change_pct < 0:
    print("INTERPRETATION: BEARISH DIVERGENCE - Price is increasing but OBV is decreasing, suggesting that the price rise is not supported by volume and may reverse.")
elif price_change_pct < 0 and obv_change_pct > 0:
    print("INTERPRETATION: BULLISH DIVERGENCE - Price is decreasing but OBV is increasing, suggesting that accumulation may be taking place despite the price decline.")
elif price_change_pct > 0 and obv_change_pct > 0:
    print("INTERPRETATION: BULLISH CONFIRMATION - Both price and OBV are increasing, suggesting that the uptrend is supported by volume and likely to continue.")
elif price_change_pct < 0 and obv_change_pct < 0:
    print("INTERPRETATION: BEARISH CONFIRMATION - Both price and OBV are decreasing, suggesting that the downtrend is supported by volume and likely to continue.")
else:
    print("INTERPRETATION: NEUTRAL - No clear pattern between price and OBV.")

# Recent trend
recent_price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-10]
recent_obv_change = stock_data['OBV'].iloc[-1] - stock_data['OBV'].iloc[-10]

if recent_price_change > 0 and recent_obv_change > 0:
    print("INTERPRETATION: RECENT BULLISH CONFIRMATION - Both price and OBV have increased over the last 10 days, suggesting strong buying pressure.")
elif recent_price_change < 0 and recent_obv_change < 0:
    print("INTERPRETATION: RECENT BEARISH CONFIRMATION - Both price and OBV have decreased over the last 10 days, suggesting strong selling pressure.")
elif recent_price_change > 0 and recent_obv_change < 0:
    print("INTERPRETATION: RECENT BEARISH DIVERGENCE - Price has increased but OBV has decreased over the last 10 days, suggesting potential weakness in the uptrend.")
elif recent_price_change < 0 and recent_obv_change > 0:
    print("INTERPRETATION: RECENT BULLISH DIVERGENCE - Price has decreased but OBV has increased over the last 10 days, suggesting potential strength despite the price decline.")
"""

    def _generate_stochastic_code(self, symbol: str, period: str) -> str:
        """Generate code for Stochastic Oscillator analysis."""
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import talib

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')

# Calculate Stochastic Oscillator
stock_data['slowk'], stock_data['slowd'] = talib.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'], 
                                                       fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={{'height_ratios': [3, 1]}})

# Plot price on top subplot
ax1.plot(stock_data.index, stock_data['Close'], 'b-', label='{symbol} Price')
ax1.set_title(f'{symbol} Stock Price and Stochastic Oscillator ({period})')
ax1.set_ylabel('Price ($)')
ax1.grid(True)
ax1.legend()

# Plot Stochastic Oscillator on bottom subplot
ax2.plot(stock_data.index, stock_data['slowk'], 'k-', label='%K')
ax2.plot(stock_data.index, stock_data['slowd'], 'r-', label='%D')
ax2.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Overbought')
ax2.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Oversold')
ax2.fill_between(stock_data.index, y1=80, y2=stock_data['slowk'].where(stock_data['slowk'] >= 80), color='g', alpha=0.2)
ax2.fill_between(stock_data.index, y1=20, y2=stock_data['slowk'].where(stock_data['slowk'] <= 20), color='r', alpha=0.2)
ax2.set_ylabel('Stochastic')
ax2.set_ylim(0, 100)
ax2.grid(True)
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Interpretation
latest_k = stock_data['slowk'].iloc[-1]
latest_d = stock_data['slowd'].iloc[-1]
print(f"Latest %K: {latest_k:.2f}")
print(f"Latest %D: {latest_d:.2f}")

# Overbought/Oversold conditions
if latest_k > 80:
    print("INTERPRETATION: OVERBOUGHT - The stock is currently showing overbought conditions (Stochastic %K > 80), suggesting a potential reversal or pullback.")
elif latest_k < 20:
    print("INTERPRETATION: OVERSOLD - The stock is currently showing oversold conditions (Stochastic %K < 20), suggesting a potential reversal or bounce.")
else:
    print("INTERPRETATION: NEUTRAL - The stock is currently in a neutral zone, neither overbought nor oversold.")

# Crossovers
prev_k = stock_data['slowk'].iloc[-2]
prev_d = stock_data['slowd'].iloc[-2]

if latest_k > latest_d and prev_k <= prev_d:
    if latest_k < 20 or latest_d < 20:
        print("INTERPRETATION: STRONG BULLISH SIGNAL - %K crossed above %D in the oversold region, suggesting a potentially strong buy signal.")
    else:
        print("INTERPRETATION: BULLISH SIGNAL - %K crossed above %D, suggesting a potential buy signal.")
elif latest_k < latest_d and prev_k >= prev_d:
    if latest_k > 80 or latest_d > 80:
        print("INTERPRETATION: STRONG BEARISH SIGNAL - %K crossed below %D in the overbought region, suggesting a potentially strong sell signal.")
    else:
        print("INTERPRETATION: BEARISH SIGNAL - %K crossed below %D, suggesting a potential sell signal.")
else:
    if latest_k > latest_d:
        print("INTERPRETATION: BULLISH TREND - %K is above %D, suggesting ongoing bullish momentum.")
    else:
        print("INTERPRETATION: BEARISH TREND - %K is below %D, suggesting ongoing bearish momentum.")
"""

    def _generate_ichimoku_code(self, symbol: str, period: str) -> str:
        """Generate code for Ichimoku Cloud analysis."""
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fetch stock data
stock_data = yf.download('{symbol}', period='{period}', interval='1d')

# Calculate Ichimoku Cloud components
high_9 = stock_data['High'].rolling(window=9).max()
low_9 = stock_data['Low'].rolling(window=9).min()
stock_data['Tenkan-sen'] = (high_9 + low_9) / 2  # Conversion Line

high_26 = stock_data['High'].rolling(window=26).max()
low_26 = stock_data['Low'].rolling(window=26).min()
stock_data['Kijun-sen'] = (high_26 + low_26) / 2  # Base Line

stock_data['Senkou Span A'] = ((stock_data['Tenkan-sen'] + stock_data['Kijun-sen']) / 2).shift(26)  # Leading Span A

high_52 = stock_data['High'].rolling(window=52).max()
low_52 = stock_data['Low'].rolling(window=52).min()
stock_data['Senkou Span B'] = ((high_52 + low_52) / 2).shift(26)  # Leading Span B

stock_data['Chikou Span'] = stock_data['Close'].shift(-26)  # Lagging Span

# Plot
plt.figure(figsize=(12, 6))

# Plot price
plt.plot(stock_data.index, stock_data['Close'], 'k-', label='{symbol} Price')

# Plot Ichimoku components
plt.plot(stock_data.index, stock_data['Tenkan-sen'], 'r-', label='Tenkan-sen (9)')
plt.plot(stock_data.index, stock_data['Kijun-sen'], 'b-', label='Kijun-sen (26)')
plt.plot(stock_data.index, stock_data['Senkou Span A'], 'g--', label='Senkou Span A')
plt.plot(stock_data.index, stock_data['Senkou Span B'], 'y--', label='Senkou Span B')

# Fill the cloud
cloud_green = stock_data[stock_data['Senkou Span A'] >= stock_data['Senkou Span B']]
cloud_red = stock_data[stock_data['Senkou Span A'] < stock_data['Senkou Span B']]

plt.fill_between(cloud_green.index, cloud_green['Senkou Span A'], cloud_green['Senkou Span B'], color='g', alpha=0.2)
plt.fill_between(cloud_red.index, cloud_red['Senkou Span A'], cloud_red['Senkou Span B'], color='r', alpha=0.2)

# Complete the plot
plt.title(f'{symbol} Stock Price with Ichimoku Cloud ({period})')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Interpretation
latest_close = stock_data['Close'].iloc[-1]
latest_tenkan = stock_data['Tenkan-sen'].iloc[-1]
latest_kijun = stock_data['Kijun-sen'].iloc[-1]
latest_senkou_a = stock_data['Senkou Span A'].iloc[-1]
latest_senkou_b = stock_data['Senkou Span B'].iloc[-1]
latest_chikou = stock_data['Chikou Span'].iloc[-26] if len(stock_data) > 26 else None  # 26 periods ago relative to Chikou

print(f"Latest Close: ${latest_close:.2f}")
print(f"Tenkan-sen: ${latest_tenkan:.2f}")
print(f"Kijun-sen: ${latest_kijun:.2f}")
print(f"Senkou Span A: ${latest_senkou_a:.2f}")
print(f"Senkou Span B: ${latest_senkou_b:.2f}")

# Determine trend based on cloud
if latest_senkou_a > latest_senkou_b:
    print("INTERPRETATION: BULLISH CLOUD - Senkou Span A is above Senkou Span B, indicating a bullish trend.")
else:
    print("INTERPRETATION: BEARISH CLOUD - Senkou Span A is below Senkou Span B, indicating a bearish trend.")

# Price position relative to the cloud
if latest_close > latest_senkou_a and latest_close > latest_senkou_b:
    print("INTERPRETATION: STRONG BULLISH SIGNAL - Price is above the cloud, indicating a bullish trend.")
elif latest_close < latest_senkou_a and latest_close < latest_senkou_b:
    print("INTERPRETATION: STRONG BEARISH SIGNAL - Price is below the cloud, indicating a bearish trend.")
else:
    print("INTERPRETATION: NEUTRAL/TRANSITIONING - Price is within the cloud, indicating a neutral or transitioning market.")

# Tenkan/Kijun Cross
if latest_tenkan > latest_kijun and stock_data['Tenkan-sen'].iloc[-2] <= stock_data['Kijun-sen'].iloc[-2]:
    print("INTERPRETATION: BULLISH TK CROSS - Tenkan-sen crossed above Kijun-sen, generating a bullish signal.")
elif latest_tenkan < latest_kijun and stock_data['Tenkan-sen'].iloc[-2] >= stock_data['Kijun-sen'].iloc[-2]:
    print("INTERPRETATION: BEARISH TK CROSS - Tenkan-sen crossed below Kijun-sen, generating a bearish signal.")

# Chikou Span analysis
if latest_chikou is not None:
    chikou_price_26_periods_ago = stock_data['Close'].iloc[-26-26] if len(stock_data) > 26+26 else None
    
    if chikou_price_26_periods_ago is not None:
        if latest_chikou > chikou_price_26_periods_ago:
            print("INTERPRETATION: BULLISH CHIKOU - Chikou Span is above the price from 26 periods ago, confirming bullish momentum.")
        else:
            print("INTERPRETATION: BEARISH CHIKOU - Chikou Span is below the price from 26 periods ago, confirming bearish momentum.")
"""

    # Additional methods as needed for implementation
    def analyze(self, symbol: str, indicator: str, period: str) -> TechnicalIndicatorResult:
        """
        Perform technical analysis and return structured results.
        
        Args:
            symbol: The stock ticker symbol
            indicator: The technical indicator to calculate
            period: The time period for analysis
            
        Returns:
            TechnicalIndicatorResult: Structured result of the analysis
        """
        # Implementation depends on how you want to execute and capture the analysis results
        # This would typically involve running the generated code and parsing the output
        
        # Placeholder implementation
        return TechnicalIndicatorResult(
            indicator_name=indicator,
            values=[],
            dates=[],
            interpretation="Analysis not implemented yet."
        )
