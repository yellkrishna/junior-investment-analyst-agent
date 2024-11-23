from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.components.tools import FunctionTool
from autogen_ext.models import OpenAIChatCompletionClient
from config import OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID
from agents.fundamental_analysis.fundamental_analysis_agent import fundamental_analysis_agent
from agents.google_search.google_search import search_agent
from scipy.stats import linregress

#!pip install yfinance matplotlib pytz numpy pandas python-dotenv requests bs4

# Define the main stock analysis function
def analyze_stock(ticker: str, benchmark_ticker: str = '^GSPC') -> dict:
    import os
    from datetime import datetime, timedelta

    import matplotlib
    matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from pytz import timezone

    stock = yf.Ticker(ticker)
    benchmark = yf.Ticker(benchmark_ticker)

    # Get historical data (1 year of data to ensure we have enough for 200-day MA)
    end_date = datetime.now(timezone("UTC"))
    start_date = end_date - timedelta(days=365)
    hist = stock.history(start=start_date, end=end_date)
    benchmark_hist = benchmark.history(start=start_date, end=end_date)

    # Ensure we have data
    if hist.empty:
        return {"error": "No historical data available for the specified ticker."}
    if benchmark_hist.empty:
        return {"error": "No historical data available for the specified benchmark ticker."}

    # Align the dates
    hist = hist.dropna()
    benchmark_hist = benchmark_hist.dropna()
    common_dates = hist.index.intersection(benchmark_hist.index)
    hist = hist.loc[common_dates]
    benchmark_hist = benchmark_hist.loc[common_dates]


    # Compute basic statistics and additional metrics
    current_price = stock.info.get("currentPrice", hist["Close"].iloc[-1])
    year_high = stock.info.get("fiftyTwoWeekHigh", hist["High"].max())
    year_low = stock.info.get("fiftyTwoWeekLow", hist["Low"].min())

    # Calculate 50-day and 200-day moving averages
    ma_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
    ma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]

    # Calculate YTD price change and percent change
    ytd_start = datetime(end_date.year, 1, 1, tzinfo=timezone("UTC"))
    ytd_data = hist.loc[ytd_start:]
    if not ytd_data.empty:
        price_change = ytd_data["Close"].iloc[-1] - ytd_data["Close"].iloc[0]
        percent_change = (price_change / ytd_data["Close"].iloc[0]) * 100
    else:
        price_change = percent_change = np.nan

    # Determine trend
    if pd.notna(ma_50) and pd.notna(ma_200):
        if ma_50 > ma_200:
            trend = "Upward"
        elif ma_50 < ma_200:
            trend = "Downward"
        else:
            trend = "Neutral"
    else:
        trend = "Insufficient data for trend analysis"

    # Calculate volatility (standard deviation of daily returns)
    daily_returns = hist["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

    # Calculate Benchmark Returns
    benchmark_returns = benchmark_hist["Close"].pct_change().dropna()

    # Calculate Beta
    aligned_returns = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
    aligned_returns.columns = ['Stock_Returns', 'Benchmark_Returns']
    slope, intercept, r_value, p_value, std_err = linregress(
        aligned_returns['Benchmark_Returns'], aligned_returns['Stock_Returns']
    )
    beta = slope
    alpha = intercept

    # Calculate cumulative returns
    stock_cumulative_returns = (1 + daily_returns).cumprod() - 1
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

    # Relative Performance
    relative_performance = stock_cumulative_returns.iloc[-1] - benchmark_cumulative_returns.iloc[-1]


    # Calculate Exponential Moving Average (EMA)
    ema_span = 20
    ema = hist["Close"].ewm(span=ema_span).mean()
    ema_current = ema.iloc[-1]

    # Calculate PercentB (Bollinger Bands %B)
    bb_span = 20
    sma = hist["Close"].rolling(window=bb_span).mean()
    std = hist["Close"].rolling(window=bb_span).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    percentB = (hist["Close"] - lower_band) / (upper_band - lower_band)
    percentB_current = percentB.iloc[-1]

    # Calculate Stochastic Oscillator (%K)
    sto_span = 14
    low_min = hist["Low"].rolling(window=sto_span).min()
    high_max = hist["High"].rolling(window=sto_span).max()
    percentK = (hist["Close"] - low_min) / (high_max - low_min) * 100
    percentK_current = percentK.iloc[-1]

    # Calculate Momentum
    mom_span = 10
    momentum = hist["Close"].pct_change(periods=mom_span)
    momentum_current = momentum.iloc[-1]

    # Calculate MACD
    ema_12 = hist["Close"].ewm(span=12).mean()
    ema_26 = hist["Close"].ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9).mean()
    macd_current = macd_line.iloc[-1]
    signal_current = signal_line.iloc[-1]

    # Create result dictionary
    result = {
        "ticker": ticker,
        "benchmark_ticker": benchmark_ticker,
        "current_price": current_price,
        "52_week_high": year_high,
        "52_week_low": year_low,
        "50_day_ma": ma_50,
        "200_day_ma": ma_200,
        "ytd_price_change": price_change,
        "ytd_percent_change": percent_change,
        "trend": trend,
        "volatility": volatility,
        "beta": beta,
        "alpha": alpha,
        "r_value": r_value,
        "p_value": p_value,
        "relative_performance": relative_performance,
        "ema_20": ema_current,
        "percentB": percentB_current,
        "stochastic_%K": percentK_current,
        "momentum": momentum_current,
        "macd": macd_current,
        "macd_signal": signal_current,
    }

    # Convert numpy types to Python native types for better JSON serialization
    for key, value in result.items():
        if isinstance(value, np.generic):
            result[key] = value.item()

    # Create a directory for plots
    os.makedirs("technical_plots", exist_ok=True)

    # Plot Close Price with Moving Averages and EMA
    plt.figure(figsize=(14, 7))
    plt.plot(hist.index, hist["Close"], label="Close Price", color='blue')
    plt.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-day MA", color='orange')
    plt.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-day MA", color='green')
    plt.plot(hist.index, ema, label=f"EMA ({ema_span})", color='red')
    plt.title(f"{ticker} Stock Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plot_file_path = f"technical_plots/{ticker}_stockprice.png"
    plt.savefig(plot_file_path)
    plt.close()
    result["plot_file_path"] = plot_file_path

    # Plot Benchmark Comparison
    plt.figure(figsize=(14, 7))
    plt.plot(stock_cumulative_returns.index, stock_cumulative_returns, label=f"{ticker} Cumulative Returns", color='blue')
    plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns, label=f"{benchmark_ticker} Cumulative Returns", color='orange')
    plt.title(f"{ticker} vs {benchmark_ticker} Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    benchmark_plot_file_path = f"technical_plots/{ticker}_vs_{benchmark_ticker}_returns.png"
    plt.savefig(benchmark_plot_file_path)
    plt.close()
    result["benchmark_plot_file_path"] = benchmark_plot_file_path

    # Plot Beta (scatter plot of returns)
    plt.figure(figsize=(7, 7))
    plt.scatter(aligned_returns['Benchmark_Returns'], aligned_returns['Stock_Returns'], alpha=0.5)
    plt.plot(aligned_returns['Benchmark_Returns'], intercept + slope * aligned_returns['Benchmark_Returns'], color='red', label=f"Beta = {beta:.2f}")
    plt.title(f"{ticker} vs {benchmark_ticker} Daily Returns")
    plt.xlabel(f"{benchmark_ticker} Daily Returns")
    plt.ylabel(f"{ticker} Daily Returns")
    plt.legend()
    plt.grid(True)
    beta_plot_file_path = f"technical_plots/{ticker}_beta.png"
    plt.savefig(beta_plot_file_path)
    plt.close()
    result["beta_plot_file_path"] = beta_plot_file_path

    # Plot Bollinger Bands and PercentB
    plt.figure(figsize=(14, 7))
    plt.plot(hist.index, hist["Close"], label="Close Price", color='blue')
    plt.plot(hist.index, upper_band, label="Upper Bollinger Band", color='cyan', linestyle='--')
    plt.plot(hist.index, lower_band, label="Lower Bollinger Band", color='cyan', linestyle='--')
    plt.fill_between(hist.index, lower_band, upper_band, color='cyan', alpha=0.1)
    plt.title(f"{ticker} Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    bb_plot_file_path = f"technical_plots/{ticker}_bollinger_bands.png"
    plt.savefig(bb_plot_file_path)
    plt.close()
    result["bollinger_bands_plot_file_path"] = bb_plot_file_path

    # Plot PercentB
    plt.figure(figsize=(14, 4))
    plt.plot(percentB.index, percentB, label='%B', color='purple')
    plt.axhline(1, color='red', linestyle='--', label='Overbought Threshold')
    plt.axhline(0, color='green', linestyle='--', label='Oversold Threshold')
    plt.title(f"{ticker} Bollinger Bands %B")
    plt.xlabel("Date")
    plt.ylabel("PercentB")
    plt.legend()
    plt.grid(True)
    percentB_plot_file_path = f"technical_plots/{ticker}_percentB.png"
    plt.savefig(percentB_plot_file_path)
    plt.close()
    result["percentB_plot_file_path"] = percentB_plot_file_path

    # Plot Stochastic Oscillator
    plt.figure(figsize=(14, 4))
    plt.plot(percentK.index, percentK, label='Stochastic %K', color='brown')
    plt.axhline(80, color='red', linestyle='--', label='Overbought Threshold')
    plt.axhline(20, color='green', linestyle='--', label='Oversold Threshold')
    plt.title(f"{ticker} Stochastic Oscillator")
    plt.xlabel("Date")
    plt.ylabel("Stochastic %K")
    plt.legend()
    plt.grid(True)
    stochastic_plot_file_path = f"technical_plots/{ticker}_stochastic.png"
    plt.savefig(stochastic_plot_file_path)
    plt.close()
    result["stochastic_plot_file_path"] = stochastic_plot_file_path

    # Plot Momentum
    plt.figure(figsize=(14, 4))
    plt.plot(momentum.index, momentum, label='Momentum', color='orange')
    plt.title(f"{ticker} Momentum")
    plt.xlabel("Date")
    plt.ylabel("Momentum")
    plt.legend()
    plt.grid(True)
    momentum_plot_file_path = f"technical_plots/{ticker}_momentum.png"
    plt.savefig(momentum_plot_file_path)
    plt.close()
    result["momentum_plot_file_path"] = momentum_plot_file_path

    # Plot MACD
    plt.figure(figsize=(14, 7))
    plt.plot(macd_line.index, macd_line, label='MACD Line', color='blue')
    plt.plot(signal_line.index, signal_line, label='Signal Line', color='red')
    plt.bar(macd_line.index, macd_line - signal_line, label='MACD Histogram', color='gray', alpha=0.3)
    plt.title(f"{ticker} MACD")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.legend()
    plt.grid(True)
    macd_plot_file_path = f"technical_plots/{ticker}_MACD.png"
    plt.savefig(macd_plot_file_path)
    plt.close()
    result["macd_plot_file_path"] = macd_plot_file_path

    return result

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

stock_analysis_tool = FunctionTool(
    analyze_stock,
    description="Function to perform technical stock market analysis on a company's stock using its ticker and a benchmark ticker. Generates plots for various technical indicators and returns the calculated indicators."
)

stock_analysis_agent = AssistantAgent(
    name="TechnicalStockAnalyst",
    model_client=model_client,
    tools=[stock_analysis_tool],
    description="Agent specialized in technical stock market analysis of a company's stock. Uses the company's ticker and a benchmark ticker to analyze stock price trends, calculate technical indicators, generate plots, and return the results in the form of a dictioanry to be used by report agent to generate a report."
)