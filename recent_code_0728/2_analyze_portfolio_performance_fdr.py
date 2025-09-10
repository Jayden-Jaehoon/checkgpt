import os
import json
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import random
import requests
import math
import yfinance as yf



# Constants
PROMPTS_DICT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Default/prompts_dict.json'
REPHRASE_PROMPTS_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Default/rephrase_prompts.json'
PORTFOLIOS_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Prompts/prompts_repetition.json'
OUTPUT_DIR = '/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Performance_Analysis'
PRICE_DATA_FILE = os.path.join(OUTPUT_DIR, 'sp500_2year_returns.csv')
# Note: Market cap files are now date-specific (e.g., market_caps_20231001.csv) and created dynamically

# Performance evaluation parameters
START_DATE = '2023-10-01'
END_DATE = '2025-07-31'  # Current date
RISK_FREE_RATE = 0  # Risk-free rate (0 as specified)

# Historical market cap weights files are created with date-specific names (e.g., market_caps_20231001.csv)
# This ensures we use market cap weights from the portfolio creation date
INVESTOR_TYPES = [
    "neutral_investor", 
    "value_investor", 
    "growth_investor",
    "momentum_investor", 
    "speculative_trader", 
    "index_mimicker",
    "thematic_investor", 
    "sentiment_driven_investor", 
    "non_financial_background_investor",
    "low_risk_aversion_investor", 
    "high_risk_aversion_investor"
]
PERIODS = {
    '1M': 30,     # ~1 month in calendar days
    '6M': 180,    # ~6 months in calendar days
    '1Y': 365,    # ~1 year in calendar days
    '2Y': 730     # ~2 years in calendar days
}

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ticker mapping for company name changes
TICKER_MAPPING = {
    'FB': 'META',  # Facebook -> Meta
    'BRK.B': 'BRK-B',  # Berkshire Hathaway B shares
    'BRK-B': 'BRK-B',  # Already in correct format
    'GOOGL': 'GOOGL',  # Keep as is
    'GOOG': 'GOOG',    # Keep as is
    'SQ': 'XYZ',       # Square -> Block Inc. (ticker changed to XYZ in Jan 2025)
    'FISV': 'FI',      # Fiserv Inc. ticker updated to FI
}

# Delisted tickers that should be excluded from analysis
DELISTED_TICKERS = {
    'ATVI',  # Activision Blizzard - delisted Oct 2023 (Microsoft acquisition)
    'TWTR',  # Twitter - delisted after Elon Musk acquisition (became private)
}

def extract_50_reps_subset(data):
    """
    Extract the first 50 repetitions from the 100 repetitions data.
    
    Args:
        data (dict): Portfolio data with 100 repetitions for each investor type
        
    Returns:
        dict: Portfolio data with only the first 50 repetitions
    """
    # Check if subset file already exists
    subset_file = os.path.join(OUTPUT_DIR, 'portfolios_50_subset.json')
    
    if os.path.exists(subset_file):
        print(f"📁 Loading existing 50 reps subset from {subset_file}")
        try:
            with open(subset_file, 'r') as f:
                subset = json.load(f)
            print(f"✅ Successfully loaded 50 reps subset with {len(subset)} investor types")
            return subset
        except Exception as e:
            print(f"❌ Error loading existing subset file: {e}")
            print("🔄 Will create new subset file...")
    
    print("🆕 Creating new 50 reps subset from 100 reps data...")
    subset = {}
    
    for investor_type in INVESTOR_TYPES:
        if investor_type in data:
            portfolios = data[investor_type]
            if len(portfolios) >= 50:
                subset[investor_type] = portfolios[:50]  # Take first 50 reps
                print(f"  {investor_type}: {len(subset[investor_type])} reps (from {len(portfolios)} total)")
            else:
                print(f"Warning: {investor_type} has only {len(portfolios)} reps, expected at least 50")
                subset[investor_type] = portfolios
        else:
            print(f"Error: investor_type '{investor_type}' not found in JSON data")
            print(f"Available keys in JSON: {list(data.keys())}")
            # Continue processing other investor types instead of crashing
    
    # Save newly created subset to file for future use
    with open(subset_file, 'w') as f:
        json.dump(subset, f, indent=2)
    
    print(f"💾 New 50 reps subset created and saved to {subset_file}")
    return subset

def calculate_top_stocks(portfolios, n=30):
    """
    Calculate the top N stocks based on frequency across all portfolios.
    
    Args:
        portfolios (list): List of portfolios, where each portfolio is a list of stock tickers
        n (int): Number of top stocks to return
        
    Returns:
        list: List of top N stocks by frequency
    """
    # Flatten the list of portfolios
    all_stocks = [stock for portfolio in portfolios for stock in portfolio]
    
    # Count frequencies
    counter = Counter(all_stocks)
    
    # Get top N stocks
    top_stocks = [stock for stock, _ in counter.most_common(n)]
    
    return top_stocks

def get_market_cap_weights(tickers, date):
    """
    Calculate market capitalization weights for the given date using historical price and shares data.
    
    Args:
        tickers (list): List of stock tickers
        date (str): Date for market cap calculation in YYYY-MM-DD format (e.g., '2023-10-01')
        
    Returns:
        dict: Dictionary mapping tickers to their market cap weights
        
    Raises:
        FileNotFoundError: If the market cap weights file doesn't exist and cannot be created
        ValueError: If market cap data is missing or invalid
    """
    # Create date-specific market cap file
    date_str = date.replace('-', '')  # Convert 2023-10-01 to 20231001
    market_cap_file = os.path.join(OUTPUT_DIR, f'market_caps_{date_str}.csv')
    
    # Check if market cap file exists
    if not os.path.exists(market_cap_file):
        print(f"Market cap weights file not found: {market_cap_file}")
        print(f"Attempting to create market cap weights file for {date}...")
        
        # Try to create the file
        try:
            market_caps = _fetch_historical_market_caps(tickers, date, market_cap_file)
            # Note: Data is already saved incrementally in _fetch_historical_market_caps
        except Exception as e:
            raise FileNotFoundError(f"Could not create market cap weights file for {date}: {e}")
    
    # Load market cap data
    print(f"Loading market cap weights from {market_cap_file}")
    try:
        market_caps_df = pd.read_csv(market_cap_file, index_col=0)
        market_caps = market_caps_df['marketCap'].to_dict()
    except Exception as e:
        raise ValueError(f"Error reading market cap weights file: {e}")
    
    # Check if all tickers are present
    missing_tickers = [ticker for ticker in tickers if ticker not in market_caps]
    if missing_tickers:
        print(f"Missing market cap data for {len(missing_tickers)} tickers: {missing_tickers}")
        print(f"Attempting to fetch missing tickers for {date}...")
        
        try:
            new_market_caps = _fetch_historical_market_caps(missing_tickers, date, market_cap_file)
            market_caps.update(new_market_caps)
            # Note: Data is already saved incrementally in _fetch_historical_market_caps
        except Exception as e:
            raise ValueError(f"Could not fetch market cap data for missing tickers on {date}: {e}")
    
    # Calculate weights
    available_market_caps = {ticker: market_caps.get(ticker, 0) for ticker in tickers if market_caps.get(ticker, 0) > 0}
    total_market_cap = sum(available_market_caps.values())
    
    if total_market_cap <= 0:
        raise ValueError(f"Total market cap is zero or negative for date {date}")
    
    weights = {ticker: market_caps.get(ticker, 0) / total_market_cap if market_caps.get(ticker, 0) > 0 else 0 
               for ticker in tickers}
    
    # Print summary
    print(f"\nMarket Cap Weights Summary for {date}:")
    for ticker in sorted(tickers, key=lambda x: weights.get(x, 0), reverse=True):
        weight = weights.get(ticker, 0)
        market_cap = market_caps.get(ticker, 0)
        if weight > 0:
            print(f"{ticker}: {market_cap:,.0f} ({weight:.2%})")
    
    return weights


def _fetch_historical_market_caps(tickers, date, market_cap_file):
    """
    Fetch historical market cap data for given tickers on a specific date.
    
    Main Method: Price Ratio - Uses current market cap adjusted by price ratio
    Fallback Method: Price × Shares - Traditional calculation method
    
    Args:
        tickers (list): List of stock tickers
        date (str): Date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary mapping tickers to their historical market caps
    """
    print(f"Fetching historical market cap data for {len(tickers)} tickers on {date}...")
    market_caps = {}
    failed_tickers = []
    
    # Load existing market cap data if file exists to avoid re-downloading
    if os.path.exists(market_cap_file):
        try:
            existing_df = pd.read_csv(market_cap_file, index_col=0)
            market_caps = existing_df.iloc[:, 0].to_dict()  # First column contains market caps
            print(f"Loaded {len(market_caps)} existing market cap entries")
        except Exception as e:
            print(f"Error loading existing market cap file: {e}")
    
    # Filter out tickers that already have market cap data
    missing_tickers = [ticker for ticker in tickers if ticker not in market_caps]
    if not missing_tickers:
        print(f"All {len(tickers)} tickers already have market cap data. Skipping download.")
        return market_caps
    
    print(f"Need to fetch market cap for {len(missing_tickers)} missing tickers: {missing_tickers}")
    
    # Convert date for calculations  
    from datetime import datetime, timedelta
    target_date = datetime.strptime(date, '%Y-%m-%d')
    
    for ticker in missing_tickers:
        # Apply ticker mapping if needed
        mapped_ticker = TICKER_MAPPING.get(ticker, ticker)
        print(f"Fetching historical market cap for {mapped_ticker} on {date}...")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
        
        try:
            ticker_obj = yf.Ticker(mapped_ticker)
            info = ticker_obj.info
            
            # Get historical price data first (needed for both methods)
            start_date_range = (target_date - timedelta(days=5)).strftime('%Y-%m-%d')
            end_date_range = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
            
            hist_data = ticker_obj.history(start=start_date_range, end=end_date_range)
            
            if hist_data.empty:
                print(f"  {mapped_ticker}: No historical price data available")
                failed_tickers.append(ticker)
                continue
            
            # Find the closest date to our target date
            target_timestamp = pd.Timestamp(date)
            # Match timezone if hist_data.index has timezone info
            if hist_data.index.tz is not None:
                target_timestamp = target_timestamp.tz_localize(hist_data.index.tz)
            closest_date_idx = hist_data.index.get_indexer([target_timestamp], method='nearest')[0]
            
            if closest_date_idx < 0 or closest_date_idx >= len(hist_data):
                print(f"  {mapped_ticker}: Could not find price data for {date}")
                failed_tickers.append(ticker)
                continue
            
            # Get historical price
            price_column = 'Adj Close' if 'Adj Close' in hist_data.columns else 'Close'
            historical_price = hist_data.iloc[closest_date_idx][price_column]
            actual_date = hist_data.index[closest_date_idx].strftime('%Y-%m-%d')
            
            if historical_price <= 0:
                print(f"  {mapped_ticker}: Invalid historical price: {historical_price}")
                failed_tickers.append(ticker)
                continue
            
            # Main Method: Price Ratio (더 정확하고 효율적)
            current_market_cap = info.get('marketCap')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if current_market_cap and current_market_cap > 0 and current_price and current_price > 0:
                # Calculate historical market cap using price ratio
                # Ensure numeric operations: convert to float explicitly
                historical_price_num = float(historical_price)
                current_price_num = float(current_price)
                current_market_cap_num = float(current_market_cap)
                
                price_ratio = historical_price_num / current_price_num
                historical_market_cap = current_market_cap_num * price_ratio
                
                market_caps[ticker] = historical_market_cap
                # Save immediately after each ticker calculation (incremental saving)
                _save_market_cap_weights(market_caps, market_cap_file)
                print(f"✅ {mapped_ticker} [MAIN]: Current=${current_price_num:.2f}, Historical=${historical_price_num:.2f} ({actual_date})")
                print(f"    CurrentMarketCap=${current_market_cap_num:,.0f}, Ratio={price_ratio:.4f}")
                print(f"    HistoricalMarketCap=${historical_market_cap:,.0f} - SAVED")
                continue
            
            # Fallback Method: Price × Shares (전통적인 방식)
            print(f"  {mapped_ticker}: Using fallback method (Price × Shares)")
            shares_outstanding = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
            
            if shares_outstanding and shares_outstanding > 0:
                historical_price_num = float(historical_price)
                shares_outstanding_num = float(shares_outstanding)
                historical_market_cap = historical_price_num * shares_outstanding_num
                market_caps[ticker] = historical_market_cap
                # Save immediately after each ticker calculation (incremental saving)
                _save_market_cap_weights(market_caps, market_cap_file)
                print(f"✅ {mapped_ticker} [FALLBACK]: Price=${historical_price_num:.2f} ({actual_date})")
                print(f"    Shares={shares_outstanding_num:,.0f}, MarketCap={historical_market_cap:,.0f} - SAVED")
                continue
            
            # If both methods fail
            print(f"  {mapped_ticker}: Both methods failed - no market cap or shares data")
            failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"  {mapped_ticker}: Error fetching data - {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"\n⚠️  Warning: Failed to get market cap for {len(failed_tickers)} ticker(s): {failed_tickers}")
        # Set failed tickers to 0 market cap and save to ensure file exists
        for ticker in failed_tickers:
            market_caps[ticker] = 0
        
        # Save the file even with failed tickers to prevent file not found errors
        if market_caps:
            market_cap_file = os.path.join(OUTPUT_DIR, f'market_caps_{date.replace("-", "")}.csv')
            _save_market_cap_weights(market_caps, market_cap_file)
            print(f"💾 Market cap file created with available data: {market_cap_file}")
    
    success_count = len([v for v in market_caps.values() if v > 0])
    print(f"\n📊 Market cap calculation summary for {date}:")
    print(f"  ✅ Successfully calculated: {success_count} tickers")
    print(f"  ❌ Failed: {len(failed_tickers)} tickers")
    print(f"  💾 Total saved: {len(market_caps)} tickers")
    
    return market_caps


def _save_market_cap_weights(market_caps, file_path):
    """Save market cap data to CSV file."""
    df = pd.DataFrame({'marketCap': market_caps})
    df.to_csv(file_path)
    # Reduced logging to avoid spam during incremental saves
    # print(f"Market cap data saved to {file_path}")


def get_progress_status(tickers, start_date, date):
    """
    Check progress status for price data and market cap data.
    
    Args:
        tickers (list): List of tickers to check
        start_date (str): Start date for price data
        date (str): Date for market cap data
        
    Returns:
        dict: Progress status information
    """
    # Check price data progress
    price_tickers = []
    if os.path.exists(PRICE_DATA_FILE):
        try:
            price_df = pd.read_csv(PRICE_DATA_FILE, index_col=0)
            price_tickers = list(price_df.columns)
        except:
            pass
    
    # Check market cap data progress
    date_str = date.replace('-', '')
    market_cap_file = os.path.join(OUTPUT_DIR, f'market_caps_{date_str}.csv')
    market_cap_tickers = []
    if os.path.exists(market_cap_file):
        try:
            market_cap_df = pd.read_csv(market_cap_file, index_col=0)
            market_cap_tickers = list(market_cap_df.index)
        except:
            pass
    
    # Apply ticker mapping for comparison
    mapped_tickers = [TICKER_MAPPING.get(ticker, ticker) for ticker in tickers]
    
    # Calculate progress
    missing_price_tickers = [t for t in mapped_tickers if t not in price_tickers]
    missing_market_cap_tickers = [t for t in tickers if t not in market_cap_tickers]  # Use original tickers for market cap
    
    status = {
        'total_tickers': len(tickers),
        'price_data': {
            'completed': len(price_tickers),
            'missing': len(missing_price_tickers),
            'missing_tickers': missing_price_tickers,
            'progress_pct': (len(price_tickers) / len(mapped_tickers)) * 100 if mapped_tickers else 0
        },
        'market_cap_data': {
            'completed': len(market_cap_tickers),
            'missing': len(missing_market_cap_tickers),
            'missing_tickers': missing_market_cap_tickers,
            'progress_pct': (len(market_cap_tickers) / len(tickers)) * 100 if tickers else 0
        }
    }
    
    return status


def print_progress_status(tickers, start_date, date):
    """Print current progress status."""
    status = get_progress_status(tickers, start_date, date)
    
    print(f"\n📊 Progress Status for {len(tickers)} tickers:")
    print(f"  💰 Price Data: {status['price_data']['completed']}/{status['total_tickers']} "
          f"({status['price_data']['progress_pct']:.1f}%) completed")
    if status['price_data']['missing_tickers']:
        print(f"    Missing: {', '.join(status['price_data']['missing_tickers'][:5])}{'...' if len(status['price_data']['missing_tickers']) > 5 else ''}")
    
    print(f"  🏢 Market Cap: {status['market_cap_data']['completed']}/{status['total_tickers']} "
          f"({status['market_cap_data']['progress_pct']:.1f}%) completed")
    if status['market_cap_data']['missing_tickers']:
        print(f"    Missing: {', '.join(status['market_cap_data']['missing_tickers'][:5])}{'...' if len(status['market_cap_data']['missing_tickers']) > 5 else ''}")
    
    if status['price_data']['missing'] == 0 and status['market_cap_data']['missing'] == 0:
        print("  ✅ All data is complete!")
    
    return status

def get_portfolio_returns(tickers, start_date, end_date, weighting='equal'):
    """
    Calculate portfolio returns for different time horizons.
    
    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        weighting (str): Weighting method ('equal' or 'market_cap')
        
    Returns:
        dict: Dictionary with performance metrics for different time horizons
    """
    if not tickers:
        return {}
    
    # Filter out delisted tickers
    original_count = len(tickers)
    original_tickers = tickers.copy()
    tickers = [ticker for ticker in tickers if ticker not in DELISTED_TICKERS]
    if len(tickers) < original_count:
        excluded = [ticker for ticker in original_tickers if ticker in DELISTED_TICKERS]
        print(f"Excluded {original_count - len(tickers)} delisted tickers: {excluded}")
    
    if not tickers:
        print("All tickers are delisted. Cannot calculate returns.")
        return {}
    
    # Show progress status before starting (useful for resuming interrupted processes)
    if weighting == 'market_cap':
        print_progress_status(tickers, start_date, start_date)
    
    # Apply ticker mapping
    full_modified_tickers = [TICKER_MAPPING.get(ticker, ticker) for ticker in tickers]
    
    # Check if price data file exists
    price_data_exists = False
    all_prices = pd.DataFrame()
    
    if os.path.exists(PRICE_DATA_FILE):
        print(f"Loading existing price data from {PRICE_DATA_FILE}")
        try:
            all_prices = pd.read_csv(PRICE_DATA_FILE, index_col=0, parse_dates=True)
            # Check if all tickers are in the price data
            missing_tickers = [ticker for ticker in full_modified_tickers if ticker not in all_prices.columns]
            if not missing_tickers:
                price_data_exists = True
                print(f"All {len(full_modified_tickers)} tickers already exist in price data. Skipping download.")
            else:
                print(f"Missing price data for {len(missing_tickers)} tickers: {missing_tickers}")
                print(f"Existing tickers: {len(full_modified_tickers) - len(missing_tickers)}")
                # Update modified_tickers to only include missing ones for download
                modified_tickers = missing_tickers
            # Keep full_modified_tickers for later use
        except Exception as e:
            print(f"Error loading price data: {e}")
            # If there's an error, use all tickers for download
            modified_tickers = full_modified_tickers
    
    if not price_data_exists:
        # Download price data
        print(f"Downloading price data for {len(modified_tickers)} tickers...")
        
        # Process tickers in chunks to reduce API load
        chunk_size = 1  # Process one ticker at a time
        
        # Browser session setup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        # Process each ticker
        for i in range(0, len(modified_tickers), chunk_size):
            chunk = modified_tickers[i:i+chunk_size]
            
            for ticker in chunk:
                print(f"Downloading data for {ticker}...")
                
                # Random delay to avoid rate limiting
                delay = 1 + random.random() * 10
                print(f"Waiting {delay:.1f} seconds before downloading...")
                time.sleep(delay)
                
                # Retry mechanism
                max_retries = 8
                success = False
                
                for attempt in range(max_retries):
                    try:
                        # Try to download data using FinanceDataReader
                        print(f"Attempt {attempt+1}/{max_retries}: Using FinanceDataReader for {ticker}")
                        ticker_data = fdr.DataReader(ticker, start=start_date, end=end_date)
                        
                        if not ticker_data.empty:
                            # Use Close price if Adj Close is not available
                            column_to_use = "Adj Close" if "Adj Close" in ticker_data else "Close"
                            ticker_prices = ticker_data[column_to_use]
                            
                            # Convert to DataFrame if it's a Series
                            if isinstance(ticker_prices, pd.Series):
                                ticker_prices = ticker_prices.to_frame(name=ticker)
                            
                            # Append to all_prices and save immediately
                            if all_prices.empty:
                                all_prices = ticker_prices
                            else:
                                all_prices = pd.concat([all_prices, ticker_prices], axis=1)
                            
                            # Save immediately after each ticker (incremental saving)
                            all_prices.to_csv(PRICE_DATA_FILE)
                            print(f"✅ Successfully downloaded and saved data for {ticker}")
                            success = True
                            break  # Success, exit retry loop
                        
                    except (ValueError, KeyError, IndexError) as e:
                        print(f"Data error downloading data for {ticker}: {e}")
                    except (requests.RequestException, ConnectionError) as e:
                        print(f"Network error downloading data for {ticker}: {e}")
                    except Exception as e:
                        print(f"Unexpected error downloading data for {ticker}: {e}")
                        
                        if attempt < max_retries - 1:
                            # Exponential backoff with jitter
                            wait_time = (2 ** attempt) + random.random() * 10
                            print(f"Waiting {wait_time:.1f} seconds before retrying...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed to download data for {ticker} after {max_retries} attempts")
                
                if not success:
                    print(f"Skipping {ticker} due to download failure")
        
        # Price data is already saved incrementally after each ticker
        print(f"📊 Price data download completed. Final data contains {len(all_prices.columns)} tickers.")
    
    # Handle empty data
    if all_prices.empty:
        print("No price data available. Cannot calculate returns.")
        return {}
    
    # Handle missing data
    prices = all_prices.copy()
    prices = prices.dropna(axis=1, how='all')  # Drop columns with all NaN
    prices = prices.ffill()  # Forward fill remaining NaN
    
    if prices.empty:
        print("No valid price data after handling missing values.")
        return {}
    
    # Filter to include only the tickers we need
    available_tickers = [ticker for ticker in full_modified_tickers if ticker in prices.columns]
    if not available_tickers:
        print("None of the requested tickers are available in the price data.")
        return {}
    
    prices = prices[available_tickers]
    
    # Calculate weights
    if weighting == 'equal':
        weights = np.ones(len(prices.columns)) / len(prices.columns)
    elif weighting == 'market_cap':
        # Use historical market cap weighting for the portfolio start date
        weight_dict = get_market_cap_weights(available_tickers, start_date)
        weights = np.array([weight_dict.get(ticker, 0) for ticker in available_tickers])
        # Normalize weights
        if sum(weights) > 0:
            weights = weights / sum(weights)
        else:
            weights = np.ones(len(prices.columns)) / len(prices.columns)
    
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Calculate weighted portfolio returns
    portfolio_returns = pd.DataFrame()
    for i, ticker in enumerate(prices.columns):
        if i < len(weights):  # Ensure we have a weight for this ticker
            portfolio_returns[ticker] = daily_returns[ticker] * weights[i]
    
    # Sum across stocks to get portfolio returns
    portfolio_daily_returns = portfolio_returns.sum(axis=1)
    
    # Calculate performance metrics for different horizons
    results = {
        'tickers': available_tickers,
        'weights': dict(zip(available_tickers, weights[:len(available_tickers)]))
    }
    
    for period_name, days in PERIODS.items():
        # Calculate date range using calendar days, then filter for actual trading days
        end_date_dt = pd.to_datetime(end_date)
        start_date_for_period = end_date_dt - pd.Timedelta(days=days)
        
        # Ensure timezone compatibility for comparison
        if portfolio_daily_returns.index.tz is not None:
            start_date_for_period = start_date_for_period.tz_localize(portfolio_daily_returns.index.tz)
        
        # Get returns for this horizon using actual trading days in the data
        # 기간동안의 일별 수익률
        horizon_returns = portfolio_daily_returns[portfolio_daily_returns.index >= start_date_for_period]
        actual_trading_days = len(horizon_returns)
        
        if not horizon_returns.empty and len(horizon_returns) > 1:
            # Calculate cumulative return
            # 아래 요약정보를 위한 누적수익류
            cum_return = (1 + horizon_returns).prod() - 1
            
            # Calculate annualized volatility
            # 아래 요약정보를 위한 변동성
            annual_vol = horizon_returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio
            # 초과수익률(사실상 일별수익률) 의 평균 * 252  / 초과수익률의 표준변차 * sqrt(252) = 초과수익률평균 / 초과수익률 표준편차 * sqrt(252)
            excess_returns = horizon_returns - RISK_FREE_RATE / 252  # Daily risk-free rate 여기서는 0
            if len(excess_returns) > 0 and excess_returns.std() > 0:
                sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
            else:
                sharpe = 0
            
            # Store results
            results[f'{period_name}_return'] = cum_return
            results[f'{period_name}_volatility'] = annual_vol
            results[f'{period_name}_sharpe'] = sharpe
        else:
            results[f'{period_name}_return'] = np.nan
            results[f'{period_name}_volatility'] = np.nan
            results[f'{period_name}_sharpe'] = np.nan
    
    return results

def get_sp500_returns(start_date, end_date):
    """
    Calculate S&P 500 index returns for different time horizons.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary with performance metrics for different time horizons
    """
    # S&P 500 index data file
    sp500_file = os.path.join(OUTPUT_DIR, 'sp500_index.csv')
    
    # Check if S&P 500 data file exists
    sp500_data_exists = False
    sp500 = pd.DataFrame()
    
    if os.path.exists(sp500_file):
        print(f"Loading existing S&P 500 data from {sp500_file}")
        try:
            sp500 = pd.read_csv(sp500_file, index_col=0, parse_dates=True)
            if not sp500.empty:
                sp500_data_exists = True
            else:
                print("S&P 500 data file is empty. Fetching new data.")
        except Exception as e:
            print(f"Error loading S&P 500 data: {e}")
    
    if not sp500_data_exists:
        # Download S&P 500 data
        print("Downloading S&P 500 index data...")
        
        # Browser session setup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        # Retry mechanism
        max_retries = 8
        
        # Try different S&P 500 tickers with multiple data sources
        sp500_sources = [
            ("US500", "FDR"),  # FinanceDataReader
            ("^GSPC", "YF")    # Yahoo Finance fallback
        ]
        
        for sp_ticker, source in sp500_sources:
            print(f"Trying S&P 500 ticker: {sp_ticker} using {source}")
            
            # Random delay to avoid rate limiting
            delay = 1 + random.random() * 10
            print(f"Waiting {delay:.1f} seconds before downloading...")
            time.sleep(delay)
            
            for attempt in range(max_retries):
                try:
                    if source == "FDR":
                        # Try to download data using FinanceDataReader
                        print(f"Attempt {attempt+1}/{max_retries}: Using FinanceDataReader for {sp_ticker}")
                        sp500 = fdr.DataReader(sp_ticker, start=start_date, end=end_date)
                    elif source == "YF":
                        # Try to download data using yfinance
                        print(f"Attempt {attempt+1}/{max_retries}: Using yfinance for {sp_ticker}")
                        ticker_obj = yf.Ticker(sp_ticker)
                        sp500 = ticker_obj.history(start=start_date, end=end_date)
                    
                    if not sp500.empty:
                        # Standardize column names - ensure Adj Close exists
                        if "Adj Close" not in sp500.columns:
                            if "Close" in sp500.columns:
                                sp500["Adj Close"] = sp500["Close"]
                            else:
                                print(f"Warning: Neither 'Adj Close' nor 'Close' found in {sp_ticker} data")
                                print(f"Available columns: {list(sp500.columns)}")
                                continue  # Skip this ticker and try next one
                        
                        # Save S&P 500 data for future use
                        sp500.to_csv(sp500_file)
                        print(f"S&P 500 data ({source}) saved to {sp500_file}")
                        break  # Success, exit retry loop
                    
                except (ValueError, KeyError, IndexError) as e:
                    print(f"Data error downloading S&P 500 data with ticker {sp_ticker}: {e}")
                except (requests.RequestException, ConnectionError) as e:
                    print(f"Network error downloading S&P 500 data with ticker {sp_ticker}: {e}")
                except Exception as e:
                    print(f"Unexpected error downloading S&P 500 data with ticker {sp_ticker}: {e}")
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.random() * 4
                        print(f"Waiting {wait_time:.1f} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to download S&P 500 data with ticker {sp_ticker} after {max_retries} attempts")
            
            if not sp500.empty:
                print(f"Successfully downloaded S&P 500 data with ticker {sp_ticker}")
                break  # Success, exit ticker loop
    
    # Handle empty data
    if sp500.empty:
        print("No S&P 500 data available. Cannot calculate returns.")
        return {}
    
    # Calculate daily returns
    column_to_use = "Adj Close" if "Adj Close" in sp500 else "Close"
    sp500_prices = sp500[column_to_use]
    sp500_returns = sp500_prices.pct_change().dropna()
    
    # Calculate performance metrics for different horizons
    results = {}
    
    for period_name, days in PERIODS.items():
        # Calculate date range using calendar days, then filter for actual trading days
        end_date_dt = pd.to_datetime(end_date)
        start_date_for_period = end_date_dt - pd.Timedelta(days=days)
        
        # Ensure timezone compatibility for comparison
        if sp500_returns.index.tz is not None:
            start_date_for_period = start_date_for_period.tz_localize(sp500_returns.index.tz)
        
        # Get returns for this horizon using actual trading days in the data
        horizon_returns = sp500_returns[sp500_returns.index >= start_date_for_period]
        actual_trading_days = len(horizon_returns)
        
        if not horizon_returns.empty and len(horizon_returns) > 1:
            # Calculate cumulative return
            cum_return = (1 + horizon_returns).prod() - 1
            
            # Calculate annualized volatility
            annual_vol = horizon_returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio
            excess_returns = horizon_returns - RISK_FREE_RATE / 252  # Daily risk-free rate
            if len(excess_returns) > 0 and excess_returns.std() > 0:
                sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
            else:
                sharpe = 0
            
            # Store results
            results[f'{period_name}_return'] = cum_return
            results[f'{period_name}_volatility'] = annual_vol
            results[f'{period_name}_sharpe'] = sharpe
        else:
            results[f'{period_name}_return'] = np.nan
            results[f'{period_name}_volatility'] = np.nan
            results[f'{period_name}_sharpe'] = np.nan
    
    return results

def analyze_portfolios(data_50):
    """
    Analyze portfolios and calculate performance metrics.
    
    Args:
        data_50 (dict): Portfolio data with 50 repetitions for each investor type
        
    Returns:
        dict: Dictionary with performance results
    """
    # Initialize results dictionary
    results = {
        'equal_weighted': {},
        'value_weighted': {},
        'sp500': {}
    }
    
    # Get S&P 500 returns for benchmarking
    print("\nCalculating S&P 500 benchmark returns...")
    sp500_results = get_sp500_returns(START_DATE, END_DATE)
    results['sp500'] = sp500_results
    
    # Portfolio sizes to analyze
    portfolio_sizes = [30]
    
    # Process each investor type
    for investor_type in INVESTOR_TYPES:
        if investor_type not in data_50:
            print(f"Warning: {investor_type} not found in data. Skipping...")
            continue
            
        print(f"\nProcessing {investor_type}...")
        
        # Initialize results for this investor type
        results['equal_weighted'][investor_type] = {}
        results['value_weighted'][investor_type] = {}
        
        # Get portfolios for this investor type
        portfolios_50 = data_50[investor_type]
        
        print(f"  Found {len(portfolios_50)} repetitions")
        
        # Analyze different portfolio sizes
        for size in portfolio_sizes:
            print(f"\n  Analyzing top {size} stocks...")
            
            # Calculate top stocks for 50 reps
            top_stocks_50 = calculate_top_stocks(portfolios_50, n=size)
            
            print(f"    Top {size} stocks: {', '.join(top_stocks_50[:5])}...")
            
            # Calculate performance for 50 reps
            print("    Calculating equal-weighted portfolio performance...")
            equal_weighted_perf_50 = get_portfolio_returns(
                top_stocks_50, START_DATE, END_DATE, weighting='equal')
            
            print("    Calculating value-weighted portfolio performance...")
            value_weighted_perf_50 = get_portfolio_returns(
                top_stocks_50, START_DATE, END_DATE, weighting='market_cap')
            
            # Store results
            size_key = f'top_{size}'
            
            results['equal_weighted'][investor_type][size_key] = equal_weighted_perf_50
            results['value_weighted'][investor_type][size_key] = value_weighted_perf_50
    
    # Save results to file
    results_file = os.path.join(OUTPUT_DIR, 'portfolio_performance_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    return results

def create_summary_tables(results):
    """
    Create summary tables for the paper.
    
    Args:
        results (dict): Results dictionary from analyze_portfolios
        
    Returns:
        dict: Dictionary with summary tables as DataFrames
    """
    # Initialize summary tables
    summary = {}
    
    # Portfolio sizes to analyze
    portfolio_sizes = [30]
    
    # Create summary table for equal-weighted portfolios
    equal_weighted_data = []
    for investor_type in INVESTOR_TYPES:
        if investor_type in results['equal_weighted']:
            for size in portfolio_sizes:
                size_key = f'top_{size}'
                if size_key in results['equal_weighted'][investor_type]:
                    perf = results['equal_weighted'][investor_type][size_key]
                    row = {
                        'investor_type': investor_type,
                        'portfolio_size': size
                    }
                    for period_name in PERIODS.keys():
                        return_key = f'{period_name}_return'
                        sharpe_key = f'{period_name}_sharpe'
                        if return_key in perf:
                            row[return_key] = perf[return_key]
                        else:
                            row[return_key] = np.nan
                        if sharpe_key in perf:
                            row[sharpe_key] = perf[sharpe_key]
                        else:
                            row[sharpe_key] = np.nan
                    equal_weighted_data.append(row)
    
    # Create DataFrame
    equal_weighted_df = pd.DataFrame(equal_weighted_data)
    
    # Create summary table for value-weighted portfolios
    value_weighted_data = []
    for investor_type in INVESTOR_TYPES:
        if investor_type in results['value_weighted']:
            for size in portfolio_sizes:
                size_key = f'top_{size}'
                if size_key in results['value_weighted'][investor_type]:
                    perf = results['value_weighted'][investor_type][size_key]
                    row = {
                        'investor_type': investor_type,
                        'portfolio_size': size
                    }
                    for period_name in PERIODS.keys():
                        return_key = f'{period_name}_return'
                        sharpe_key = f'{period_name}_sharpe'
                        if return_key in perf:
                            row[return_key] = perf[return_key]
                        else:
                            row[return_key] = np.nan
                        if sharpe_key in perf:
                            row[sharpe_key] = perf[sharpe_key]
                        else:
                            row[sharpe_key] = np.nan
                    value_weighted_data.append(row)
    
    # Create DataFrame
    value_weighted_df = pd.DataFrame(value_weighted_data)
    
    # Add S&P 500 benchmark to summary
    sp500_data = []
    if 'sp500' in results:
        row = {'benchmark': 'S&P 500'}
        
        for period_name in PERIODS.keys():
            return_key = f'{period_name}_return'
            sharpe_key = f'{period_name}_sharpe'
            if return_key in results['sp500']:
                row[return_key] = results['sp500'][return_key]
            else:
                row[return_key] = np.nan
            if sharpe_key in results['sp500']:
                row[sharpe_key] = results['sp500'][sharpe_key]
            else:
                row[sharpe_key] = np.nan
        
        sp500_data.append(row)
        summary['sp500'] = pd.DataFrame(sp500_data)
    
    # Store summary tables
    summary['equal_weighted'] = equal_weighted_df
    summary['value_weighted'] = value_weighted_df
    
    # Save summary tables to CSV
    equal_weighted_df.to_csv(os.path.join(OUTPUT_DIR, 'equal_weighted_summary.csv'), index=False)
    value_weighted_df.to_csv(os.path.join(OUTPUT_DIR, 'value_weighted_summary.csv'), index=False)
    
    if 'sp500' in summary:
        summary['sp500'].to_csv(os.path.join(OUTPUT_DIR, 'sp500_summary.csv'), index=False)
    
    return summary

def create_visualizations(results, summary):
    """
    Create improved visualizations for the paper.
    
    Args:
        results (dict): Results dictionary from analyze_portfolios
        summary (dict): Summary tables from create_summary_tables
    """
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Create averaged returns comparison (solving overcrowded issue)
    _create_averaged_returns_chart(results)
    
    # 2. Create investor type performance heatmap
    _create_performance_heatmap(results)
    
    # 3. Create Sharpe ratio comparison
    _create_sharpe_comparison(results)


def _create_averaged_returns_chart(results):
    """Create bar chart with averaged returns by weighting method and horizon."""
    plt.figure(figsize=(12, 8))
    
    # Calculate averages across all investor types
    avg_data = []
    size_key = 'top_30'
    
    for period_name in PERIODS.keys():
        return_key = f'{period_name}_return'
        
        # Equal-weighted average
        equal_returns = []
        for investor_type in INVESTOR_TYPES:
            if (investor_type in results['equal_weighted'] and 
                size_key in results['equal_weighted'][investor_type]):
                perf = results['equal_weighted'][investor_type][size_key]
                if return_key in perf and not pd.isna(perf[return_key]):
                    equal_returns.append(perf[return_key])
        
        if equal_returns:
            avg_data.append({
                'horizon': period_name,
                'weighting': 'Equal-Weighted',
                'return': np.mean(equal_returns),
                'std': np.std(equal_returns) + 1e-5  # Add small value to ensure visible error bars
            })
        
        # Value-weighted average
        value_returns = []
        for investor_type in INVESTOR_TYPES:
            if (investor_type in results['value_weighted'] and 
                size_key in results['value_weighted'][investor_type]):
                perf = results['value_weighted'][investor_type][size_key]
                if return_key in perf and not pd.isna(perf[return_key]):
                    value_returns.append(perf[return_key])
        
        if value_returns:
            avg_data.append({
                'horizon': period_name,
                'weighting': 'Value-Weighted',
                'return': np.mean(value_returns),
                'std': np.std(value_returns) + 1e-5  # Add small value to ensure visible error bars
            })
        
        # S&P 500 benchmark
        if 'sp500' in results and return_key in results['sp500']:
            if not pd.isna(results['sp500'][return_key]):
                avg_data.append({
                    'horizon': period_name,
                    'weighting': 'S&P 500',
                    'return': results['sp500'][return_key],
                    'std': 0  # No standard deviation for single benchmark
                })
    
    # Convert to DataFrame and create plot
    if avg_data:
        avg_df = pd.DataFrame(avg_data)
        
        # Create grouped bar chart with error bars
        ax = sns.barplot(x='horizon', y='return', hue='weighting', data=avg_df, 
                        palette=['skyblue', 'lightcoral', 'lightgreen'])
        
        # Add error bars for standard deviation
        for i, (horizon, group) in enumerate(avg_df.groupby('horizon')):
            for j, (weighting, row) in enumerate(group.iterrows()):
                if row['std'] > 0:  # Only add error bars if std > 0
                    x_pos = i + (j - 1) * 0.27  # Adjust position for grouped bars
                    ax.errorbar(x_pos, row['return'], yerr=row['std'], 
                              fmt='none', color='black', capsize=3, alpha=0.7)
        
        plt.title('Average Portfolio Returns by Horizon and Weighting Method\n(Averaged across all 11 Investor Types)', fontsize=14)
        plt.xlabel('Time Horizon', fontsize=12)
        plt.ylabel('Average Return', fontsize=12)
        plt.legend(title='Weighting Method', fontsize=10)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'averaged_returns_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


def _create_performance_heatmap(results):
    """Create heatmap showing individual investor type performance."""
    # Create separate heatmaps for equal-weighted and value-weighted
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    size_key = 'top_30'
    
    # Equal-weighted heatmap
    eq_data = []
    for investor_type in INVESTOR_TYPES:
        if (investor_type in results['equal_weighted'] and 
            size_key in results['equal_weighted'][investor_type]):
            row_data = {'Investor Type': investor_type}
            perf = results['equal_weighted'][investor_type][size_key]
            for period_name in PERIODS.keys():
                return_key = f'{period_name}_return'
                row_data[period_name] = perf.get(return_key, np.nan)
            eq_data.append(row_data)
    
    if eq_data:
        eq_df = pd.DataFrame(eq_data).set_index('Investor Type')
        sns.heatmap(eq_df, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax1,
                   cbar_kws={'label': 'Return'})
        ax1.set_title('Equal-Weighted Portfolio Returns', fontsize=12)
        ax1.set_xlabel('Time Horizon', fontsize=10)
    
    # Value-weighted heatmap
    vw_data = []
    for investor_type in INVESTOR_TYPES:
        if (investor_type in results['value_weighted'] and 
            size_key in results['value_weighted'][investor_type]):
            row_data = {'Investor Type': investor_type}
            perf = results['value_weighted'][investor_type][size_key]
            for period_name in PERIODS.keys():
                return_key = f'{period_name}_return'
                row_data[period_name] = perf.get(return_key, np.nan)
            vw_data.append(row_data)
    
    if vw_data:
        vw_df = pd.DataFrame(vw_data).set_index('Investor Type')
        sns.heatmap(vw_df, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax2,
                   cbar_kws={'label': 'Return'})
        ax2.set_title('Value-Weighted Portfolio Returns', fontsize=12)
        ax2.set_xlabel('Time Horizon', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'returns_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _create_sharpe_comparison(results):
    """Create Sharpe ratio comparison chart."""
    plt.figure(figsize=(12, 8))
    
    # Calculate averages across all investor types for Sharpe ratios
    avg_data = []
    size_key = 'top_30'
    
    for period_name in PERIODS.keys():
        sharpe_key = f'{period_name}_sharpe'
        
        # Equal-weighted average
        equal_sharpes = []
        for investor_type in INVESTOR_TYPES:
            if (investor_type in results['equal_weighted'] and 
                size_key in results['equal_weighted'][investor_type]):
                perf = results['equal_weighted'][investor_type][size_key]
                if sharpe_key in perf and not pd.isna(perf[sharpe_key]):
                    equal_sharpes.append(perf[sharpe_key])
        
        if equal_sharpes:
            avg_data.append({
                'horizon': period_name,
                'weighting': 'Equal-Weighted',
                'sharpe': np.mean(equal_sharpes),
                'std': np.std(equal_sharpes) + 1e-5  # Add small value to ensure visible error bars
            })
        
        # Value-weighted average
        value_sharpes = []
        for investor_type in INVESTOR_TYPES:
            if (investor_type in results['value_weighted'] and 
                size_key in results['value_weighted'][investor_type]):
                perf = results['value_weighted'][investor_type][size_key]
                if sharpe_key in perf and not pd.isna(perf[sharpe_key]):
                    value_sharpes.append(perf[sharpe_key])
        
        if value_sharpes:
            avg_data.append({
                'horizon': period_name,
                'weighting': 'Value-Weighted',
                'sharpe': np.mean(value_sharpes),
                'std': np.std(value_sharpes) + 1e-5  # Add small value to ensure visible error bars
            })
        
        # S&P 500 benchmark
        if 'sp500' in results and sharpe_key in results['sp500']:
            if not pd.isna(results['sp500'][sharpe_key]):
                avg_data.append({
                    'horizon': period_name,
                    'weighting': 'S&P 500',
                    'sharpe': results['sp500'][sharpe_key],
                    'std': 0
                })
    
    # Convert to DataFrame and create plot
    if avg_data:
        avg_df = pd.DataFrame(avg_data)
        
        # Create grouped bar chart
        ax = sns.barplot(x='horizon', y='sharpe', hue='weighting', data=avg_df,
                        palette=['skyblue', 'lightcoral', 'lightgreen'])
        
        # Add error bars for standard deviation
        for i, (horizon, group) in enumerate(avg_df.groupby('horizon')):
            for j, (weighting, row) in enumerate(group.iterrows()):
                if row['std'] > 0:
                    x_pos = i + (j - 1) * 0.27
                    ax.errorbar(x_pos, row['sharpe'], yerr=row['std'], 
                              fmt='none', color='black', capsize=3, alpha=0.7)
        
        plt.title('Average Sharpe Ratios by Horizon and Weighting Method\n(Averaged across all 11 Investor Types)', fontsize=14)
        plt.xlabel('Time Horizon', fontsize=12)
        plt.ylabel('Average Sharpe Ratio', fontsize=12)
        plt.legend(title='Weighting Method', fontsize=10)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Reference line at 0
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sharpe_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the analysis"""
    print("Starting portfolio performance analysis for 11 investor types...")
    print("Analyzing top 30 stocks portfolio")
    print("Using 50 repetitions per investor type")
    
    # Load portfolio data
    print(f"\nLoading portfolio data from {PORTFOLIOS_FILE}...")
    with open(PORTFOLIOS_FILE, 'r') as f:
        data_100 = json.load(f)
    
    print(f"Found data for {len(data_100)} investor types")
    for investor_type in data_100.keys():
        print(f"  - {investor_type}: {len(data_100[investor_type])} repetitions")
    
    # Extract 50 reps subset
    # data_50 = extract_50_reps_subset(data_100)
    
    # Analyze portfolios
    results = analyze_portfolios(data_100)
    
    # Create summary tables
    summary = create_summary_tables(results)
    
    # Create visualizations
    create_visualizations(results, summary)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - portfolio_performance_results.json: Complete results")
    print("  - equal_weighted_summary.csv: Equal-weighted performance summary")
    print("  - value_weighted_summary.csv: Value-weighted performance summary")
    print("  - sp500_summary.csv: S&P 500 benchmark performance")
    print("  - Various PNG charts in the output directory")

if __name__ == "__main__":
    main()