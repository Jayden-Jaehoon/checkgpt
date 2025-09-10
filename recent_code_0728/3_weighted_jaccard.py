import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.signal import find_peaks
from collections import Counter, defaultdict
import time
import yfinance as yf
from datetime import datetime, timedelta

# # Constants
RESULT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Prompts/prompts_repetition.json'
MARKET_CAP_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Performance_Analysis/market_caps_20231001.csv'
OUTPUT_DIR = '/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Weighted_Jaccard_Analysis/openai_jaccard'

# Constants
# RESULT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Prompts/prompts_repetition_claude.json'
# MARKET_CAP_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Performance_Analysis_Claude/market_caps_20231001.csv'
# OUTPUT_DIR = '/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Weighted_Jaccard_Analysis/claude_jaccard'

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

PORTFOLIO_SIZES = [10, 20, 30]  # Different portfolio sizes to test
SAMPLE_SIZES = [50, 100]  # Different sample sizes to test

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ticker mapping for company name changes and format differences
TICKER_MAPPING = {
    'FB': 'META',  # Facebook -> Meta
    'BRK.B': 'BRK-B',  # Berkshire Hathaway B shares
    'BRK-B': 'BRK-B',  # Already in correct format
    'GOOGL': 'GOOGL',  # Keep as is
    'GOOG': 'GOOG',    # Keep as is
    'SQ': 'XYZ',       # Square -> Block Inc. (ticker changed to XYZ in Jan 2025)
    'FISV': 'FI',      # Fiserv Inc. ticker updated to FI
}

def _save_market_cap_weights(market_caps, file_path):
    """
    Save market cap data to CSV file.
    
    Args:
        market_caps (dict): Dictionary mapping tickers to their market caps
        file_path (str): Path to save the CSV file
    """
    df = pd.DataFrame({'marketCap': market_caps})
    df.to_csv(file_path)
    print(f"Market cap data saved to {file_path}")

def _fetch_historical_market_caps(tickers, date, market_cap_file):
    """
    Fetch historical market cap data for given tickers on a specific date.
    
    Main Method: Price Ratio - Uses current market cap adjusted by price ratio
    Fallback Method: Price × Shares - Traditional calculation method
    
    Args:
        tickers (list): List of stock tickers
        date (str): Date in YYYY-MM-DD format
        market_cap_file (str): Path to save the market cap data
        
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
            
            # Main Method: Price Ratio (more accurate and efficient)
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
            
            # Fallback Method: Price × Shares (traditional method)
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
            _save_market_cap_weights(market_caps, market_cap_file)
            print(f"💾 Market cap file created with available data: {market_cap_file}")
    
    success_count = len([v for v in market_caps.values() if v > 0])
    print(f"\n📊 Market cap calculation summary for {date}:")
    print(f"  ✅ Successfully calculated: {success_count} tickers")
    print(f"  ❌ Failed: {len(failed_tickers)} tickers")
    print(f"  💾 Total saved: {len(market_caps)} tickers")
    
    return market_caps

def load_market_caps(all_tickers=None):
    """
    Load market capitalization data from CSV file.
    If tickers are provided, check if any are missing and fetch their data.
    
    Args:
        all_tickers (list, optional): List of all tickers to check for missing data
        
    Returns:
        dict: Dictionary mapping tickers to their market caps
    """
    print(f"Loading market cap data from {MARKET_CAP_FILE}...")
    
    # Check if market cap file exists
    if not os.path.exists(MARKET_CAP_FILE):
        print(f"Market cap file not found: {MARKET_CAP_FILE}")
        if all_tickers:
            print(f"Creating new market cap file with {len(all_tickers)} tickers...")
            # Create a new market cap file with all tickers
            market_caps = _fetch_historical_market_caps(all_tickers, '2023-10-01', MARKET_CAP_FILE)
            return market_caps
        else:
            raise FileNotFoundError(f"Market cap file not found: {MARKET_CAP_FILE}")
    
    # Load existing market cap data
    market_caps_df = pd.read_csv(MARKET_CAP_FILE, index_col=0)
    market_caps = market_caps_df['marketCap'].to_dict()
    print(f"Loaded market cap data for {len(market_caps)} tickers")
    
    # Print some key tickers for verification
    key_tickers = ['BRK-B', 'BRK.B', 'META', 'FB']
    print("\nVerifying key tickers in market cap data:")
    for ticker in key_tickers:
        mapped_ticker = TICKER_MAPPING.get(ticker, ticker)
        market_cap = market_caps.get(mapped_ticker, 0)
        if market_cap > 0:
            print(f"  {ticker} -> {mapped_ticker}: ${market_cap:,.0f}")
        else:
            print(f"  {ticker} -> {mapped_ticker}: Not found")
    
    # Check for missing tickers if all_tickers is provided
    if all_tickers:
        # Apply ticker mapping to all tickers
        mapped_tickers = [TICKER_MAPPING.get(ticker, ticker) for ticker in all_tickers]
        
        # Find missing tickers
        missing_tickers = [ticker for ticker in all_tickers if TICKER_MAPPING.get(ticker, ticker) not in market_caps]
        
        if missing_tickers:
            print(f"\nFound {len(missing_tickers)} missing tickers in market cap data: {missing_tickers[:10]}{'...' if len(missing_tickers) > 10 else ''}")
            print("Fetching market cap data for missing tickers...")
            
            # Fetch market cap data for missing tickers
            updated_market_caps = _fetch_historical_market_caps(missing_tickers, '2023-10-01', MARKET_CAP_FILE)
            
            # Update market_caps with new data
            market_caps.update(updated_market_caps)
            
            print(f"Updated market cap data with {len(updated_market_caps)} new tickers")
    
    return market_caps

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

def weighted_jaccard_similarity(set1, set2, weights):
    """
    Calculate weighted Jaccard similarity between two sets using normalized market cap weights.
    
    Args:
        set1 (set): First set of stock tickers
        set2 (set): Second set of stock tickers  
        weights (dict): Dictionary mapping tickers to their weights (market caps)
        
    Returns:
        float: Normalized weighted Jaccard similarity
    """
    # If either set is empty, return 0
    if len(set1) == 0 or len(set2) == 0:
        return 0
    
    # Calculate total weights for each portfolio with ticker mapping
    # More verbose implementation for debugging
    total_weight1 = 0
    missing_weights_set1 = []
    for ticker in set1:
        mapped_ticker = TICKER_MAPPING.get(ticker, ticker)
        weight = weights.get(mapped_ticker, 0)
        if weight == 0:
            missing_weights_set1.append((ticker, mapped_ticker))
        total_weight1 += weight
    
    total_weight2 = 0
    missing_weights_set2 = []
    for ticker in set2:
        mapped_ticker = TICKER_MAPPING.get(ticker, ticker)
        weight = weights.get(mapped_ticker, 0)
        if weight == 0:
            missing_weights_set2.append((ticker, mapped_ticker))
        total_weight2 += weight
        
    # Print debugging information if there are missing weights
    if missing_weights_set1:
        print(f"Warning: {len(missing_weights_set1)} tickers in set1 have no weights:")
        for ticker, mapped_ticker in missing_weights_set1:
            print(f"  - {ticker} (mapped to {mapped_ticker})")
    
    if missing_weights_set2:
        print(f"Warning: {len(missing_weights_set2)} tickers in set2 have no weights:")
        for ticker, mapped_ticker in missing_weights_set2:
            print(f"  - {ticker} (mapped to {mapped_ticker})")
    
    # If either portfolio has zero total weight, return 0
    if total_weight1 == 0 or total_weight2 == 0:
        return 0
    
    # Get all tickers (union)
    all_tickers = set1.union(set2)
    
    # Calculate numerator and denominator for the weighted Jaccard
    numerator = 0
    denominator = 0
    
    for ticker in all_tickers:
        # Apply ticker mapping
        mapped_ticker = TICKER_MAPPING.get(ticker, ticker)
        
        # Calculate normalized weights for each ticker in each portfolio
        weight1 = weights.get(mapped_ticker, 0) / total_weight1 if ticker in set1 else 0
        weight2 = weights.get(mapped_ticker, 0) / total_weight2 if ticker in set2 else 0
        
        # Add min weight to numerator (intersection)
        numerator += min(weight1, weight2)
        
        # Add max weight to denominator (union)
        denominator += max(weight1, weight2)
    
    return numerator / denominator if denominator > 0 else 0

def jaccard_similarity(set1, set2):
    """
    Calculate standard (unweighted) Jaccard similarity between two sets.
    
    Args:
        set1 (set): First set of items
        set2 (set): Second set of items
        
    Returns:
        float: Jaccard similarity (intersection / union)
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_pairwise_similarities(portfolios, weights=None):
    """
    Calculate pairwise similarities for a list of portfolios.
    
    Args:
        portfolios (list): List of portfolios, where each portfolio is a list of stock tickers
        weights (dict, optional): Market cap weights for weighted similarity
        
    Returns:
        tuple: (unweighted_similarities, weighted_similarities)
    """
    # Convert portfolios to sets for faster operations
    portfolio_sets = [set(portfolio) for portfolio in portfolios]
    
    unweighted_similarities = []
    weighted_similarities = []
    
    for i, j in combinations(range(len(portfolio_sets)), 2):
        # Calculate unweighted Jaccard similarity
        unweighted_sim = jaccard_similarity(portfolio_sets[i], portfolio_sets[j])
        unweighted_similarities.append(unweighted_sim)
        
        # Calculate weighted Jaccard similarity if weights provided
        if weights is not None:
            weighted_sim = weighted_jaccard_similarity(portfolio_sets[i], portfolio_sets[j], weights)
            weighted_similarities.append(weighted_sim)
        else:
            weighted_similarities.append(unweighted_sim)
    
    return unweighted_similarities, weighted_similarities

def check_multimodality(histogram_counts, bins):
    """
    Check if a histogram distribution is multimodal by finding peaks.
    
    Args:
        histogram_counts (array): Counts from histogram
        bins (array): Bin edges from histogram
        
    Returns:
        tuple: (number of modes, peak positions)
    """
    # Find peaks in the histogram
    peaks, _ = find_peaks(histogram_counts, height=0.01)
    
    # Get the bin centers for the peaks
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    peak_positions = [bin_centers[p] for p in peaks]
    
    return len(peaks), peak_positions

def plot_comparison_histogram(unweighted_vals, weighted_vals, title, output_path):
    """
    Plot comparison histogram of unweighted vs weighted Jaccard similarities.
    
    Args:
        unweighted_vals (list): Unweighted Jaccard similarity values
        weighted_vals (list): Weighted Jaccard similarity values
        title (str): Title for the plot
        output_path (str): Path to save the plot
        
    Returns:
        dict: Statistics for both methods
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Unweighted histogram
    counts1, bins1, _ = ax1.hist(unweighted_vals, bins=20, density=True, alpha=0.6, color='blue', label='Unweighted')
    mean1 = np.mean(unweighted_vals)
    sd1 = np.std(unweighted_vals)
    ax1.axvline(mean1, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean1:.3f}')
    ax1.set_title(f'Unweighted Jaccard\nMean: {mean1:.3f}, SD: {sd1:.3f}')
    ax1.set_xlabel('Jaccard Similarity')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # Check multimodality for unweighted
    num_modes1, peak_positions1 = check_multimodality(counts1, bins1)
    modality_text1 = "Unimodal" if num_modes1 <= 1 else f"Multimodal ({num_modes1} modes)"
    ax1.text(0.05, 0.95, modality_text1, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Weighted histogram
    counts2, bins2, _ = ax2.hist(weighted_vals, bins=20, density=True, alpha=0.6, color='green', label='Weighted')
    mean2 = np.mean(weighted_vals)
    sd2 = np.std(weighted_vals)
    ax2.axvline(mean2, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean2:.3f}')
    ax2.set_title(f'Market-Cap Weighted Jaccard\nMean: {mean2:.3f}, SD: {sd2:.3f}')
    ax2.set_xlabel('Jaccard Similarity') 
    ax2.set_ylabel('Density')
    ax2.legend()
    
    # Check multimodality for weighted
    num_modes2, peak_positions2 = check_multimodality(counts2, bins2)
    modality_text2 = "Unimodal" if num_modes2 <= 1 else f"Multimodal ({num_modes2} modes)"
    ax2.text(0.05, 0.95, modality_text2, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'unweighted': {
            'mean': mean1,
            'sd': sd1,
            'num_modes': num_modes1,
            'peak_positions': peak_positions1
        },
        'weighted': {
            'mean': mean2,
            'sd': sd2,
            'num_modes': num_modes2,
            'peak_positions': peak_positions2
        }
    }

def get_representative_portfolios(data, market_caps):
    """
    Get representative portfolios for all investor types across different conditions.
    
    Args:
        data (dict): Portfolio data
        market_caps (dict): Market cap weights
        
    Returns:
        dict: Representative portfolios for each condition
    """
    representative_portfolios = {}
    
    for sample_size in SAMPLE_SIZES:
        representative_portfolios[f'sample_{sample_size}'] = {}
        
        for portfolio_size in PORTFOLIO_SIZES:
            representative_portfolios[f'sample_{sample_size}'][f'portfolio_{portfolio_size}'] = {}
            
            print(f"Calculating representative portfolios: {sample_size} samples, {portfolio_size} stocks...")
            
            for investor_type in INVESTOR_TYPES:
                if investor_type not in data:
                    continue
                
                # Get all portfolios for this investor type
                portfolios_full = data[investor_type]
                
                # Use subset if specified
                portfolios = portfolios_full
                if sample_size is not None and len(portfolios_full) > sample_size:
                    portfolios = portfolios_full[:sample_size]
                    print(f"  Using first {sample_size} repetitions for {investor_type}")
                
                # Flatten the list of portfolios to count frequencies
                all_stocks = [stock for portfolio in portfolios for stock in portfolio]
                
                # Count frequencies using Counter
                counter = Counter(all_stocks)
                
                # Get top N stocks by frequency
                top_stocks = [stock for stock, _ in counter.most_common(portfolio_size)]
                
                # Store representative portfolio
                representative_portfolios[f'sample_{sample_size}'][f'portfolio_{portfolio_size}'][investor_type] = top_stocks
                
                print(f"  {investor_type}: {', '.join(top_stocks[:3])}... (top {portfolio_size})")
    
    return representative_portfolios

def calculate_pairwise_similarity_matrix(representative_portfolios, market_caps, sample_size, portfolio_size):
    """
    Calculate pairwise similarity matrix between representative portfolios.
    
    Args:
        representative_portfolios (dict): Representative portfolios
        market_caps (dict): Market cap weights
        sample_size (int): Sample size to analyze
        portfolio_size (int): Portfolio size to analyze
        
    Returns:
        tuple: (unweighted_matrix, weighted_matrix, investor_types)
    """
    condition_key = f'sample_{sample_size}'
    size_key = f'portfolio_{portfolio_size}'
    
    portfolios_dict = representative_portfolios[condition_key][size_key]
    investor_types = list(portfolios_dict.keys())
    n = len(investor_types)
    
    # Initialize matrices
    unweighted_matrix = np.ones((n, n))
    weighted_matrix = np.ones((n, n))
    
    # Calculate pairwise similarities
    for i in range(n):
        for j in range(i+1, n):
            portfolio_i = set(portfolios_dict[investor_types[i]])
            portfolio_j = set(portfolios_dict[investor_types[j]])
            
            # Calculate similarities
            unweighted_sim = jaccard_similarity(portfolio_i, portfolio_j)
            weighted_sim = weighted_jaccard_similarity(portfolio_i, portfolio_j, market_caps)
            
            # Add special logging for comparisons involving Value Investor
            if investor_types[i] == 'value_investor' or investor_types[j] == 'value_investor':
                other_type = investor_types[j] if investor_types[i] == 'value_investor' else investor_types[i]
                print(f"\nComparison between Value Investor and {other_type}:")
                print(f"  Unweighted Jaccard: {unweighted_sim:.4f}")
                print(f"  Weighted Jaccard: {weighted_sim:.4f}")
                print(f"  Difference: {weighted_sim - unweighted_sim:.4f}")
            
            # Fill symmetric matrix
            unweighted_matrix[i, j] = unweighted_sim
            unweighted_matrix[j, i] = unweighted_sim
            
            weighted_matrix[i, j] = weighted_sim
            weighted_matrix[j, i] = weighted_sim
    
    return unweighted_matrix, weighted_matrix, investor_types

def create_similarity_tables(representative_portfolios, market_caps):
    """
    Create similarity tables for all conditions and save them.
    
    Args:
        representative_portfolios (dict): Representative portfolios
        market_caps (dict): Market cap weights
        
    Returns:
        dict: All similarity matrices
    """
    all_matrices = {}
    
    # Create tables directory
    tables_dir = os.path.join(OUTPUT_DIR, 'similarity_tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    for sample_size in SAMPLE_SIZES:
        for portfolio_size in PORTFOLIO_SIZES:
            print(f"Creating similarity table: {sample_size} samples, {portfolio_size} stocks...")
            
            # Calculate matrices
            unweighted_matrix, weighted_matrix, investor_types = calculate_pairwise_similarity_matrix(
                representative_portfolios, market_caps, sample_size, portfolio_size
            )
            
            # Create short names for better display
            short_names = {
                'neutral_investor': 'Neutral',
                'value_investor': 'Value', 
                'growth_investor': 'Growth',
                'momentum_investor': 'Mom.',
                'speculative_trader': 'Spec.',
                'index_mimicker': 'Index',
                'thematic_investor': 'Theme',
                'sentiment_driven_investor': 'Sent.',
                'non_financial_background_investor': 'Non-Fin',
                'low_risk_aversion_investor': 'Low Risk',
                'high_risk_aversion_investor': 'High Risk'
            }
            
            short_investor_names = [short_names.get(inv, inv) for inv in investor_types]
            
            # Create DataFrames
            unweighted_df = pd.DataFrame(unweighted_matrix, 
                                       index=short_investor_names, 
                                       columns=short_investor_names)
            weighted_df = pd.DataFrame(weighted_matrix, 
                                     index=short_investor_names, 
                                     columns=short_investor_names)
            
            # Save to CSV
            unweighted_file = f'jaccard_unweighted_{sample_size}samples_{portfolio_size}stocks.csv'
            weighted_file = f'jaccard_weighted_{sample_size}samples_{portfolio_size}stocks.csv'
            
            unweighted_df.to_csv(os.path.join(tables_dir, unweighted_file))
            weighted_df.to_csv(os.path.join(tables_dir, weighted_file))
            
            # Create heatmap visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Unweighted heatmap
            sns.heatmap(unweighted_df, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1,
                       ax=ax1, cbar_kws={'label': 'Jaccard Similarity'})
            ax1.set_title(f'Unweighted Jaccard Similarity\n({sample_size} samples, {portfolio_size} stocks)')
            
            # Weighted heatmap  
            sns.heatmap(weighted_df, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1,
                       ax=ax2, cbar_kws={'label': 'Weighted Jaccard Similarity'})
            ax2.set_title(f'Market-Cap Weighted Jaccard Similarity\n({sample_size} samples, {portfolio_size} stocks)')
            
            plt.tight_layout()
            heatmap_file = f'jaccard_heatmap_{sample_size}samples_{portfolio_size}stocks.png'
            plt.savefig(os.path.join(tables_dir, heatmap_file), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store results
            condition_key = f'{sample_size}samples_{portfolio_size}stocks'
            all_matrices[condition_key] = {
                'unweighted_matrix': unweighted_matrix,
                'weighted_matrix': weighted_matrix,
                'investor_types': investor_types,
                'short_names': short_investor_names,
                'unweighted_df': unweighted_df,
                'weighted_df': weighted_df
            }
            
            print(f"  Saved: {unweighted_file}, {weighted_file}, {heatmap_file}")
    
    return all_matrices

def create_summary_comparison_plots(all_matrices):
    """
    Create summary plots comparing unweighted vs weighted similarities.
    
    Args:
        all_matrices (dict): All similarity matrices
    """
    print("Creating summary comparison plots...")
    
    # Prepare comparison data
    comparison_data = []
    
    for condition_key, matrices in all_matrices.items():
        # Parse condition
        parts = condition_key.split('_')
        sample_size = int(parts[0].replace('samples', ''))
        portfolio_size = int(parts[1].replace('stocks', ''))
        
        unweighted_matrix = matrices['unweighted_matrix']
        weighted_matrix = matrices['weighted_matrix']
        
        # Get upper triangular values (excluding diagonal)
        n = unweighted_matrix.shape[0]
        unweighted_vals = []
        weighted_vals = []
        
        for i in range(n):
            for j in range(i+1, n):
                unweighted_vals.append(unweighted_matrix[i, j])
                weighted_vals.append(weighted_matrix[i, j])
        
        # Calculate statistics
        unweighted_mean = np.mean(unweighted_vals)
        weighted_mean = np.mean(weighted_vals)
        
        comparison_data.append({
            'sample_size': sample_size,
            'portfolio_size': portfolio_size,
            'unweighted_mean': unweighted_mean,
            'weighted_mean': weighted_mean,
            'mean_difference': weighted_mean - unweighted_mean,
            'unweighted_std': np.std(unweighted_vals),
            'weighted_std': np.std(weighted_vals),
            'n_comparisons': len(unweighted_vals)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'similarity_comparison_summary.csv'), index=False)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean comparison by portfolio size
    ax1 = axes[0, 0]
    portfolio_means = comparison_df.groupby('portfolio_size')[['unweighted_mean', 'weighted_mean']].mean()
    portfolio_means.plot(kind='bar', ax=ax1)
    ax1.set_title('Average Jaccard Similarity by Portfolio Size')
    ax1.set_xlabel('Portfolio Size')
    ax1.set_ylabel('Mean Jaccard Similarity')
    ax1.legend(['Unweighted', 'Market-Cap Weighted'])
    ax1.tick_params(axis='x', rotation=0)
    
    # Plot 2: Mean comparison by sample size
    ax2 = axes[0, 1]
    sample_means = comparison_df.groupby('sample_size')[['unweighted_mean', 'weighted_mean']].mean()
    sample_means.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Jaccard Similarity by Sample Size')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Mean Jaccard Similarity')
    ax2.legend(['Unweighted', 'Market-Cap Weighted'])
    ax2.tick_params(axis='x', rotation=0)
    
    # Plot 3: Difference distribution
    ax3 = axes[1, 0]
    ax3.hist(comparison_df['mean_difference'], bins=10, alpha=0.7, color='purple')
    ax3.set_title('Distribution of Mean Differences\n(Weighted - Unweighted)')
    ax3.set_xlabel('Mean Difference')
    ax3.set_ylabel('Frequency')
    ax3.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: Scatter plot of unweighted vs weighted
    ax4 = axes[1, 1]
    ax4.scatter(comparison_df['unweighted_mean'], comparison_df['weighted_mean'], 
               c=comparison_df['portfolio_size'], cmap='viridis', s=100)
    ax4.plot([0, 1], [0, 1], 'r--', alpha=0.7)
    ax4.set_xlabel('Unweighted Mean')
    ax4.set_ylabel('Weighted Mean')
    ax4.set_title('Unweighted vs Weighted Similarity')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Portfolio Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df

def generate_summary_report(comparison_df, all_matrices):
    """
    Generate a comprehensive summary report.
    
    Args:
        comparison_df (pd.DataFrame): Comparison data
        all_matrices (dict): All similarity matrices
    """
    print("\n" + "="*80)
    print("WEIGHTED JACCARD SIMILARITY ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\nOVERALL RESULTS:")
    print(f"  Total conditions analyzed: {len(comparison_df)}")
    print(f"  Portfolio sizes tested: {sorted(comparison_df['portfolio_size'].unique())}")
    print(f"  Sample sizes tested: {sorted(comparison_df['sample_size'].unique())}")
    print(f"  Number of pairwise comparisons per condition: {comparison_df['n_comparisons'].iloc[0]} (11x11 matrix)")
    
    # Mean differences
    print(f"\nMEAN JACCARD SIMILARITY:")
    print(f"  Average unweighted: {comparison_df['unweighted_mean'].mean():.4f}")
    print(f"  Average weighted: {comparison_df['weighted_mean'].mean():.4f}")
    print(f"  Average difference (weighted - unweighted): {comparison_df['mean_difference'].mean():.4f}")
    print(f"  Percentage change: {(comparison_df['mean_difference'].mean() / comparison_df['unweighted_mean'].mean() * 100):.2f}%")
    
    # By portfolio size
    print(f"\nBY PORTFOLIO SIZE:")
    for size in sorted(comparison_df['portfolio_size'].unique()):
        subset = comparison_df[comparison_df['portfolio_size'] == size]
        print(f"  Size {size}:")
        print(f"    Unweighted mean: {subset['unweighted_mean'].mean():.4f}")
        print(f"    Weighted mean: {subset['weighted_mean'].mean():.4f}")
        print(f"    Difference: {subset['mean_difference'].mean():.4f}")
    
    # By sample size
    print(f"\nBY SAMPLE SIZE:")
    for size in sorted(comparison_df['sample_size'].unique()):
        subset = comparison_df[comparison_df['sample_size'] == size]
        print(f"  Size {size}:")
        print(f"    Unweighted mean: {subset['unweighted_mean'].mean():.4f}")
        print(f"    Weighted mean: {subset['weighted_mean'].mean():.4f}")
        print(f"    Difference: {subset['mean_difference'].mean():.4f}")
    
    # Key findings
    print(f"\nKEY FINDINGS:")
    if comparison_df['mean_difference'].mean() > 0.01:
        print("  ✓ Market-cap weighting significantly increases Jaccard similarity")
    elif comparison_df['mean_difference'].mean() < -0.01:
        print("  ✓ Market-cap weighting significantly decreases Jaccard similarity")
    else:
        print("  ✓ Market-cap weighting has minimal impact on Jaccard similarity")
    
    if abs(comparison_df.groupby('portfolio_size')['mean_difference'].mean().max() - 
           comparison_df.groupby('portfolio_size')['mean_difference'].mean().min()) > 0.01:
        print("  ✓ Portfolio size affects the impact of market-cap weighting")
    else:
        print("  ✓ Portfolio size does not significantly affect market-cap weighting impact")
    
    if abs(comparison_df.groupby('sample_size')['mean_difference'].mean().max() - 
           comparison_df.groupby('sample_size')['mean_difference'].mean().min()) > 0.005:
        print("  ✓ Sample size affects the results")
    else:
        print("  ✓ Results are stable across different sample sizes")
    
    print("="*80)

def main():
    """Main function to run the comprehensive analysis."""
    print("Starting comprehensive weighted Jaccard similarity analysis...")
    print(f"Portfolio sizes: {PORTFOLIO_SIZES}")
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    print(f"\nLoading portfolio data from {RESULT_FILE}...")
    with open(RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Found data for {len(data)} investor types")
    for investor_type in data.keys():
        print(f"  - {investor_type}: {len(data[investor_type])} repetitions")
    
    # Collect all unique tickers from all portfolios
    all_tickers = set()
    for investor_type, portfolios in data.items():
        for portfolio in portfolios:
            all_tickers.update(portfolio)
    
    print(f"\nCollected {len(all_tickers)} unique tickers from all portfolios")
    
    # Load market caps and check for missing tickers
    market_caps = load_market_caps(list(all_tickers))
    
    # Get representative portfolios for all conditions
    print(f"\nCalculating representative portfolios...")
    representative_portfolios = get_representative_portfolios(data, market_caps)
    
    # Create similarity tables and matrices
    print(f"\nCreating similarity tables...")
    all_matrices = create_similarity_tables(representative_portfolios, market_caps)
    
    # Create summary comparisons
    print(f"\nCreating summary comparison plots...")
    comparison_df = create_summary_comparison_plots(all_matrices)
    
    # Generate summary report
    generate_summary_report(comparison_df, all_matrices)
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  - similarity_tables/: 12 CSV files (6 unweighted + 6 weighted)")
    print(f"  - similarity_tables/: 6 heatmap PNG files")
    print(f"  - similarity_comparison_summary.csv: Summary comparison data")
    print(f"  - similarity_comparison_summary.png: Overall comparison plots")
    print(f"  - representative_portfolios.json: All representative portfolios")
    
    # Save representative portfolios for reference
    with open(os.path.join(OUTPUT_DIR, 'representative_portfolios.json'), 'w') as f:
        json.dump(representative_portfolios, f, indent=2)

if __name__ == "__main__":
    main()