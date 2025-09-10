import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
import math

# Constants
RESULT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Prompts/prompts_repetition.json'
OUTPUT_DIR = '/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Shannon_Analysis'
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
SUBSET_SIZE = 50  # Size of the subset for baseline analysis (from 100 repetitions)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def shannon_entropy(stock_list):
    """
    Calculate Shannon entropy for a list of stocks.

    Shannon entropy measures the information content or uncertainty in a distribution.

    Args:
        stock_list (list): List of stocks

    Returns:
        float: Shannon entropy value
    """
    c = Counter(stock_list)
    total = sum(c.values())
    probs = [count / total for count in c.values()]
    return -sum(p * math.log2(p) for p in probs)

def calculate_collective_shannon_entropy(portfolios):
    """
    Calculate Shannon entropy for combined stocks from all portfolios (new.py method).
    This combines all stocks from all portfolios into a single list and calculates 
    entropy from the collective frequency distribution.
    
    Args:
        portfolios (list): List of portfolios, where each portfolio is a list of stock tickers
        
    Returns:
        float: Single Shannon entropy value for the collective distribution
    """
    # Combine all stocks from all portfolios into one list (new.py approach)
    all_stocks = [stock for portfolio in portfolios for stock in portfolio if portfolio]
    
    if not all_stocks:
        return 0.0
    
    # Calculate entropy from the collective frequency distribution
    entropy_value = shannon_entropy(all_stocks)
    return entropy_value


def analyze_portfolios(data, subset_size=None, output_suffix=""):
    """
    Analyze portfolios by calculating Shannon entropies and generating histograms.
    
    Args:
        data (dict): Portfolio data dictionary with investor types as keys and lists of portfolios as values
        subset_size (int, optional): Size of subset to use. If None, use all data.
        output_suffix (str): Suffix to add to output filenames
        
    Returns:
        dict: Results dictionary with statistics for each investor type
    """
    results = defaultdict(dict)
    summary_data = []
    
    for investor_type in INVESTOR_TYPES:
        if investor_type not in data:
            print(f"Warning: {investor_type} not found in data. Skipping...")
            continue
            
        print(f"Processing {investor_type}...")
        
        # Create investor-specific directory
        investor_dir = os.path.join(OUTPUT_DIR, f"{investor_type}{output_suffix}")
        os.makedirs(investor_dir, exist_ok=True)
        
        # Get portfolios for this investor type
        portfolios = data[investor_type]
        print(f"  Found {len(portfolios)} repetitions")
        
        # Use subset if specified
        if subset_size is not None and len(portfolios) > subset_size:
            portfolios = portfolios[:subset_size]
            print(f"  Using first {subset_size} repetitions")
        
        # Skip if no portfolios
        if len(portfolios) < 1:
            print(f"    Skipping {investor_type} - no portfolios")
            continue
        
        # Calculate collective Shannon entropy using new.py method
        entropy_value = calculate_collective_shannon_entropy(portfolios)
        
        if entropy_value == 0.0:
            print(f"    Skipping {investor_type} - no valid entropy value")
            continue
        
        print(f"  Collective Shannon entropy: {entropy_value:.4f}")
        
        # Store results (single entropy value per investor type)
        results[investor_type] = {
            "entropy": entropy_value,
            "n_portfolios": len(portfolios),
            "n_total_stocks": sum(len(portfolio) for portfolio in portfolios if portfolio)
        }
        
        # Add to summary data
        summary_data.append({
            "investor_type": investor_type,
            "entropy": entropy_value,
            "n_portfolios": len(portfolios),
            "n_total_stocks": sum(len(portfolio) for portfolio in portfolios if portfolio)
        })
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(OUTPUT_DIR, f"shannon_summary{output_suffix}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Save detailed results to JSON
    results_json_path = os.path.join(OUTPUT_DIR, f"shannon_results{output_suffix}.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_comparison_plots(results_50, results_100):
    """
    Create comparison plots between 50-rep and 100-rep results.
    Now compares single entropy values instead of distributions.
    
    Args:
        results_50 (dict): Results from 50-rep analysis
        results_100 (dict): Results from 100-rep analysis
    """
    comparison_data = []
    
    for investor_type in INVESTOR_TYPES:
        if investor_type in results_50 and investor_type in results_100:
            r50 = results_50[investor_type]
            r100 = results_100[investor_type]
            
            comparison_data.append({
                "investor_type": investor_type,
                "entropy_50": r50["entropy"],
                "entropy_100": r100["entropy"],
                "entropy_diff": r100["entropy"] - r50["entropy"],
                "entropy_diff_pct": ((r100["entropy"] - r50["entropy"]) / r50["entropy"] * 100) if r50["entropy"] != 0 else 0,
                "n_portfolios_50": r50["n_portfolios"],
                "n_portfolios_100": r100["n_portfolios"],
                "n_stocks_50": r50["n_total_stocks"],
                "n_stocks_100": r100["n_total_stocks"]
            })
    
    # Create comparison DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(OUTPUT_DIR, "shannon_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    
    # Create summary statistics
    entropy_diff_avg = comparison_df["entropy_diff"].mean()
    entropy_diff_pct_avg = comparison_df["entropy_diff_pct"].mean()
    
    # Save summary statistics
    summary = {
        "entropy_diff_avg": entropy_diff_avg,
        "entropy_diff_pct_avg": entropy_diff_pct_avg,
        "total_comparisons": len(comparison_df)
    }
    
    with open(os.path.join(OUTPUT_DIR, "comparison_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison plots for single entropy values
    plt.figure(figsize=(12, 6))
    x_pos = range(len(comparison_df))
    
    # Bar chart comparing entropy values
    plt.bar([x - 0.2 for x in x_pos], comparison_df["entropy_50"], width=0.4, label="50 reps", alpha=0.7, color='skyblue')
    plt.bar([x + 0.2 for x in x_pos], comparison_df["entropy_100"], width=0.4, label="100 reps", alpha=0.7, color='lightcoral')
    
    plt.xticks(x_pos, comparison_df["investor_type"], rotation=45, ha='right')
    plt.ylabel("Shannon Entropy (Collective)")
    plt.title("Shannon Entropy Comparison: 50 vs 100 Repetitions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "entropy_comparison.png"))
    plt.close()
    
    # Create scatter plot showing relationship
    plt.figure(figsize=(10, 6))
    plt.scatter(comparison_df["entropy_50"], comparison_df["entropy_100"], alpha=0.7, s=100)
    
    # Add diagonal line
    min_entropy = min(comparison_df["entropy_50"].min(), comparison_df["entropy_100"].min())
    max_entropy = max(comparison_df["entropy_50"].max(), comparison_df["entropy_100"].max())
    plt.plot([min_entropy, max_entropy], [min_entropy, max_entropy], 'k--', alpha=0.5)
    
    # Add investor type labels
    for i, row in comparison_df.iterrows():
        plt.annotate(row["investor_type"], 
                    (row["entropy_50"], row["entropy_100"]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.xlabel("Shannon Entropy (50 reps)")
    plt.ylabel("Shannon Entropy (100 reps)")
    plt.title("Shannon Entropy: 50 vs 100 Repetitions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "entropy_scatter_comparison.png"))
    plt.close()
    
    return summary

def main():
    """Main function to run the Shannon entropy analysis"""
    print(f"Loading portfolio data from {RESULT_FILE}...")
    with open(RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Found data for {len(data)} investor types")
    for investor_type in data.keys():
        print(f"  - {investor_type}: {len(data[investor_type])} repetitions")
    
    # Analyze with 50 reps (subset)
    print(f"\nAnalyzing Shannon entropy with {SUBSET_SIZE} reps (subset)...")
    results_50 = analyze_portfolios(data, subset_size=SUBSET_SIZE, output_suffix="_50reps")
    
    # Analyze with all reps (100)
    print("\nAnalyzing Shannon entropy with all reps (100)...")
    results_100 = analyze_portfolios(data, subset_size=None, output_suffix="_100reps")
    
    # Create comparison between 50 and 100 reps
    print("\nCreating comparison between 50 and 100 reps...")
    comparison_summary = create_comparison_plots(results_50, results_100)
    
    print("\nShannon entropy analysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")
    
    # Print key findings
    print("\nKey findings:")
    print(f"- Average change in Shannon entropy: {comparison_summary['entropy_diff_avg']:.4f} ({comparison_summary['entropy_diff_pct_avg']:.1f}%)")
    print(f"- Total comparisons: {comparison_summary['total_comparisons']} investor types")
    
    # Calculate overall statistics
    if results_50 and results_100:
        print("\nOverall Shannon Entropy Statistics (Collective Method):")
        print("50 repetitions:")
        entropy_50_avg = np.mean([r["entropy"] for r in results_50.values()])
        print(f"  - Average collective entropy: {entropy_50_avg:.3f}")
        
        print("100 repetitions:")
        entropy_100_avg = np.mean([r["entropy"] for r in results_100.values()])
        print(f"  - Average collective entropy: {entropy_100_avg:.3f}")
        
        print("\nInterpretation:")
        print("- Higher Shannon entropy indicates more diverse/uniform stock selection across all portfolios")
        print("- Lower Shannon entropy indicates more concentrated stock selection patterns")
        print("- This method measures collective diversity rather than individual portfolio diversity")
        if entropy_100_avg > entropy_50_avg:
            print("- 100 repetitions show slightly higher collective entropy than 50 repetitions")
        else:
            print("- 50 repetitions show similar or higher collective entropy than 100 repetitions")

if __name__ == "__main__":
    main()