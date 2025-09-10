import os
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.signal import find_peaks
import pandas as pd
import seaborn as sns
from collections import defaultdict

# Constants
RESULT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Prompts/prompts_repetition.json'
OUTPUT_DIR = '/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Jaccard_Analysis'
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

def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets.
    
    Args:
        set1 (set): First set of items
        set2 (set): Second set of items
        
    Returns:
        float: Jaccard similarity (intersection / union)
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_pairwise_jaccard(portfolios):
    """
    Calculate pairwise Jaccard similarities for a list of portfolios.
    
    Args:
        portfolios (list): List of portfolios, where each portfolio is a list of stock tickers
        
    Returns:
        list: List of Jaccard similarity values
    """
    # Convert portfolios to sets for faster intersection/union operations
    portfolio_sets = [set(portfolio) for portfolio in portfolios]
    
    # Calculate pairwise Jaccard similarities
    jaccard_values = []
    for i, j in combinations(range(len(portfolio_sets)), 2):
        similarity = jaccard_similarity(portfolio_sets[i], portfolio_sets[j])
        jaccard_values.append(similarity)
    
    return jaccard_values

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

def plot_histogram(jaccard_values, title, output_path, check_modes=True):
    """
    Plot histogram of Jaccard similarity values and check for multimodality.
    
    Args:
        jaccard_values (list): List of Jaccard similarity values
        title (str): Title for the plot
        output_path (str): Path to save the plot
        check_modes (bool): Whether to check for multimodality
        
    Returns:
        tuple: (mean, standard deviation, number of modes, peak positions)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate histogram
    counts, bins, _ = ax.hist(jaccard_values, bins=20, density=True, alpha=0.6, color='b')
    
    # Calculate statistics
    mean_jac = np.mean(jaccard_values)
    sd_jac = np.std(jaccard_values)
    
    # Add mean line
    ax.axvline(mean_jac, color='r', linestyle='dashed', linewidth=1)
    
    # Set title and labels
    ax.set_title(f'{title}\nMean: {mean_jac:.3f}, SD: {sd_jac:.3f}')
    ax.set_xlabel('Jaccard Similarity')
    ax.set_ylabel('Density')
    
    # Check for multimodality if requested
    num_modes = 1
    peak_positions = []
    if check_modes:
        num_modes, peak_positions = check_multimodality(counts, bins)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return mean_jac, sd_jac, num_modes, peak_positions

def analyze_portfolios(data, subset_size=None, output_suffix=""):
    """
    Analyze portfolios by calculating Jaccard similarities and generating histograms.
    
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
        
        # Skip if too few portfolios
        if len(portfolios) < 2:
            print(f"    Skipping {investor_type} - not enough portfolios")
            continue
        
        # Calculate pairwise Jaccard similarities
        jaccard_values = calculate_pairwise_jaccard(portfolios)
        
        # Plot histogram for this investor type
        title = f"{investor_type} - Repetition Analysis (n={len(portfolios)})"
        output_path = os.path.join(OUTPUT_DIR, f"{investor_type}{output_suffix}.png")
        mean_jac, sd_jac, num_modes, peak_positions = plot_histogram(jaccard_values, title, output_path)
        
        # Store results
        results[investor_type] = {
            "mean": mean_jac,
            "sd": sd_jac,
            "num_modes": num_modes,
            "peak_positions": peak_positions,
            "n_portfolios": len(portfolios),
            "n_comparisons": len(jaccard_values)
        }
        
        # Add to summary data
        summary_data.append({
            "investor_type": investor_type,
            "mean": mean_jac,
            "sd": sd_jac,
            "num_modes": num_modes,
            "multimodal": num_modes > 1,
            "n_portfolios": len(portfolios),
            "n_comparisons": len(jaccard_values)
        })
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(OUTPUT_DIR, f"jaccard_summary{output_suffix}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Create summary table with multimodality counts
    multimodal_counts = summary_df.groupby('investor_type')['multimodal'].sum().reset_index()
    multimodal_counts['total_rephrases'] = summary_df.groupby('investor_type').size().values
    multimodal_counts['multimodal_percentage'] = (multimodal_counts['multimodal'] / multimodal_counts['total_rephrases'] * 100).round(1)
    multimodal_counts.to_csv(os.path.join(OUTPUT_DIR, f"multimodality_summary{output_suffix}.csv"), index=False)
    
    # Save detailed results to JSON
    results_json_path = os.path.join(OUTPUT_DIR, f"jaccard_results{output_suffix}.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_comparison_plots(results_50, results_100):
    """
    Create comparison plots between 50-rep and 100-rep results.
    
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
                "mean_50": r50["mean"],
                "mean_100": r100["mean"],
                "mean_diff": r100["mean"] - r50["mean"],
                "mean_diff_pct": ((r100["mean"] - r50["mean"]) / r50["mean"] * 100) if r50["mean"] != 0 else 0,
                "sd_50": r50["sd"],
                "sd_100": r100["sd"],
                "sd_diff": r100["sd"] - r50["sd"],
                "modes_50": r50["num_modes"],
                "modes_100": r100["num_modes"],
                "modes_changed": r50["num_modes"] != r100["num_modes"]
            })
    
    # Create comparison DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(OUTPUT_DIR, "jaccard_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    
    # Create summary statistics
    mean_diff_avg = comparison_df["mean_diff"].mean()
    mean_diff_pct_avg = comparison_df["mean_diff_pct"].mean()
    sd_diff_avg = comparison_df["sd_diff"].mean()
    modes_changed_count = comparison_df["modes_changed"].sum()
    modes_changed_pct = (modes_changed_count / len(comparison_df) * 100)
    
    # Save summary statistics
    summary = {
        "mean_diff_avg": mean_diff_avg,
        "mean_diff_pct_avg": mean_diff_pct_avg,
        "sd_diff_avg": sd_diff_avg,
        "modes_changed_count": int(modes_changed_count),
        "modes_changed_pct": modes_changed_pct,
        "total_comparisons": len(comparison_df)
    }
    
    with open(os.path.join(OUTPUT_DIR, "comparison_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison plots
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=comparison_df, x="mean_50", y="mean_100", hue="investor_type")
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title("Jaccard Similarity Mean: 50 reps vs 100 reps")
    plt.xlabel("Mean (50 reps)")
    plt.ylabel("Mean (100 reps)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_comparison.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=comparison_df, x="sd_50", y="sd_100", hue="investor_type")
    plt.plot([0, 0.2], [0, 0.2], 'k--')  # Diagonal line
    plt.title("Jaccard Similarity SD: 50 reps vs 100 reps")
    plt.xlabel("SD (50 reps)")
    plt.ylabel("SD (100 reps)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sd_comparison.png"))
    plt.close()
    
    # Create mode change visualization
    mode_changes = comparison_df.groupby(["modes_50", "modes_100"]) .size().reset_index()
    mode_changes.columns = ["modes_50", "modes_100", "count"]
    
    plt.figure(figsize=(8, 6))
    for _, row in mode_changes.iterrows():
        plt.text(row["modes_50"], row["modes_100"], str(row["count"]), 
                 ha='center', va='center', size=14)
    
    plt.scatter(mode_changes["modes_50"], mode_changes["modes_100"], 
                s=mode_changes["count"]*50, alpha=0.5)
    plt.plot([0, 5], [0, 5], 'k--')  # Diagonal line
    plt.title("Number of Modes: 50 reps vs 100 reps")
    plt.xlabel("Modes (50 reps)")
    plt.ylabel("Modes (100 reps)")
    plt.xticks(range(1, 5))
    plt.yticks(range(1, 5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "modes_comparison.png"))
    plt.close()
    
    return summary


def compute_convergence_table(data, ns=None):
    """Compute convergence table of mean Jaccard by increasing sample size.
    Args:
        data (dict): {investor_type: [portfolios...]}
        ns (list[int]|None): list of sample sizes; default [50,60,...,100]
    Returns:
        pd.DataFrame: index=n, columns=investor types, values=mean jaccard
    """
    if ns is None:
        ns = list(range(50, 101, 10))
    # Ensure only valid sizes (>=2) and ascending unique
    ns = sorted({n for n in ns if n >= 2})

    rows = []
    for n in ns:
        row = {"n": n}
        for investor_type in INVESTOR_TYPES:
            if investor_type not in data:
                continue
            portfolios = data[investor_type]
            if len(portfolios) < 2:
                continue
            n_use = min(n, len(portfolios))
            jvals = calculate_pairwise_jaccard(portfolios[:n_use])
            row[investor_type] = float(np.mean(jvals)) if len(jvals) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("n").sort_index()
    # Save to CSV and Markdown for paper inclusion
    csv_path = os.path.join(OUTPUT_DIR, "convergence_table.csv")
    df.to_csv(csv_path)
    try:
        md_path = os.path.join(OUTPUT_DIR, "convergence_table.md")
        with open(md_path, "w") as f:
            f.write(df.round(3).to_markdown())
    except Exception:
        pass
    return df

def main():
    """Main function to run the analysis"""
    print(f"Loading portfolio data from {RESULT_FILE}...")
    with open(RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Found data for {len(data)} investor types")
    for investor_type in data.keys():
        print(f"  - {investor_type}: {len(data[investor_type])} repetitions")
    
    # Analyze with 50 reps (subset)
    print(f"\nAnalyzing with {SUBSET_SIZE} reps (subset)...")
    results_50 = analyze_portfolios(data, subset_size=SUBSET_SIZE, output_suffix="_50reps")
    
    # Analyze with all reps (100)
    print("\nAnalyzing with all reps (100)...")
    results_100 = analyze_portfolios(data, subset_size=None, output_suffix="_100reps")
    
    # Create comparison between 50 and 100 reps
    print("\nCreating comparison between 50 and 100 reps...")
    comparison_summary = create_comparison_plots(results_50, results_100)

    # Convergence test: compute mean Jaccard as n increases 50->100
    print("\nComputing convergence table (n=50,60,...,100)...")
    conv_df = compute_convergence_table(data)
    print(f"Convergence table saved to {os.path.join(OUTPUT_DIR, 'convergence_table.csv')}")

    # Create a concise stability summary (deltas and stabilization threshold)
    try:
        rows = []
        ns = list(conv_df.index)
        for inv in [c for c in conv_df.columns if c in INVESTOR_TYPES]:
            vals = conv_df[inv]
            entry = {"investor_type": inv}
            for target in [80, 90, 100]:
                if target in conv_df.index:
                    entry[f"mean_{target}"] = float(vals.loc[target])
            if 80 in conv_df.index and 90 in conv_df.index:
                entry["delta_80_90"] = float(vals.loc[90] - vals.loc[80])
            if 90 in conv_df.index and 100 in conv_df.index:
                entry["delta_90_100"] = float(vals.loc[100] - vals.loc[90])
            if 80 in conv_df.index and 100 in conv_df.index:
                entry["abs_delta_80_100"] = float(abs(vals.loc[100] - vals.loc[80]))
            # Find smallest n where |mean_n - mean_100| < 0.01 (stable at 2nd decimal)
            stabilized_from = None
            if 100 in conv_df.index and not np.isnan(vals.loc[100]):
                m100 = float(vals.loc[100])
                for n in ns:
                    v = vals.loc[n]
                    if not np.isnan(v) and abs(float(v) - m100) < 0.01:
                        stabilized_from = n
                        break
            entry["stabilized_from_n_(<0.01)"] = stabilized_from
            rows.append(entry)
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "convergence_changes.csv"), index=False)
        print(f"Convergence changes saved to {os.path.join(OUTPUT_DIR, 'convergence_changes.csv')}")
    except Exception as e:
        print(f"Convergence summary generation failed: {e}")
    
    print("\nAnalysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")
    
    # Print key findings
    print("\nKey findings:")
    print(f"- Average change in mean Jaccard similarity: {comparison_summary['mean_diff_avg']:.4f} ({comparison_summary['mean_diff_pct_avg']:.1f}%)")
    print(f"- Average change in standard deviation: {comparison_summary['sd_diff_avg']:.4f}")
    print(f"- Number of investor types with changed modality: {comparison_summary['modes_changed_count']} out of {comparison_summary['total_comparisons']} ({comparison_summary['modes_changed_pct']:.1f}%)")
    print("- Convergence: See convergence_table.csv and convergence_changes.csv for stabilization around n=80-90.")
    
    # Determine if there's significant multimodality
    multimodal_summary_path = os.path.join(OUTPUT_DIR, "multimodality_summary_50reps.csv")
    if os.path.exists(multimodal_summary_path):
        multimodal_df = pd.read_csv(multimodal_summary_path)
        total_multimodal = multimodal_df['multimodal'].sum()
        total_investor_types = len(multimodal_df)
        multimodal_pct = (total_multimodal / total_investor_types * 100)
        
        print(f"- Multimodal distributions: {total_multimodal} out of {total_investor_types} investor types ({multimodal_pct:.1f}%)")
        
        if multimodal_pct > 20:
            print("  FINDING: Significant multimodality detected in the distributions.")
            print("  INTERPRETATION: The variation in portfolios shows multiple distinct patterns/clusters.")
        else:
            print("  FINDING: Limited multimodality in the distributions.")
            print("  INTERPRETATION: The variation in portfolios appears to be random rather than clustered.")

if __name__ == "__main__":
    main()