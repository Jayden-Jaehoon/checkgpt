import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tabulate import tabulate
import os

# Constants
RESULT_FILE = '/results/Rephrase/Rephrase_Repetition_Result_NVG.json'
MARKET_CAP_FILE = '/rivision/Performance_Analysis/market_caps_20231001.csv'
OUTPUT_DIR = '/rivision/Rephrase_Variability/openai_weighted_jaccard'

# Portfolio sizes to analyze
PORTFOLIO_SIZES = [10, 20, 30]

def load_market_caps():
    """
    Load market capitalization data from CSV file.
    
    Returns:
        dict: Dictionary mapping tickers to their market caps
    """
    print(f"Loading market cap data from {MARKET_CAP_FILE}...")
    market_caps_df = pd.read_csv(MARKET_CAP_FILE, index_col=0)
    market_caps = market_caps_df['marketCap'].to_dict()
    print(f"Loaded market cap data for {len(market_caps)} tickers")
    return market_caps

def extract_stocks_from_portfolio(portfolio):
    """
    Extract the stock tickers from a portfolio.
    Each portfolio is a list of stock tickers.
    """
    return portfolio

def calculate_representative_portfolio(portfolios, top_n=30):
    """
    Calculate a representative portfolio from a list of portfolios.
    The representative portfolio consists of the top_n most frequently occurring stocks.

    Args:
        portfolios (list): List of portfolios, where each portfolio is a list of stock tickers
        top_n (int): Number of top stocks to include in the representative portfolio

    Returns:
        list: List of stock tickers representing the representative portfolio
    """
    # Extract all stocks from all portfolios
    all_stocks = []
    for portfolio in portfolios:
        all_stocks.extend(extract_stocks_from_portfolio(portfolio))

    # Count the frequency of each stock
    stock_counts = Counter(all_stocks)

    # Get the top_n most common stocks
    top_stocks = stock_counts.most_common(top_n)

    # Create the representative portfolio
    representative_portfolio = [stock for stock, count in top_stocks]

    return representative_portfolio

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

def weighted_jaccard_similarity(set1, set2, weights):
    """
    Calculate weighted Jaccard similarity between two sets using market cap weights.
    
    Args:
        set1 (set): First set of stock tickers
        set2 (set): Second set of stock tickers  
        weights (dict): Dictionary mapping tickers to their weights (market caps)
        
    Returns:
        float: Weighted Jaccard similarity
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if len(union) == 0:
        return 0
    
    # Calculate weighted intersection and union
    intersection_weight = sum(weights.get(ticker, 0) for ticker in intersection)
    union_weight = sum(weights.get(ticker, 0) for ticker in union)
    
    return intersection_weight / union_weight if union_weight > 0 else 0

def calculate_similarity_matrix(prompt_data, market_caps, portfolio_size=30, weighted=False):
    """
    Calculate the Jaccard similarity matrix for a given prompt's rephrases.
    
    Args:
        prompt_data (dict): Dictionary containing rephrase data
        market_caps (dict): Dictionary mapping tickers to their market caps
        portfolio_size (int): Size of the representative portfolio
        weighted (bool): Whether to use weighted Jaccard similarity
        
    Returns:
        numpy.ndarray: Jaccard similarity matrix
    """
    num_rephrases = 10
    matrix = np.zeros((num_rephrases, num_rephrases))

    # Extract stock sets for each rephrase
    stock_sets = []
    for i in range(1, num_rephrases + 1):
        rephrase_key = f"rephrase_{i}"
        if rephrase_key in prompt_data:
            # Calculate representative portfolio for this rephrase
            portfolios = prompt_data[rephrase_key]
            representative_portfolio = calculate_representative_portfolio(portfolios, top_n=portfolio_size)
            stock_sets.append(set(representative_portfolio))
        else:
            print(f"Warning: {rephrase_key} not found in prompt data")
            stock_sets.append(set())

    # Calculate Jaccard similarity for each pair of rephrases
    for i in range(num_rephrases):
        for j in range(num_rephrases):
            if i == j:
                matrix[i][j] = 1.0  # A set is identical to itself
            else:
                if weighted:
                    matrix[i][j] = weighted_jaccard_similarity(stock_sets[i], stock_sets[j], market_caps)
                else:
                    matrix[i][j] = jaccard_similarity(stock_sets[i], stock_sets[j])

    return matrix

def plot_combined_clustermaps(matrices, titles, output_path, weighted=False):
    """
    Plot multiple clustermaps for comparison.
    
    Args:
        matrices (list): List of similarity matrices
        titles (list): List of titles for each matrix
        output_path (str): Path to save the output
        weighted (bool): Whether these are weighted Jaccard matrices
    """
    n_matrices = len(matrices)

    # Create individual clustermaps
    clustergrids = []
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        # Create a DataFrame for the matrix
        df = pd.DataFrame(matrix)
        df.columns = [f"R{j+1}" for j in range(len(matrix))]
        df.index = [f"R{j+1}" for j in range(len(matrix))]

        # Create the clustermap
        cmap = sns.color_palette("YlGnBu", as_cmap=True)

        # Create a separate figure for each clustermap
        plt.figure(figsize=(6, 5))
        clustergrid = sns.clustermap(
            df,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            figsize=(8, 6),
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Jaccard Similarity"}
        )

        # Add title
        similarity_type = "Weighted Jaccard" if weighted else "Jaccard"
        plt.suptitle(f"{similarity_type} Similarity Matrix for {title}", fontsize=14, y=1.02)

        # Save individual clustermap
        individual_path = output_path.replace('.png', f'_{title.replace(" ", "_")}.png')
        plt.savefig(individual_path, bbox_inches='tight', dpi=300)
        print(f"Clustermap for {title} saved to {individual_path}")

        # Close the figure to free memory
        plt.close()

        clustergrids.append(clustergrid)

    # Create a combined figure with all matrices (without clustering)
    # This is a simplified view for comparison
    fig, axes = plt.subplots(1, n_matrices, figsize=(n_matrices * 6, 5))

    if n_matrices == 1:
        axes = [axes]  # Make axes iterable if there's only one matrix

    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        # Create a DataFrame for the matrix
        df = pd.DataFrame(matrix)
        df.columns = [f"R{j+1}" for j in range(len(matrix))]
        df.index = [f"R{j+1}" for j in range(len(matrix))]

        # Create a simple heatmap for the combined view
        sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f", ax=axes[i])
        axes[i].set_title(title)

    similarity_type = "Weighted Jaccard" if weighted else "Jaccard"
    plt.suptitle(f"{similarity_type} Similarity Comparison", fontsize=16, y=1.05)
    plt.tight_layout()

    # Save the combined figure
    combined_path = output_path.replace('.png', '_combined.png')
    plt.savefig(combined_path)
    print(f"Combined heatmap overview saved to {combined_path}")

    # Close the figure to free memory
    plt.close()

    return clustergrids

def format_matrix_as_table(matrix):
    """
    Format the similarity matrix as a table.
    """
    num_rephrases = len(matrix)
    headers = [""] + [f"R{i+1}" for i in range(num_rephrases)]
    table = []

    for i in range(num_rephrases):
        row = [f"R{i+1}"]
        for j in range(num_rephrases):
            if i == j:
                row.append("-")
            else:
                row.append(f"{matrix[i][j]:.2f}")
        table.append(row)

    return tabulate(table, headers=headers, tablefmt="plain")

def calculate_statistics(matrices, titles, weighted=False):
    """
    Calculate statistics for each matrix and compare them.
    """
    stats = []

    for matrix, title in zip(matrices, titles):
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(matrix)

        # Get the upper triangle (excluding diagonal)
        upper_tri = np.triu(matrix, k=1).flatten()
        upper_tri = upper_tri[upper_tri != 0]  # Remove zeros

        # Calculate statistics
        stat = {
            'Prompt Type': title,
            'Mean Similarity': np.mean(upper_tri),
            'Median Similarity': np.median(upper_tri),
            'Min Similarity': np.min(upper_tri),
            'Max Similarity': np.max(upper_tri),
            'Std Deviation': np.std(upper_tri)
        }
        stats.append(stat)

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats)
    
    # Add similarity type
    similarity_type = "Weighted Jaccard" if weighted else "Jaccard"
    stats_df['Similarity Type'] = similarity_type

    return stats_df

def plot_statistics(stats_df, output_path, weighted=False):
    """
    Plot the statistics for comparison.
    """
    plt.figure(figsize=(12, 6))

    # Plot mean similarity
    plt.subplot(1, 2, 1)
    sns.barplot(x='Prompt Type', y='Mean Similarity', data=stats_df)
    similarity_type = "Weighted Jaccard" if weighted else "Jaccard"
    plt.title(f'Mean {similarity_type} Similarity by Prompt Type')
    plt.ylim(0, 1)

    # Plot standard deviation
    plt.subplot(1, 2, 2)
    sns.barplot(x='Prompt Type', y='Std Deviation', data=stats_df)
    plt.title(f'Standard Deviation of {similarity_type} Similarity by Prompt Type')

    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    print(f"Statistics plot saved to {output_path}")

    # Close the figure to free memory
    plt.close()

def compare_weighted_unweighted(unweighted_stats, weighted_stats, portfolio_size, output_dir):
    """
    Compare weighted and unweighted statistics and create comparison plots.
    
    Args:
        unweighted_stats (DataFrame): Statistics for unweighted Jaccard
        weighted_stats (DataFrame): Statistics for weighted Jaccard
        portfolio_size (int): Size of the portfolio
        output_dir (str): Directory to save the output
    """
    # Combine the stats
    combined_stats = pd.concat([unweighted_stats, weighted_stats])
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    # Plot mean similarity comparison
    plt.subplot(1, 2, 1)
    sns.barplot(x='Prompt Type', y='Mean Similarity', hue='Similarity Type', data=combined_stats)
    plt.title(f'Mean Similarity Comparison (Size {portfolio_size})')
    plt.ylim(0, 1)
    plt.legend(title='Method')
    
    # Plot standard deviation comparison
    plt.subplot(1, 2, 2)
    sns.barplot(x='Prompt Type', y='Std Deviation', hue='Similarity Type', data=combined_stats)
    plt.title(f'Standard Deviation Comparison (Size {portfolio_size})')
    plt.legend(title='Method')
    
    plt.tight_layout()
    
    # Save the figure
    comparison_path = os.path.join(output_dir, f'openai_100reps_{portfolio_size}_jaccard_comparison.png')
    plt.savefig(comparison_path)
    print(f"Comparison plot saved to {comparison_path}")
    
    # Close the figure to free memory
    plt.close()
    
    # Save the combined stats
    combined_stats.to_csv(os.path.join(output_dir, f'openai_100reps_{portfolio_size}_jaccard_comparison_stats.csv'), index=False)
    
    # Calculate the difference between weighted and unweighted
    diff_stats = []
    for prompt_type in unweighted_stats['Prompt Type'].unique():
        unweighted_mean = unweighted_stats[unweighted_stats['Prompt Type'] == prompt_type]['Mean Similarity'].values[0]
        weighted_mean = weighted_stats[weighted_stats['Prompt Type'] == prompt_type]['Mean Similarity'].values[0]
        
        diff = {
            'Prompt Type': prompt_type,
            'Unweighted Mean': unweighted_mean,
            'Weighted Mean': weighted_mean,
            'Difference': weighted_mean - unweighted_mean,
            'Percent Change': (weighted_mean - unweighted_mean) / unweighted_mean * 100
        }
        diff_stats.append(diff)
    
    diff_df = pd.DataFrame(diff_stats)
    diff_df.to_csv(os.path.join(output_dir, f'openai_100reps_{portfolio_size}_jaccard_difference_stats.csv'), index=False)
    
    return diff_df

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the data file
    print(f"Loading portfolio data from {RESULT_FILE}...")
    with open(RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    # Load market caps
    market_caps = load_market_caps()

    # Define prompt types
    prompt_types = {
        "neutral_investor": "Neutral Investor",
        "value_investor": "Value Investor",
        "growth_investor": "Growth Investor"
    }

    # Process each portfolio size
    for portfolio_size in PORTFOLIO_SIZES:
        print(f"\n\n{'='*80}")
        print(f"Processing representative portfolios of size {portfolio_size}")
        print(f"{'='*80}")

        # Calculate unweighted matrices for all prompt types
        unweighted_matrices = []
        weighted_matrices = []
        titles = []

        for prompt_key, prompt_title in prompt_types.items():
            if prompt_key in data:
                print(f"\nCalculating Jaccard Similarity Matrices for {prompt_title} with portfolio size {portfolio_size}...")
                
                # Calculate unweighted matrix
                unweighted_matrix = calculate_similarity_matrix(data[prompt_key], market_caps, portfolio_size=portfolio_size, weighted=False)
                
                # Calculate weighted matrix
                weighted_matrix = calculate_similarity_matrix(data[prompt_key], market_caps, portfolio_size=portfolio_size, weighted=True)

                # Format the matrices as tables
                unweighted_table = format_matrix_as_table(unweighted_matrix)
                weighted_table = format_matrix_as_table(weighted_matrix)

                # Save the formatted tables to text files
                unweighted_txt_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_{prompt_key}_jaccard_matrix_formatted.txt')
                weighted_txt_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_{prompt_key}_weighted_jaccard_matrix_formatted.txt')
                
                with open(unweighted_txt_path, 'w') as f:
                    f.write(unweighted_table)
                print(f"Unweighted formatted table saved to {unweighted_txt_path}")
                
                with open(weighted_txt_path, 'w') as f:
                    f.write(weighted_table)
                print(f"Weighted formatted table saved to {weighted_txt_path}")

                # Save the matrices as CSVs
                unweighted_csv_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_{prompt_key}_jaccard_matrix.csv')
                weighted_csv_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_{prompt_key}_weighted_jaccard_matrix.csv')
                
                pd.DataFrame(unweighted_matrix).to_csv(unweighted_csv_path)
                print(f"Unweighted matrix saved to {unweighted_csv_path}")
                
                pd.DataFrame(weighted_matrix).to_csv(weighted_csv_path)
                print(f"Weighted matrix saved to {weighted_csv_path}")

                unweighted_matrices.append(unweighted_matrix)
                weighted_matrices.append(weighted_matrix)
                titles.append(f"{prompt_title} (Size {portfolio_size})")
            else:
                print(f"Warning: {prompt_key} not found in the data")

        # Plot combined clustermaps for unweighted matrices
        unweighted_output_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_jaccard_matrices_comparison.png')
        plot_combined_clustermaps(unweighted_matrices, titles, unweighted_output_path, weighted=False)

        # Plot combined clustermaps for weighted matrices
        weighted_output_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_weighted_jaccard_matrices_comparison.png')
        plot_combined_clustermaps(weighted_matrices, titles, weighted_output_path, weighted=True)

        # Calculate and plot statistics for unweighted matrices
        unweighted_stats_df = calculate_statistics(unweighted_matrices, titles, weighted=False)
        print("\nUnweighted Jaccard Similarity Statistics:")
        print(unweighted_stats_df.to_string(index=False))

        # Save unweighted statistics to CSV
        unweighted_stats_csv_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_jaccard_similarity_stats.csv')
        unweighted_stats_df.to_csv(unweighted_stats_csv_path, index=False)
        print(f"Unweighted statistics saved to {unweighted_stats_csv_path}")

        # Plot unweighted statistics
        unweighted_stats_plot_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_jaccard_similarity_stats_plot.png')
        plot_statistics(unweighted_stats_df, unweighted_stats_plot_path, weighted=False)

        # Calculate and plot statistics for weighted matrices
        weighted_stats_df = calculate_statistics(weighted_matrices, titles, weighted=True)
        print("\nWeighted Jaccard Similarity Statistics:")
        print(weighted_stats_df.to_string(index=False))

        # Save weighted statistics to CSV
        weighted_stats_csv_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_weighted_jaccard_similarity_stats.csv')
        weighted_stats_df.to_csv(weighted_stats_csv_path, index=False)
        print(f"Weighted statistics saved to {weighted_stats_csv_path}")

        # Plot weighted statistics
        weighted_stats_plot_path = os.path.join(OUTPUT_DIR, f'openai_100reps_{portfolio_size}_weighted_jaccard_similarity_stats_plot.png')
        plot_statistics(weighted_stats_df, weighted_stats_plot_path, weighted=True)
        
        # Compare weighted and unweighted statistics
        diff_df = compare_weighted_unweighted(unweighted_stats_df, weighted_stats_df, portfolio_size, OUTPUT_DIR)
        print("\nDifference between Weighted and Unweighted Jaccard Similarity:")
        print(diff_df.to_string(index=False))

    print("\nAnalysis complete for all portfolio sizes!")

if __name__ == "__main__":
    main()