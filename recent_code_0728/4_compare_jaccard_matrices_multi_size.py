import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tabulate import tabulate
import os

def calculate_jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.
    Jaccard similarity = |intersection| / |union|
    """
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union if union > 0 else 0

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

def calculate_similarity_matrix(prompt_data, portfolio_size=30, subset_size=None):
    """
    Calculate the Jaccard similarity matrix for a given prompt's rephrases.
    
    Args:
        prompt_data (dict): Dictionary containing rephrase data
        portfolio_size (int): Size of the representative portfolio
        subset_size (int, optional): Size of subset to use. If None, use all data.
        
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
            # Get all portfolios for this rephrase
            portfolios_full = prompt_data[rephrase_key]
            
            # Use subset if specified
            portfolios = portfolios_full
            if subset_size is not None and len(portfolios_full) > subset_size:
                portfolios = portfolios_full[:subset_size]
                print(f"  Using first {subset_size} repetitions for {rephrase_key}")
            
            # Calculate representative portfolio for this rephrase
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
                matrix[i][j] = calculate_jaccard_similarity(stock_sets[i], stock_sets[j])

    return matrix

def plot_combined_clustermaps(matrices, titles, output_path):
    """
    Plot multiple clustermaps for comparison.
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
        plt.suptitle(f"Jaccard Similarity Matrix for {title}", fontsize=14, y=1.02)

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

def calculate_statistics(matrices, titles):
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

    return stats_df

def plot_statistics(stats_df, output_path):
    """
    Plot the statistics for comparison.
    """
    plt.figure(figsize=(12, 6))

    # Plot mean similarity
    plt.subplot(1, 2, 1)
    sns.barplot(x='Prompt Type', y='Mean Similarity', data=stats_df)
    plt.title('Mean Jaccard Similarity by Prompt Type')
    plt.ylim(0, 1)

    # Plot standard deviation
    plt.subplot(1, 2, 2)
    sns.barplot(x='Prompt Type', y='Std Deviation', data=stats_df)
    plt.title('Standard Deviation of Jaccard Similarity by Prompt Type')

    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    print(f"Statistics plot saved to {output_path}")

    # Close the figure to free memory
    plt.close()

def main():
    # Load the data file
    file_path = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG.json'
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Define prompt types
    prompt_types = {
        "neutral_investor": "Neutral Investor",
        "value_investor": "Value Investor",
        "growth_investor": "Growth Investor"
    }

    # Define portfolio sizes
    portfolio_sizes = [30]
    
    # Define sample sizes
    sample_sizes = [100]

    # Create output directory if it doesn't exist
    output_dir = '/Users/jaehoon/Alphatross/70_Research/checkgpt/rivision/Rephrase_Variability/openai_jaccard'
    os.makedirs(output_dir, exist_ok=True)

    # Process each sample size
    for sample_size in sample_sizes:
        sample_suffix = f"{sample_size}reps"
        subset_size = sample_size if sample_size < 100 else None
        
        print(f"\n\n{'#'*80}")
        print(f"Processing with {sample_size} repetitions")
        print(f"{'#'*80}")
        
        # Process each portfolio size
        for portfolio_size in portfolio_sizes:
            print(f"\n\n{'='*80}")
            print(f"Processing representative portfolios of size {portfolio_size} with {sample_size} repetitions")
            print(f"{'='*80}")

            # Calculate matrices for all prompt types
            matrices = []
            titles = []

            for prompt_key, prompt_title in prompt_types.items():
                if prompt_key in data:
                    print(f"\nCalculating Jaccard Similarity Matrix for {prompt_title} with portfolio size {portfolio_size}...")
                    matrix = calculate_similarity_matrix(data[prompt_key], portfolio_size=portfolio_size, subset_size=subset_size)

                    # Format the matrix as a table
                    table = format_matrix_as_table(matrix)

                    # Save the formatted table to a text file
                    txt_path = os.path.join(output_dir, f'openai_{sample_suffix}_{portfolio_size}_{prompt_key}_jaccard_matrix_formatted.txt')
                    with open(txt_path, 'w') as f:
                        f.write(table)
                    print(f"Formatted table saved to {txt_path}")

                    # Save the matrix as a CSV
                    csv_path = os.path.join(output_dir, f'openai_{sample_suffix}_{portfolio_size}_{prompt_key}_jaccard_matrix.csv')
                    pd.DataFrame(matrix).to_csv(csv_path)
                    print(f"Matrix saved to {csv_path}")

                    matrices.append(matrix)
                    titles.append(f"{prompt_title} (Size {portfolio_size})")
                else:
                    print(f"Warning: {prompt_key} not found in the data")

            # Plot combined clustermaps
            output_path = os.path.join(output_dir, f'openai_{sample_suffix}_{portfolio_size}_jaccard_matrices_comparison.png')
            plot_combined_clustermaps(matrices, titles, output_path)

            # Calculate and plot statistics
            stats_df = calculate_statistics(matrices, titles)
            print("\nJaccard Similarity Statistics:")
            print(stats_df.to_string(index=False))

            # Save statistics to CSV
            stats_csv_path = os.path.join(output_dir, f'openai_{sample_suffix}_{portfolio_size}_jaccard_similarity_stats.csv')
            stats_df.to_csv(stats_csv_path, index=False)
            print(f"Statistics saved to {stats_csv_path}")

            # Plot statistics
            stats_plot_path = os.path.join(output_dir, f'openai_{sample_suffix}_{portfolio_size}_jaccard_similarity_stats_plot.png')
            plot_statistics(stats_df, stats_plot_path)

    print("\nAnalysis complete for all portfolio sizes and sample sizes!")

if __name__ == "__main__":
    main()