import os
import sys
import pandas as pd

# Add the current directory to the path
sys.path.append('.')

# Import the module using importlib to handle numeric filename
import importlib.util
spec = importlib.util.spec_from_file_location("portfolio_analysis", "2_analyze_portfolio_performance_fdr.py")
portfolio_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portfolio_analysis)

def test_delisted_ticker_fix():
    """Test the fix for delisted ticker exclusion logic"""
    # Create a test list with some delisted tickers
    test_tickers = ["AAPL", "MSFT", "ATVI", "TWTR", "GOOGL"]
    
    # Call the function that uses the fixed code
    result = portfolio_analysis.get_portfolio_returns(
        test_tickers, 
        "2023-01-01", 
        "2023-12-31", 
        "equal"
    )
    
    print("Test completed for delisted ticker fix")
    return result

def test_price_data_logic_fix():
    """Test the fix for price data loading and download logic"""
    # Create a test list with some tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]
    
    # Call the function that uses the fixed code
    result = portfolio_analysis.get_portfolio_returns(
        test_tickers, 
        "2023-01-01", 
        "2023-12-31", 
        "equal"
    )
    
    print("Test completed for price data logic fix")
    return result

if __name__ == "__main__":
    print("Testing delisted ticker fix...")
    test_delisted_ticker_fix()
    
    print("\nTesting price data logic fix...")
    test_price_data_logic_fix()
    
    print("\nAll tests completed")