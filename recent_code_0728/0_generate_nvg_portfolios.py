import os
import json
import time
import ast
import random
from dotenv import load_dotenv
import openai
import anthropic

# Load environment variables
load_dotenv()

# Constants
PROMPTS_DICT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Default/prompts_dict.json'
REPHRASE_PROMPTS_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Default/rephrase_prompts.json'
OPENAI_RESULT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG.json'
CLAUDE_RESULT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Rephrase/Rephrase_Repetition_Result_NVG_claude.json'
MAX_PORTFOLIO_SIZE = 50  # Maximum number of stocks in a portfolio
OPENAI_ADDITIONAL_PORTFOLIOS = 50  # Additional portfolios for OpenAI (to reach 100 total)
CLAUDE_PORTFOLIOS = 100  # Total portfolios for Claude
INVESTOR_TYPES = ["neutral_investor", "value_investor", "growth_investor"]  # Investor types to process

# Configure API clients
# OpenAI
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic (Claude)
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def query_openai(system_prompt, user_prompt, max_retries=5):
    """
    Query OpenAI GPT-4 with a system prompt and user prompt, returning a list of stock tickers.
    Includes retry logic with exponential backoff for robustness.
    
    Args:
        system_prompt (str): System role and instructions
        user_prompt (str): User question
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        list: List of stock tickers
    """
    # Format instructions
    format_instructions = (
        "Follow user instructions carefully. "
        "Return output ONLY in the requested format. "
        "NO extra words, NO explanations. "
        "If the user asks for a Python list of tickers, return ONLY a Python list of tickers, nothing else."
        "Note: Please follow the format strictly.\n"
        "1. The output must be a single-line Python list.\n"
        "2. Example: ['AAPL', 'TSLA', 'MSFT']\n"
        "3. All tickers must be uppercase and enclosed in single quotes.\n"
        "4. Return only the list. No extra whitespace, sentences, or explanations.\n"
        "5. No indices, numbers, bullet points, or additional text.\n"
    )
    
    # Construct messages with separate system and user prompts
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n{format_instructions}"},
        {"role": "user", "content": user_prompt}
    ]
    
    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            # Call the API
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.2
            )
            
            # Parse response into a list of tickers
            ticker_str = response.choices[0].message.content.strip()
            try:
                tickers = ast.literal_eval(ticker_str)
                if isinstance(tickers, list):
                    # Clean and standardize tickers
                    tickers = [t.strip().upper() for t in tickers if t.strip()]
                    
                    # Check if we got a non-empty list
                    if tickers:
                        # Trim if too large
                        if len(tickers) > MAX_PORTFOLIO_SIZE:
                            print(f"Trimming portfolio from {len(tickers)} to {MAX_PORTFOLIO_SIZE} stocks")
                            tickers = tickers[:MAX_PORTFOLIO_SIZE]
                        return tickers
            except Exception as parse_error:
                print(f"Failed to parse OpenAI response: {ticker_str}")
                print(f"Parse error: {parse_error}")
            
            # If we get here, either parsing failed or we got an empty list
            print(f"Attempt {attempt+1}/{max_retries}: Failed to get valid portfolio. Retrying...")
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f} seconds before next attempt...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries}: Error querying OpenAI: {e}")
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f} seconds before next attempt...")
            time.sleep(wait_time)
    
    # If all retries fail, return an empty list
    print("All retry attempts failed. Returning empty list.")
    return []

def query_claude(system_prompt, user_prompt, max_retries=5):
    """
    Query Anthropic Claude with a system prompt and user prompt, returning a list of stock tickers.
    Includes retry logic with exponential backoff for robustness.
    
    Args:
        system_prompt (str): System role and instructions
        user_prompt (str): User question
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        list: List of stock tickers
    """
    # Format instructions
    format_instructions = (
        "Follow user instructions carefully. "
        "Return output ONLY in the requested format. "
        "NO extra words, NO explanations. "
        "If the user asks for a Python list of tickers, return ONLY a Python list of tickers, nothing else."
        "Note: Please follow the format strictly.\n"
        "1. The output must be a single-line Python list.\n"
        "2. Example: ['AAPL', 'TSLA', 'MSFT']\n"
        "3. All tickers must be uppercase and enclosed in single quotes.\n"
        "4. Return only the list. No extra whitespace, sentences, or explanations.\n"
        "5. No indices, numbers, bullet points, or additional text.\n"
    )
    
    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            # Call the API
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.2,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"{user_prompt}\n\n{format_instructions}"}
                ]
            )
            
            # Parse response into a list of tickers
            ticker_str = response.content[0].text.strip()
            try:
                tickers = ast.literal_eval(ticker_str)
                if isinstance(tickers, list):
                    # Clean and standardize tickers
                    tickers = [t.strip().upper() for t in tickers if t.strip()]
                    
                    # Check if we got a non-empty list
                    if tickers:
                        # Trim if too large
                        if len(tickers) > MAX_PORTFOLIO_SIZE:
                            print(f"Trimming portfolio from {len(tickers)} to {MAX_PORTFOLIO_SIZE} stocks")
                            tickers = tickers[:MAX_PORTFOLIO_SIZE]
                        return tickers
            except Exception as parse_error:
                print(f"Failed to parse Claude response: {ticker_str}")
                print(f"Parse error: {parse_error}")
            
            # If we get here, either parsing failed or we got an empty list
            print(f"Attempt {attempt+1}/{max_retries}: Failed to get valid portfolio from Claude. Retrying...")
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f} seconds before next attempt...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries}: Error querying Claude: {e}")
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f} seconds before next attempt...")
            time.sleep(wait_time)
    
    # If all retries fail, return an empty list
    print("All Claude retry attempts failed. Returning empty list.")
    return []

def save_portfolios_to_file(portfolios, file_path):
    """
    Save portfolios to a file with custom formatting.
    
    Args:
        portfolios (dict): Dictionary of portfolios to save
        file_path (str): Path to the file to save to
    """
    with open(file_path, 'w') as f:
        # Start with opening brace
        f.write('{\n')
        
        # Process each investor type in the data
        investor_types = list(portfolios.keys())
        for idx, investor_type in enumerate(investor_types):
            f.write(f'  "{investor_type}": {{\n')
            
            # Process each rephrase prompt for the current investor type
            rephrase_prompts = list(portfolios[investor_type].keys())
            for j, rephrase_prompt in enumerate(rephrase_prompts):
                f.write(f'    "{rephrase_prompt}": [\n')
                
                # Process each portfolio for the current rephrase prompt
                portfolios_list = portfolios[investor_type][rephrase_prompt]
                for k, portfolio in enumerate(portfolios_list):
                    # Write each portfolio on a single line
                    f.write('      ["' + '", "'.join(portfolio) + '"]')
                    
                    if k < len(portfolios_list) - 1:
                        f.write(',\n')
                    else:
                        f.write('\n')
                
                f.write('    ]')
                if j < len(rephrase_prompts) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            
            f.write('  }')
            if idx < len(investor_types) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        
        # End with closing brace
        f.write('}\n')
    
    print(f"Saved portfolios to '{file_path}'")

def generate_portfolios():
    """
    Generate additional portfolios for each investor type using OpenAI and Claude models.
    OpenAI: Add 50 more portfolios to reach a total of 100
    Claude: Generate 100 new portfolios
    
    Results are saved to separate files:
    - OpenAI: Rephrase_Repetition_Result_NVG.json (updated)
    - Claude: Rephrase_Repetition_Result_NVG_claude.json (new)
    """
    # Load system prompts
    print(f"Loading system prompts from '{PROMPTS_DICT_FILE}'...")
    with open(PROMPTS_DICT_FILE, 'r') as f:
        prompts_dict = json.load(f)
    
    # Load rephrased prompts
    print(f"Loading rephrased prompts from '{REPHRASE_PROMPTS_FILE}'...")
    with open(REPHRASE_PROMPTS_FILE, 'r') as f:
        rephrase_prompts = json.load(f)
    
    # Load existing OpenAI portfolios
    print(f"Loading existing OpenAI portfolios from '{OPENAI_RESULT_FILE}'...")
    with open(OPENAI_RESULT_FILE, 'r') as f:
        openai_portfolios = json.load(f)
    
    # Load or initialize Claude portfolios
    claude_portfolios = {}
    if os.path.exists(CLAUDE_RESULT_FILE):
        print(f"Loading existing Claude portfolios from '{CLAUDE_RESULT_FILE}'...")
        with open(CLAUDE_RESULT_FILE, 'r') as f:
            claude_portfolios = json.load(f)
    else:
        print(f"Initializing new Claude portfolios...")
        # Initialize with the same investor types as OpenAI
        for investor_type in INVESTOR_TYPES:
            claude_portfolios[investor_type] = {}
            for i, _ in enumerate(rephrase_prompts):
                rephrase_key = f"rephrase_{i+1}"
                claude_portfolios[investor_type][rephrase_key] = []
    
    # Process each investor type
    for investor_type in INVESTOR_TYPES:
        print(f"\nProcessing investor type: {investor_type}")
        system_prompt = prompts_dict.get(investor_type, "")
        
        # Ensure investor type exists in both portfolio dictionaries
        if investor_type not in openai_portfolios:
            print(f"Adding missing investor type '{investor_type}' to OpenAI portfolios")
            openai_portfolios[investor_type] = {}
        
        if investor_type not in claude_portfolios:
            print(f"Adding missing investor type '{investor_type}' to Claude portfolios")
            claude_portfolios[investor_type] = {}
        
        # Process each rephrased prompt
        for i, user_prompt in enumerate(rephrase_prompts):
            rephrase_key = f"rephrase_{i+1}"
            print(f"\nProcessing rephrased prompt: {rephrase_key}")
            
            # Ensure rephrase key exists in both portfolio dictionaries
            if rephrase_key not in openai_portfolios[investor_type]:
                print(f"Adding missing rephrase key '{rephrase_key}' to OpenAI portfolios")
                openai_portfolios[investor_type][rephrase_key] = []
            
            if rephrase_key not in claude_portfolios[investor_type]:
                print(f"Adding missing rephrase key '{rephrase_key}' to Claude portfolios")
                claude_portfolios[investor_type][rephrase_key] = []
            
            # Calculate how many portfolios to generate with each model
            current_openai_count = len(openai_portfolios[investor_type][rephrase_key])
            openai_count = max(0, min(OPENAI_ADDITIONAL_PORTFOLIOS, 100 - current_openai_count))
            
            current_claude_count = len(claude_portfolios[investor_type][rephrase_key])
            claude_count = max(0, min(CLAUDE_PORTFOLIOS, 100 - current_claude_count))
            
            print(f"Generation plan:")
            print(f"  OpenAI: {openai_count} portfolios (current: {current_openai_count})")
            print(f"  Claude: {claude_count} portfolios (current: {current_claude_count})")
            
            # Generate OpenAI portfolios
            if openai_count > 0:
                print(f"Generating {openai_count} portfolios using OpenAI...")
                for j in range(openai_count):
                    print(f"  Portfolio {j+1}/{openai_count}...")
                    
                    # Query OpenAI with retries
                    portfolio = query_openai(system_prompt, user_prompt)
                    
                    # Add the portfolio to the list if it's not empty
                    if portfolio:
                        openai_portfolios[investor_type][rephrase_key].append(portfolio)
                        print(f"  Added portfolio with {len(portfolio)} stocks")
                    else:
                        print(f"  Failed to generate portfolio after multiple retries. Skipping.")
                    
                    # Save intermediate results
                    save_portfolios_to_file(openai_portfolios, OPENAI_RESULT_FILE)
                    
                    # Sleep to avoid rate limits
                    delay = 1 + random.uniform(0, 1)
                    print(f"  Waiting {delay:.2f} seconds before next request...")
                    time.sleep(delay)
            
            # Generate Claude portfolios
            if claude_count > 0:
                print(f"Generating {claude_count} portfolios using Claude...")
                for j in range(claude_count):
                    print(f"  Portfolio {j+1}/{claude_count}...")
                    
                    # Query Claude with retries
                    portfolio = query_claude(system_prompt, user_prompt)
                    
                    # Add the portfolio to the list if it's not empty
                    if portfolio:
                        claude_portfolios[investor_type][rephrase_key].append(portfolio)
                        print(f"  Added portfolio with {len(portfolio)} stocks")
                    else:
                        print(f"  Failed to generate portfolio after multiple retries. Skipping.")
                    
                    # Save intermediate results
                    save_portfolios_to_file(claude_portfolios, CLAUDE_RESULT_FILE)
                    
                    # Sleep to avoid rate limits
                    delay = 1 + random.uniform(0, 1)
                    print(f"  Waiting {delay:.2f} seconds before next request...")
                    time.sleep(delay)
            
            # Print final counts for this rephrased prompt
            print(f"\nFinal portfolio counts for {investor_type}, {rephrase_key}:")
            print(f"  OpenAI: {len(openai_portfolios[investor_type][rephrase_key])}")
            print(f"  Claude: {len(claude_portfolios[investor_type][rephrase_key])}")
    
    # Print final overall counts
    print("\nFinal overall portfolio counts:")
    for investor_type in INVESTOR_TYPES:
        openai_total = sum(len(openai_portfolios[investor_type][rephrase_key]) for rephrase_key in openai_portfolios[investor_type])
        claude_total = sum(len(claude_portfolios[investor_type][rephrase_key]) for rephrase_key in claude_portfolios[investor_type])
        print(f"  {investor_type}:")
        print(f"    OpenAI: {openai_total} portfolios")
        print(f"    Claude: {claude_total} portfolios")
    
    print("\nPortfolio generation complete!")

def verify_results():
    """
    Verify that the results files have the expected number of portfolios.
    """
    print("\nVerifying results...")
    
    # Check OpenAI results
    print(f"Checking OpenAI results in '{OPENAI_RESULT_FILE}'...")
    with open(OPENAI_RESULT_FILE, 'r') as f:
        openai_portfolios = json.load(f)
    
    # Check Claude results
    if os.path.exists(CLAUDE_RESULT_FILE):
        print(f"Checking Claude results in '{CLAUDE_RESULT_FILE}'...")
        with open(CLAUDE_RESULT_FILE, 'r') as f:
            claude_portfolios = json.load(f)
    else:
        print(f"Claude results file '{CLAUDE_RESULT_FILE}' does not exist.")
        claude_portfolios = {}
    
    # Verify counts for each investor type and rephrased prompt
    for investor_type in INVESTOR_TYPES:
        print(f"\nInvestor type: {investor_type}")
        
        if investor_type in openai_portfolios:
            for rephrase_key in openai_portfolios[investor_type]:
                openai_count = len(openai_portfolios[investor_type][rephrase_key])
                print(f"  OpenAI {rephrase_key}: {openai_count} portfolios {'✓' if openai_count >= 100 else '✗'}")
        else:
            print(f"  OpenAI: Investor type not found ✗")
        
        if investor_type in claude_portfolios:
            for rephrase_key in claude_portfolios[investor_type]:
                claude_count = len(claude_portfolios[investor_type][rephrase_key])
                print(f"  Claude {rephrase_key}: {claude_count} portfolios {'✓' if claude_count >= 100 else '✗'}")
        else:
            print(f"  Claude: Investor type not found ✗")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    generate_portfolios()
    verify_results()