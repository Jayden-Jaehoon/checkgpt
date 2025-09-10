import os
import json
import time
import ast
import random
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Constants
PROMPTS_DICT_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Default/prompts_dict.json'
PROMPTS_REPETITION_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Prompts/prompts_repetition.json'
CLAUDE_REPETITION_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Prompts/prompts_repetition_claude.json'
GEMINI_REPETITION_FILE = '/Users/jaehoon/Alphatross/70_Research/checkgpt/results/Prompts/prompts_repetition_gemini.json'
USER_PROMPT = "Recommend more than 30 stocks from the S&P 500 as of September 30, 2023, to construct a portfolio that outperforms the index. Assume today's date is 2023-09-30."
MAX_PORTFOLIO_SIZE = 50  # Maximum number of stocks in a portfolio
OPENAI_ADDITIONAL_PORTFOLIOS = 50  # Additional portfolios for OpenAI
CLAUDE_PORTFOLIOS = 100  # Total portfolios for Claude
GEMINI_PORTFOLIOS = 100  # Total portfolios for Gemini

# Configure API clients
# OpenAI
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic (Claude)
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# Google (Gemini)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

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

def query_gemini(system_prompt, user_prompt, max_retries=5):
    """
    Query Google Gemini with a system prompt and user prompt, returning a list of stock tickers.
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
    
    # Construct the prompt for Gemini
    prompt = f"{system_prompt}\n\n{format_instructions}\n\n{user_prompt}"
    
    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            # Call the API
            response = gemini_model.generate_content(prompt)
            
            # Parse response into a list of tickers
            ticker_str = response.text.strip()
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
                print(f"Failed to parse Gemini response: {ticker_str}")
                print(f"Parse error: {parse_error}")
            
            # If we get here, either parsing failed or we got an empty list
            print(f"Attempt {attempt+1}/{max_retries}: Failed to get valid portfolio from Gemini. Retrying...")
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f} seconds before next attempt...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries}: Error querying Gemini: {e}")
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f} seconds before next attempt...")
            time.sleep(wait_time)
    
    # If all retries fail, return an empty list
    print("All Gemini retry attempts failed. Returning empty list.")
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
        
        # Process each key in the data
        keys = list(portfolios.keys())
        for idx, key in enumerate(keys):
            f.write(f'  "{key}": [\n')
            
            # Process each list in the current key
            for j, item in enumerate(portfolios[key]):
                # Write each list on a single line
                if isinstance(item, list):
                    f.write('    ["' + '", "'.join(item) + '"]')
                else:
                    # For non-list items, use standard JSON formatting
                    json_str = json.dumps(item, indent=4)
                    # Adjust indentation
                    json_str = '    ' + json_str.replace('\n', '\n    ')
                    f.write(json_str)
                    
                if j < len(portfolios[key]) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            
            f.write('  ]')
            if idx < len(keys) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        
        # End with closing brace
        f.write('}\n')
    
    print(f"Saved portfolios to '{file_path}'")

def generate_portfolios():
    """
    Generate additional portfolios for each investor type using OpenAI and Claude models.
    OpenAI: Generate portfolios until reaching a total of 100
    Claude: Generate portfolios until reaching a total of 100
    
    Results are saved to separate files for each model:
    - OpenAI: prompts_repetition.json
    - Claude: prompts_repetition_claude.json
    """
    # Load system prompts
    print(f"Loading system prompts from '{PROMPTS_DICT_FILE}'...")
    with open(PROMPTS_DICT_FILE, 'r') as f:
        prompts_dict = json.load(f)
    
    # Load existing portfolios for OpenAI
    print(f"Loading existing portfolios from '{PROMPTS_REPETITION_FILE}'...")
    with open(PROMPTS_REPETITION_FILE, 'r') as f:
        openai_portfolios = json.load(f)
    
    # Load or initialize portfolios for Claude
    claude_portfolios = {}
    if os.path.exists(CLAUDE_REPETITION_FILE):
        print(f"Loading existing Claude portfolios from '{CLAUDE_REPETITION_FILE}'...")
        with open(CLAUDE_REPETITION_FILE, 'r') as f:
            claude_portfolios = json.load(f)
    else:
        print(f"Initializing new Claude portfolios...")
        # Initialize with the same investor types as OpenAI
        for investor_type in openai_portfolios:
            claude_portfolios[investor_type] = []
    
    # Load or initialize portfolios for Gemini
    gemini_portfolios = {}
    if os.path.exists(GEMINI_REPETITION_FILE):
        print(f"Loading existing Gemini portfolios from '{GEMINI_REPETITION_FILE}'...")
        with open(GEMINI_REPETITION_FILE, 'r') as f:
            gemini_portfolios = json.load(f)
    else:
        print(f"Initializing new Gemini portfolios...")
        # Initialize with the same investor types as OpenAI
        for investor_type in openai_portfolios:
            gemini_portfolios[investor_type] = []
    
    # Generate additional portfolios for each investor type
    for investor_type, system_prompt in prompts_dict.items():
        print(f"\nProcessing investor type: {investor_type}")
        
        # Ensure investor type exists in all portfolio dictionaries
        for portfolios in [openai_portfolios, claude_portfolios, gemini_portfolios]:
            if investor_type not in portfolios:
                print(f"Adding missing investor type '{investor_type}' to portfolios")
                portfolios[investor_type] = []
        
        # Calculate how many portfolios to generate with each model
        openai_count = max(0, min(OPENAI_ADDITIONAL_PORTFOLIOS, 100 - len(openai_portfolios[investor_type])))
        
        # Calculate Claude count - target 100 portfolios total
        claude_count = max(0, min(CLAUDE_PORTFOLIOS, 100 - len(claude_portfolios[investor_type])))
        
        print(f"Generation plan:")
        print(f"  OpenAI: {openai_count} portfolios (current: {len(openai_portfolios[investor_type])})")
        print(f"  Claude: {claude_count} portfolios (current: {len(claude_portfolios[investor_type])})")
        print(f"  Total: {openai_count + claude_count} new portfolios")
        
        # Define models to use - only OpenAI and Claude
        models_to_use = [
            ("OpenAI", query_openai, openai_count, openai_portfolios, PROMPTS_REPETITION_FILE),
            ("Claude", query_claude, claude_count, claude_portfolios, CLAUDE_REPETITION_FILE)
        ]
        
        # Generate portfolios using each model
        for model_name, query_func, count, portfolios_dict, output_file in models_to_use:
            if count <= 0:
                print(f"Skipping {model_name} (no portfolios to generate)")
                continue
                
            print(f"Generating {count} portfolios using {model_name}...")
            for i in range(count):
                print(f"  Portfolio {i+1}/{count}...")
                
                # Query the model with retries built into the query functions
                portfolio = query_func(system_prompt, USER_PROMPT)
                
                # Add the portfolio to the list if it's not empty
                if portfolio:
                    # Check if portfolio is too large and trim if necessary
                    if len(portfolio) > MAX_PORTFOLIO_SIZE:
                        print(f"  Trimming portfolio from {len(portfolio)} to {MAX_PORTFOLIO_SIZE} stocks")
                        portfolio = portfolio[:MAX_PORTFOLIO_SIZE]
                    
                    portfolios_dict[investor_type].append(portfolio)
                    print(f"  Added portfolio with {len(portfolio)} stocks")
                else:
                    print(f"  Failed to generate portfolio after multiple retries. Skipping.")
                
                # Save intermediate results
                save_portfolios_to_file(portfolios_dict, output_file)
                
                # Sleep to avoid rate limits (random delay between 1-2 seconds)
                delay = 1 + random.uniform(0, 1)
                print(f"  Waiting {delay:.2f} seconds before next request...")
                time.sleep(delay)
        
        # Print final counts
        print(f"\nFinal portfolio counts for {investor_type}:")
        print(f"  OpenAI: {len(openai_portfolios[investor_type])}")
        print(f"  Claude: {len(claude_portfolios[investor_type])}")
    
    print("\nPortfolio generation complete!")

if __name__ == "__main__":
    generate_portfolios()