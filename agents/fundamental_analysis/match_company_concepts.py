import requests
import pandas as pd
import json
import sys

# Set up request headers with your email for identification
headers = {'User-Agent': "sivayellepeddi@gmail.com"}

def get_cik_for_ticker(ticker_symbol):
    """
    Fetches the CIK number for a given ticker symbol from the SEC API.

    Parameters:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').

    Returns:
        str: The 10-digit CIK number as a string, or None if not found.
    """
    try:
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching CIK: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred while fetching CIK: {err}")
        return None

    try:
        company_data = pd.DataFrame.from_dict(response.json(), orient='index')
        company_data['cik_str'] = company_data['cik_str'].astype(str).str.zfill(10)
    except ValueError:
        print("Error parsing JSON response for company tickers.")
        return None

    matched_company = company_data[company_data['ticker'].str.upper() == ticker_symbol.upper()]

    if not matched_company.empty:
        cik = matched_company.iloc[0]['cik_str']
        ticker = matched_company.iloc[0]['ticker']
        company_name = matched_company.iloc[0]['title']
        print(f"Found CIK: {cik} for {company_name} ({ticker})")
        return cik
    else:
        print(f"Ticker '{ticker_symbol}' not found in the SEC database.")
        return None

def fetch_company_facts(cik):
    """
    Fetches the company facts JSON from the SEC API for a given CIK.

    Parameters:
        cik (str): The 10-digit CIK number.

    Returns:
        dict: The JSON data containing company facts, or None if failed.
    """
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        facts_json = response.json()
        print(f"Successfully fetched company facts for CIK: {cik}")
        return facts_json
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching company facts: {http_err}")
    except Exception as err:
        print(f"An error occurred while fetching company facts: {err}")
    return None

def extract_us_gaap_concepts(facts_json):
    """
    Extracts all US-GAAP concepts from the company facts JSON.

    Parameters:
        facts_json (dict): The JSON data containing company facts.

    Returns:
        list: A list of dictionaries containing concept details.
    """
    concepts = []

    # The US-GAAP taxonomy is usually under 'us-gaap'
    us_gaap_data = facts_json.get('facts', {}).get('us-gaap', {})

    if not us_gaap_data:
        print("No US-GAAP data found in the company facts.")
        return concepts

    for concept, data in us_gaap_data.items():
        concepts.append(concept)
    return concepts


