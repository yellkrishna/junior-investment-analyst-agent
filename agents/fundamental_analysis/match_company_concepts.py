import requests
import pandas as pd
import json
import sys
from bs4 import BeautifulSoup, NavigableString
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up request headers with your email for identification
headers = {'User-Agent': "sivayellepeddi@gmail.com"}

def format_cik(cik):
    return cik.zfill(10)

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
        logger.error(f"HTTP error occurred while fetching CIK: {http_err}")
        return None
    except Exception as err:
        logger.error(f"An error occurred while fetching CIK: {err}")
        return None

    try:
        company_data = pd.DataFrame.from_dict(response.json(), orient='index')
        company_data['cik_str'] = company_data['cik_str'].astype(str).str.zfill(10)
    except ValueError:
        logger.error("Error parsing JSON response for company tickers.")
        return None

    matched_company = company_data[company_data['ticker'].str.upper() == ticker_symbol.upper()]

    if not matched_company.empty:
        cik = matched_company.iloc[0]['cik_str']
        ticker = matched_company.iloc[0]['ticker']
        company_name = matched_company.iloc[0]['title']
        logger.info(f"Found CIK: {cik} for {company_name} ({ticker})")
        return cik
    else:
        logger.warning(f"Ticker '{ticker_symbol}' not found in the SEC database.")
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
        logger.info(f"Successfully fetched company facts for CIK: {cik}")
        return facts_json
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while fetching company facts: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred while fetching company facts: {err}")
    return None

def extract_us_gaap_concepts(facts_json):
    """
    Extracts all US-GAAP concepts from the company facts JSON.

    Parameters:
        facts_json (dict): The JSON data containing company facts.

    Returns:
        list: A list of concept names.
    """
    concepts = []

    # The US-GAAP taxonomy is usually under 'us-gaap'
    us_gaap_data = facts_json.get('facts', {}).get('us-gaap', {})

    if not us_gaap_data:
        logger.warning("No US-GAAP data found in the company facts.")
        return concepts

    for concept in us_gaap_data.keys():
        concepts.append(concept)
    return concepts

# New functions to fetch and parse filings

# Retrieves recent SEC filings for a company using its CIK.
def get_filings(cik, filing_type='10-K', count=5):
    formatted_cik = format_cik(cik)
    submissions_url = f'https://data.sec.gov/submissions/CIK{formatted_cik}.json'
    try:
        response = requests.get(submissions_url, headers=headers)
        response.raise_for_status()
        submissions = response.json()
        filings = submissions.get('filings', {}).get('recent', {})
        if not filings:
            logger.error("No filings found in the API response.")
            return pd.DataFrame()
        df_filings = pd.DataFrame(filings)
        if 'accessionNumber' not in df_filings.columns or 'form' not in df_filings.columns:
            logger.error("Required keys ('accessionNumber', 'form') missing in filings data.")
            return pd.DataFrame()
        df_filings['accessionNumber'] = df_filings['accessionNumber'].str.replace('-', '')
        filtered_filings = df_filings[df_filings['form'] == filing_type].head(count)
        logger.info(f"Found {len(filtered_filings)} filings of type {filing_type} for CIK {cik}.")
        return filtered_filings
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while fetching filings: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred while fetching filings: {err}")
    return pd.DataFrame()

# Downloads the main document of a filing based on its accession number.
def download_filing(cik, accession_number):
    formatted_cik = format_cik(cik)
    index_url = f'https://www.sec.gov/Archives/edgar/data/{int(formatted_cik)}/{accession_number}/index.json'
    try:
        response = requests.get(index_url, headers=headers)
        response.raise_for_status()
        index_data = response.json()
        # Locate the filing's main document
        for file in index_data['directory']['item']:
            if file['name'].endswith('.htm') or file['name'].endswith('.txt'):
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(formatted_cik)}/{accession_number}/{file['name']}"
                filing_response = requests.get(filing_url, headers=headers)
                filing_response.raise_for_status()
                logger.info(f"Successfully downloaded filing from {filing_url}")
                return filing_response.text
        logger.error("Main filing document not found.")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while downloading filing: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred while downloading filing: {err}")
    return None


import re


# Parses specific sections from an SEC filing using BeautifulSoup.
def parse_filing_content(filing_content, sections=['Item 1A. Risk Factors', "Management's Discussion and Analysis"]):
    soup = BeautifulSoup(filing_content, 'html.parser')
    extracted_data = {}

    for section in sections:
        # Create a regex pattern to match the section header
        pattern = re.compile(section.replace('.', r'\.').replace(' ', r'\s+'), re.IGNORECASE)
        headers = soup.find_all(text=pattern)
        if headers:
            for header in headers:
                content = []
                sibling = header.find_parent()
                while True:
                    sibling = sibling.next_sibling
                    if sibling is None:
                        break
                    if isinstance(sibling, NavigableString):
                        continue
                    if sibling.find(text=pattern):
                        break
                    content.append(sibling.get_text(separator=' ', strip=True))
                extracted_text = ' '.join(content)
                if extracted_text:
                    extracted_data[section] = extracted_text
                    logger.info(f"Extracted '{section}' section.")
                    break  # Stop after finding the first matching section
        else:
            logger.warning(f"Could not find '{section}' section in the filing.")
    return extracted_data

