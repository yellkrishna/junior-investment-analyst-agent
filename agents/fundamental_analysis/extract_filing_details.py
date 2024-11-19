import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_financial_ratios(concepts: Dict[str, str], cik: str, ticker: str) -> pd.DataFrame:
    """
    Calculates financial ratios based on provided company concepts, CIK, and ticker.

    Parameters:
    - concepts (dict): Mapping of standard financial metric names to EDGAR filing concept names.
                       Example:
                       {
                           'NetIncomeLoss': 'NetIncomeLoss',
                           'StockholdersEquity': 'StockholdersEquity',
                           'Revenues': 'SalesRevenueNet',
                           ...
                       }
    - cik (str): Central Index Key of the company.
    - ticker (str): Stock ticker symbol of the company.

    Returns:
    - pd.DataFrame: DataFrame containing calculated financial ratios.
    """

    # Set up request headers for SEC API
    headers = {'User-Agent': "sivayellepeddi@gmail.com"}

    def get_company_concept_data(cik: str, concept: str) -> pd.DataFrame:
        """
        Fetches concept data for a given CIK and concept from SEC EDGAR API.

        Parameters:
        - cik (str): Central Index Key of the company.
        - concept (str): EDGAR concept name.

        Returns:
        - pd.DataFrame or None: DataFrame containing the concept data or None if failed.
        """
        url = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json'
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                units = data.get('units')
                if not units:
                    logger.warning(f"No units found for concept '{concept}'.")
                    return None
                # Prefer 'USD' unit if available
                unit = 'USD' if 'USD' in units else list(units.keys())[0]
                records = units[unit]
                df = pd.DataFrame(records)
                df['unit'] = unit
                df['concept'] = concept
                df['end'] = pd.to_datetime(df['end'])
                df = df.sort_values(by='end').drop_duplicates(subset='end', keep='last')
                df = df.set_index('end')
                return df[['val']]
            elif response.status_code == 404:
                logger.error(f"Concept '{concept}' not found for CIK {cik}.")
                return None
            else:
                logger.error(f"Failed to fetch data for concept '{concept}'. Status Code: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Exception occurred while fetching data for concept '{concept}': {e}")
            return None

    # Initialize a dictionary to hold data for each standard concept
    data_dict = {}

    # Fetch and process data for each concept based on the provided mapping
    for standard_metric, filing_concept in concepts.items():
        df = get_company_concept_data(cik, filing_concept)
        if df is not None:
            data_dict[standard_metric] = df.rename(columns={'val': standard_metric})
            logger.info(f"Successfully fetched data for metric '{standard_metric}'.")
        else:
            logger.warning(f"No data available for standard metric '{standard_metric}' with concept '{filing_concept}'.")

    if not data_dict:
        logger.error("No financial data could be retrieved from the provided concepts.")
        return pd.DataFrame()

    # Merge all data into a single DataFrame based on the 'end' date
    merged_df = pd.concat(data_dict.values(), axis=1)
    merged_df = merged_df.sort_index()

    # Ensure all template_concepts are present, fill with np.nan if missing
    for metric in concepts.keys():
        if metric not in merged_df.columns:
            merged_df[metric] = np.nan
            logger.warning(f"Metric '{metric}' is missing and has been filled with NaN.")

    # Define a safe division function to handle division by zero or invalid operations
    def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result.replace([np.inf, -np.inf], np.nan, inplace=True)
        return result

    # Compute financial ratios
    try:
        # Profitability Ratios
        merged_df['ROE'] = safe_divide(merged_df['NetIncomeLoss'], merged_df['StockholdersEquity'])
        merged_df['NetProfitMargin'] = safe_divide(merged_df['NetIncomeLoss'], merged_df['Revenues'])
        merged_df['GrossMargin'] = safe_divide(
            merged_df['Revenues'] - merged_df['CostOfGoodsAndServicesSold'],
            merged_df['Revenues']
        )

        # Liquidity Ratios
        merged_df['CurrentRatio'] = safe_divide(merged_df['AssetsCurrent'], merged_df['LiabilitiesCurrent'])
        merged_df['QuickRatio'] = safe_divide(
            merged_df['AssetsCurrent'] - merged_df['InventoryNet'],
            merged_df['LiabilitiesCurrent']
        )

        # Leverage Ratios
        merged_df['DebtToEquityRatio'] = safe_divide(merged_df['Liabilities'], merged_df['StockholdersEquity'])
        merged_df['InterestCoverageRatio'] = safe_divide(
            merged_df['OperatingIncomeLoss'],
            merged_df['InterestExpense']
        )

        # Efficiency Ratios
        merged_df['AverageAssets'] = merged_df['Assets'].rolling(window=2, min_periods=1).mean()
        merged_df['AssetTurnover'] = safe_divide(merged_df['Revenues'], merged_df['AverageAssets'])
        merged_df['AverageInventory'] = merged_df['InventoryNet'].rolling(window=2, min_periods=1).mean()
        merged_df['InventoryTurnover'] = safe_divide(
            merged_df['CostOfGoodsAndServicesSold'],
            merged_df['AverageInventory']
        )

        # Growth Metrics
        merged_df['EPSGrowth'] = safe_divide(
            merged_df['EarningsPerShareBasic'] - merged_df['EarningsPerShareBasic'].shift(1),
            merged_df['EarningsPerShareBasic'].shift(1)
        )
        merged_df['RevenueGrowth'] = safe_divide(
            merged_df['Revenues'] - merged_df['Revenues'].shift(1),
            merged_df['Revenues'].shift(1)
        )
    except Exception as e:
        logger.exception(f"Exception occurred while calculating financial ratios: {e}")
        return pd.DataFrame()

    # Fetch current stock price for P/E Ratio
    try:
        stock = yf.Ticker(ticker)
        stock_history = stock.history(period='1d')
        if not stock_history.empty:
            current_price = stock_history['Close'].iloc[0]
            current_price_series = pd.Series(current_price, index=merged_df.index)
            merged_df['PE_Ratio'] = safe_divide(current_price_series, merged_df['EarningsPerShareBasic'])
            logger.info("Successfully calculated PE Ratio.")
        else:
            logger.warning("Stock history is empty. Cannot calculate P/E Ratio.")
            merged_df['PE_Ratio'] = np.nan
    except Exception as e:
        logger.exception(f"Failed to fetch stock price or calculate P/E Ratio: {e}")
        merged_df['PE_Ratio'] = np.nan

    # Define the list of ratios to include in the output
    ratios = [
        'ROE', 'NetProfitMargin', 'GrossMargin', 'CurrentRatio', 'QuickRatio',
        'DebtToEquityRatio', 'InterestCoverageRatio', 'AssetTurnover',
        'InventoryTurnover', 'EPSGrowth', 'RevenueGrowth', 'PE_Ratio'
    ]

    # Return the ratios DataFrame with relevant columns, dropping rows where all ratios are NaN
    final_ratios_df = merged_df[ratios].dropna(how='all')

    if final_ratios_df.empty:
        logger.warning("All calculated financial ratios are NaN.")
    else:
        logger.info("Successfully calculated financial ratios.")

    return final_ratios_df
