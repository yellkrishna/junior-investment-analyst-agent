from agents.fundamental_analysis.extract_filing_details import calculate_financial_ratios
from autogen_core.components.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models import OpenAIChatCompletionClient
from agents.fundamental_analysis.template_matching import match_concepts
from agents.fundamental_analysis.match_company_concepts import (
    get_cik_for_ticker,
    fetch_company_facts,
    extract_us_gaap_concepts
)
from config import OPENAI_API_KEY
import os
import pandas as pd
from typing import Annotated, Tuple, Dict

# Define the fundamental_analyzer function
def fundamental_analyzer(
    ticker: Annotated[str, "The stock ticker symbol of the company"]
) -> Dict:
    """
    Analyze the fundamental financial metrics of a company based on its stock ticker.

    Parameters:
    ticker (str): The stock ticker symbol of the company.

    Returns:
    Tuple[pd.DataFrame, dict]: A DataFrame containing calculated financial ratios and a dictionary of qualitative data.
    """
    # Standardized template concepts required for analysis
    template_concepts = [
        'NetIncomeLoss', 'StockholdersEquity', 'Revenues',
        'CostOfGoodsAndServicesSold', 'AssetsCurrent', 'LiabilitiesCurrent',
        'InventoryNet', 'Liabilities', 'OperatingIncomeLoss', 'InterestExpense',
        'Assets', 'EarningsPerShareBasic'
    ]

    # Step 1: Get CIK for the provided ticker
    cik = get_cik_for_ticker(ticker)
    if not cik:
        raise ValueError(f"Invalid ticker: {ticker}")

    # Step 2: Fetch company facts
    facts_json = fetch_company_facts(cik)
    if not facts_json:
        raise ValueError(f"Failed to fetch company facts for CIK: {cik}")

    # Step 3: Extract US-GAAP concepts
    company_concepts = extract_us_gaap_concepts(facts_json)
    if not company_concepts:
        raise ValueError(f"No US-GAAP concepts found for CIK: {cik}")

    # Step 4: Match company concepts to template concepts
    matched_concepts = match_concepts(company_concepts, template_concepts)
    if not matched_concepts:
        raise ValueError(f"Failed to match company concepts for CIK: {cik}")

    # Step 5: Calculate financial ratios and extract qualitative data
    ratios_df, qualitative_data, fund_plots = calculate_financial_ratios(matched_concepts, cik, ticker)
    if ratios_df.empty:
        raise ValueError(f"Failed to calculate financial ratios for CIK: {cik}")

    # Convert ratios_df to a list of dictionaries
    financial_ratios_json = ratios_df.reset_index().to_dict(orient='records')

    # Combine ratios and qualitative data into a single dictionary
    analysis_result = {
        'financial_ratios': financial_ratios_json,
        'qualitative_data': qualitative_data,
        'fundamental_plots': fund_plots
    }

    return analysis_result

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

fundamental_analysis_tool = FunctionTool(
    fundamental_analyzer,
    description=(
        "Function to perform fundamental analysis on a company using its stock ticker. "
        "Returns a dictionary containing fundamental financial ratios (as a list of dictionaries) and qualitative data from filings, and plots."
    )
)

fundamental_analysis_agent = AssistantAgent(
    name="FundamentalAnalyst",
    model_client=model_client,
    tools=[fundamental_analysis_tool],
    description=(
        "Agent specialized in fundamental analysis of a company. "
        "Uses the company's stock ticker to calculate key fundamental ratios over multiple time periods, "
        "extracts qualitative data from the latest 10-K and 10-Q filings, and generates relevant financial plots."
        "Returns result as a dictionary to be used by the report agent. You will NOT generate TERMINATE condition."
    )
)