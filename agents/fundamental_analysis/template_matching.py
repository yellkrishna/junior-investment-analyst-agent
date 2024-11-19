from typing import List, Dict
from autogen_core.components.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models import OpenAIChatCompletionClient
import openai
import logging
from config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def match_concepts(
    company_concepts: List[str],
    template_concepts: List[str],
    openai_api_key= OPENAI_API_KEY,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0
) -> Dict[str, str]:
    """
    Match company-specific concepts to standardized template concepts using OpenAI's language model.

    Args:
        company_concepts (List[str]): List of concepts from the company's EDGAR filings.
        template_concepts (List[str]): Standardized list of concepts required for analysis.
        openai_api_key (str): OpenAI API key.
        model (str, optional): OpenAI model to use. Defaults to "gpt-4".
        temperature (float, optional): Sampling temperature. Lower values make the output more deterministic. Defaults to 0.0.

    Returns:
        Dict[str, str]: Mapping of template concepts to the closest matching company concepts.
    """
    # Initialize OpenAI client
    openai.api_key = openai_api_key
    client = openai.OpenAI(api_key=openai_api_key)

    matched_concepts = {}

    for template in template_concepts:
        prompt = (
            f"You are a FINRA approved financial analyst with deep understanding of fundamental analysis. You an expert in financial filings and data analysis.\n\n"
            f"Given the following list of company-specific concepts extracted from EDGAR filings:\n"
            f"{', '.join(company_concepts)}\n\n"
            f"To do fundamenta analysis, identify the company concept that best matches the standardized template concept: '{template}'\n\n."
            f" This company concept will then be used to calculate various financial metrics.\n"
            f"Respond with only the closest matching company concept. Provide the exact concept name as it appears in the EDGAR filings without any extra words added."
        )

        try:
            logger.info(f"Matching template concept: '{template}'")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=50,
                n=1
            )

            matched_concept = response.choices[0].message.content.strip()

            if not matched_concept:
                logger.warning(f"No matching concept found for template '{template}'.")
                matched_concept = "No matching concept found."

            matched_concepts[template] = matched_concept
            logger.info(f"Matched '{template}' to '{matched_concept}'")

        except Exception as e:
            logger.error(f"Error matching template concept '{template}': {e}")
            matched_concepts[template] = "Error in matching."

    return matched_concepts


