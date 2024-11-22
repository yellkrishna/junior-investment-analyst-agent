from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import TextMentionTermination
from autogen_ext.models import OpenAIChatCompletionClient
from config import OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID
from agents.fundamental_analysis.fundamental_analysis_agent import fundamental_analysis_agent
from agents.google_search.google_search import search_agent
from agents.technical_analysis.technical_analysis_agent import stock_analysis_agent

#!pip install yfinance matplotlib pytz numpy pandas python-dotenv requests bs4
# Create agents
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

report_agent = AssistantAgent(
    name="Report_Agent",
    model_client=model_client,
    description="As the best FINRA-approved financial analyst, the agent responsible for compiling and generating the final comprehensive financial report in Markdown format. "
        "Aggregates fundamental analysis data (financial ratios and qualitative insights), technical analysis results, "
        "SWOT analysis, and compares technical indicators with benchmarks, and incorporates information from Google searches "
        "to produce a cohesive and readable report. Be comprehensive and based on the indicators and fundamentals of the company.",
)

termination = TextMentionTermination("TERMINATE")

