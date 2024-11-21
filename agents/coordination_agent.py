from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.components.tools import FunctionTool
from autogen_ext.models import OpenAIChatCompletionClient
from config import OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID
from agents.fundamental_analysis.fundamental_analysis_agent import fundamental_analysis_agent
from agents.google_search.google_search import search_agent
from agents.technical_analysis.technical_analysis_agent import stock_analysis_agent
from agents.swot.swot_analysis import perform_swot_analysis

#!pip install yfinance matplotlib pytz numpy pandas python-dotenv requests bs4
# Create agents
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

report_agent = AssistantAgent(
    name="Report_Agent",
    model_client=model_client,
    description="As the best FINRA-approved financial analyst, the agent responsible for compiling and generating the final comprehensive financial report in Markdown format. "
        "Aggregates fundamental analysis data (financial ratios and qualitative insights), technical analysis results, "
        "SWOT analysis, and compares technical indicators with benchmarks, and incorporates information from Google searches "
        "to produce a cohesive and readable report.",
)

# New SWOT_Analysis_Agent
swot_analysis_tool = FunctionTool(
    perform_swot_analysis,
    description=(
        "Function to perform SWOT analysis on a company using financial data (as a list of dictionaries), qualitative insights from filings, "
        "and information from internet searches. Returns a dictionary containing the SWOT analysis."
    )
)

swot_analysis_agent = AssistantAgent(
    name="SWOT_Analysis_Agent",
    model_client=model_client,
    tools=[swot_analysis_tool],
    description=(
        "Agent specialized in conducting SWOT analysis of a company. "
        "Uses financial ratios, qualitative data from filings, and information from internet searches "
        "to analyze the company's strengths, weaknesses, opportunities, and threats."
        "Ensure that you collect search results from the Search_Agent before performing the analysis."
    )
)

termination = TextMentionTermination("TERMINATE")

selector_prompt = (
    "You are coordinating a team of agents with the following roles:{roles}.\n"
    "- **Google_Search_Agent**: Conducts online searches to gather relevant financial documents and data.\n"
    "- **Financial_Analysis_Agent**: Analyzes financial ratios and metrics based on the data provided.\n"
    "- **Stock_Analysis_Agent**: Performs stock performance and technical analysis.\n"
    "Based on the conversation history, select the next agent from the following participants to contribute:\n"
    "{participants}\n\n"
    "Consider the context and ensure that the Report_Agent is selected when the conversation requires compiling the final report.\n\n"
    "Conversation History:\n"
    "{history}\n\n"
    "Select the most appropriate agent to continue the conversation."
)

# Create the SelectorGroupChat
team = SelectorGroupChat(
    participants=[report_agent, search_agent, fundamental_analysis_agent, stock_analysis_agent],
    model_client=model_client,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False,
    termination_condition=termination
)

"""team = RoundRobinGroupChat(
    [search_agent, fundamental_analysis_agent, stock_analysis_agent, report_agent],
    termination_condition=termination
)"""



