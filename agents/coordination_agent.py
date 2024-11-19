from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.components.tools import FunctionTool
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
    description="Generate a report based on the search and stock analysis results",
)

termination = TextMentionTermination("TERMINATE")
team = RoundRobinGroupChat([search_agent, fundamental_analysis_agent, stock_analysis_agent, report_agent], termination_condition=termination)



