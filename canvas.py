from agents.coordination_agent import report_agent
from autogen_core.components.models import OpenAIChatCompletionClient
from main import team
import pypandoc
import asyncio
from autogen_agentchat.messages import TextMessage
from markdown_pdf import MarkdownPdf, Section
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
    description="Agent responsible for compiling and generating the final financial report in Markdown format. "
        "Aggregates fundamental analysis data (financial ratios and qualitative insights), technical analysis results, "
        "SWOT analysis, and compares technical indicators with benchmarks, and incorporates information from Google searches "
        "to produce a cohesive and readable report. ONLY this agent will generate the final report.",
)

termination = TextMentionTermination("TERMINATE")

selector_prompt = (
    "You are coordinating a team of agents with the following roles:{roles}.\n"
    "- **Google_Search_Agent**: Conducts online searches to gather relevant financial documents and data.\n"
    "- **Financial_Analysis_Agent**: Analyzes financial ratios and metrics based on the data provided.\n"
    "- **Stock_Analysis_Agent**: Performs stock performance and technical analysis.\n"
    "- **Report_Agent**: Compiles and formats the final financial report using data and analyses from the other agents.\n\n"
    "Based on the conversation history, select the next agent from the following participants to contribute:\n"
    "{participants}\n\n"
    "Consider the context and ensure that the Report_Agent is selected when the conversation requires compiling the final report.\n\n"
    "Conversation History:\n"
    "{history}\n\n"
    "Select the most appropriate agent to continue the conversation."
)



async def main():
    def markdown_to_pdf(markdown_text, output_file):
        # Initialize the MarkdownPdf object
        pdf = MarkdownPdf()

        # Add the Markdown content as a section
        pdf.add_section(Section(markdown_text))

        # Save the PDF to the specified file
        pdf.save(output_file)

    # Create the SelectorGroupChat
    team = SelectorGroupChat(
        participants=[report_agent, search_agent, fundamental_analysis_agent, stock_analysis_agent],
        model_client=model_client,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,
        termination_condition=termination
    )
    # Run the team conversation
    final_result = await team.run(task="Write a financial report on American Airlines")

    # Collect the conversation history
    conversation_history = final_result.messages

    # Format the conversation history into a string
    conversation_history_str = ""
    for message in conversation_history:
        conversation_history_str += f"{message.source}: {message.content}\n"

    # Prepare the prompt for the report agent
    report_prompt = (
        "Based on the following conversation history among agents:\n\n"
        "{conversation}\n\n"
        "Please compile and generate the final financial report in Markdown format. "
        "Aggregate fundamental analysis data (financial ratios and qualitative insights), technical analysis results, "
        "SWOT analysis, and compare technical indicators with benchmarks, and incorporate information from Google searches "
        "to produce a cohesive and readable report."
    )

    formatted_prompt = report_prompt.format(conversation=conversation_history_str)
    user_message = TextMessage(content=formatted_prompt, source="User")

    # Create a new conversation with the report agent
    response = await report_agent.on_messages([user_message], None)
    final_report = response.chat_message.content
    
    print("Final Report:\n", final_report)
    # Save the final report to a Markdown file
    md_filename = 'Financial_Report.md'
    with open(md_filename, 'w', encoding='utf-8') as md_file:
        md_file.write(final_report)
    print(f"The financial report has been saved as '{md_filename}'.")

    # Convert the Markdown content to a PDF
    pdf_filename = 'Financial_Report.pdf'
    markdown_to_pdf(final_report, pdf_filename)
    print(f"The financial report has been converted to '{pdf_filename}'.")

# Run the main function
asyncio.run(main())