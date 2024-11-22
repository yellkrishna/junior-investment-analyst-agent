import sys
import os
import streamlit as st
import asyncio
import nest_asyncio  # Import nest_asyncio
from pathlib import Path
from PIL import Image
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from autogen_agentchat.messages import TextMessage, ToolCallResultMessage, FunctionExecutionResult
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models import OpenAIChatCompletionClient
from autogen_agentchat.task import TextMentionTermination, MaxMessageTermination
import ast  # To safely evaluate string representations of Python literals
import re
import pandas as pd
import numpy as np
from markdown_pdf import MarkdownPdf, Section
from autogen_agentchat.teams import RoundRobinGroupChat

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set the page configuration as the first Streamlit command
st.set_page_config(
    page_title="Investment Analysis Agent",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Adjust the path to import from the project root
current_dir = Path(__file__).parent
project_dir = current_dir.parent
sys.path.append(str(project_dir))

# Initialize session state variables
if 'task_result' not in st.session_state:
    st.session_state['task_result'] = None
if 'report' not in st.session_state:
    st.session_state['report'] = ""
if 'fundamental_plots' not in st.session_state:
    st.session_state['fundamental_plots'] = {}
if 'fundamental_plot_names' not in st.session_state:
    st.session_state['fundamental_plot_names'] = []
if 'technical_plots' not in st.session_state:
    st.session_state['technical_plots'] = {}
if 'technical_plot_names' not in st.session_state:
    st.session_state['technical_plot_names'] = []
if 'data_tables' not in st.session_state:
    st.session_state['data_tables'] = {}

# Function to load custom CSS (if you have any)
def load_css(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS if it exists
css_file = project_dir / '.streamlit' / 'styles.css'
if css_file.exists():
    load_css(css_file)

# Title and Description
st.title("💹 Financial Analysis Agent")
st.markdown("""
Enter a prompt below to perform fundamental and technical analysis.
The agent will generate a comprehensive financial report along with relevant plots and tables.
""")

# Input Section within a form
with st.form(key='prompt_form'):
    prompt = st.text_area(
        "Enter your financial analysis prompt:",
        value="Write a financial report on Delta airlines",
        height=150,
    )
    submit_button = st.form_submit_button(label='Analyze')


# Function to run the agent asynchronously
async def run_agent(prompt):
    from agents.coordination_agent import (
        search_agent,
        fundamental_analysis_agent,
        stock_analysis_agent,
        termination,
        report_agent,
        OPENAI_API_KEY
    )

    team = RoundRobinGroupChat(
        [search_agent, fundamental_analysis_agent, stock_analysis_agent, report_agent],
        termination_condition=termination
    )
    
    # Run the team conversation
    final_result = await team.run(task=prompt)
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
        "to produce a comprehensive, cohesive and readable report."
    )

    formatted_prompt = report_prompt.format(conversation=conversation_history_str)
    user_message = TextMessage(content=formatted_prompt, source="User")

    # Create a new conversation with the report agent
    response = await report_agent.on_messages([user_message], None)
    final_report = response.chat_message.content
    st.session_state['report'] = final_report    
    print("Final Report:\n", final_report)
    # Save the final report to a Markdown file
    md_filename = 'Financial_Report.md'
    with open(md_filename, 'w', encoding='utf-8') as md_file:
        md_file.write(final_report)
    print(f"The financial report has been saved as '{md_filename}'.")

    return final_result


# Function to list existing plots in a specified directory
def list_existing_plots(project_dir, plot_type):
    """
    Scans the specified plot directory and returns a dictionary mapping
    user-friendly plot names to their paths.
    """
    plots = {}
    dir_path = project_dir / plot_type
    if dir_path.exists():
        for plot_file in dir_path.glob('*.png'):  # Adjust the pattern if plots are in different formats
            plot_name = plot_file.stem.replace('_', ' ').title()
            plots[plot_name] = str(plot_file)
    else:
        st.warning(f"Plot directory does not exist: {dir_path}")
    return plots

def clean_content_string(content_str):
    # Replace Timestamp('...') with '...'
    timestamp_pattern = r"Timestamp\('([^']+)'\)"
    content_str = re.sub(timestamp_pattern, r"'\1'", content_str)

    # Replace nan with None
    content_str = content_str.replace('nan', 'None')

    # Replace inf and -inf with None
    content_str = content_str.replace('inf', 'None')
    content_str = content_str.replace('-inf', 'None')

    return content_str

# Helper function to extract report and plots from TaskResult
def extract_report_and_plots(task_result, project_dir):
    report = ""
    technical_plots = {}
    fundamental_plots = {}
    data_tables = {}

    # Define the agents that provide plot data
    plot_providers = ['FundamentalAnalyst', 'TechnicalStockAnalyst']

    # Iterate through all messages to find data and plot paths
    for message in task_result.messages:
        # Extract plots and data from specified ToolCallResultMessages
        if isinstance(message, ToolCallResultMessage) and message.source in plot_providers:
            try:
                content = message.content

                if isinstance(content, list):
                    for item in content:
                        # Check if the item is a FunctionExecutionResult
                        if isinstance(item, FunctionExecutionResult):
                            content_str = item.content

                            # Clean the content string
                            content_str = clean_content_string(content_str)

                            # Use ast.literal_eval to parse the content string safely
                            try:
                                data_dict = ast.literal_eval(content_str)

                                # Recursively search for plot file paths and data tables
                                def recursive_extract(d):
                                    for key, value in d.items():
                                        if key.endswith('_plot_file_path'):
                                            # Convert relative paths to absolute paths
                                            absolute_path = project_dir / value
                                            plot_name = key.replace('_plot_file_path', '').replace('_', ' ').title()

                                            # Determine plot type based on directory or key
                                            if 'fundamental_plots' in value:
                                                fundamental_plots[plot_name] = str(absolute_path)
                                            elif 'technical_plots' in value:
                                                technical_plots[plot_name] = str(absolute_path)
                                            else:
                                                # If unable to determine, default to fundamental plots
                                                fundamental_plots[plot_name] = str(absolute_path)
                                        elif isinstance(value, dict):
                                            recursive_extract(value)
                                        else:
                                            data_tables[key] = value

                                recursive_extract(data_dict)

                            except Exception as e:
                                st.error(f"Error parsing content from {message.source}: {e}")
                        else:
                            st.warning(f"Unexpected item type in content from {message.source}: {type(item)}")
                else:
                    st.warning(f"Unexpected content format from {message.source}: {type(content)}")

            except Exception as e:
                st.error(f"Error processing message from {message.source}: {e}")
                continue

    # Additionally, list existing plots from directories
    existing_fundamental_plots = list_existing_plots(project_dir, plot_type='fundamental_plots')
    fundamental_plots.update(existing_fundamental_plots)

    existing_technical_plots = list_existing_plots(project_dir, plot_type='technical_plots')
    technical_plots.update(existing_technical_plots)

    return fundamental_plots, technical_plots, data_tables

# Handle form submission
if submit_button:
    with st.spinner('Analyzing... Please wait.'):
        try:
            # Use asyncio.run to execute the async function
            task_result = asyncio.run(run_agent(prompt))
            st.session_state['task_result'] = task_result

            fundamental_plots, technical_plots, data_tables = extract_report_and_plots(task_result, project_dir)

            # Store extracted data in session state
            st.session_state['fundamental_plots'] = {name: path for name, path in fundamental_plots.items()}
            st.session_state['fundamental_plot_names'] = list(st.session_state['fundamental_plots'].keys())
            st.session_state['technical_plots'] = {name: path for name, path in technical_plots.items()}
            st.session_state['technical_plot_names'] = list(st.session_state['technical_plots'].keys())
            st.session_state['data_tables'] = data_tables

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state['task_result'] = None
            st.session_state['fundamental_plots'] = {}
            st.session_state['fundamental_plot_names'] = []
            st.session_state['technical_plots'] = {}
            st.session_state['technical_plot_names'] = []
            st.session_state['data_tables'] = {}
            st.session_state['report'] = ""

# After handling form submission, display the plots and data tables if available
if st.session_state.get('task_result'):
    # **Debug Statements**
    #st.markdown("### **Debug Information**")
    #st.write(f"**task_result:** {st.session_state['task_result']}")
    #st.write(f"**Type of task_result:** {type(st.session_state['task_result'])}")
    #st.write(f"**Attributes of task_result:** {dir(st.session_state['task_result'])}")
    #st.markdown("---")
    #st.write(f"**Stop Reason:** {st.session_state['task_result'].stop_reason}")
    # Display the Report
    if st.session_state.get('report'):
        #st.subheader("📄 Financial Report")
        st.markdown(st.session_state['report'])  # Use markdown to render the report with formatting

        # Allow downloading the report as a Markdown file
        st.download_button(
            label="Download Report",
            data=st.session_state['report'],
            file_name=f"Financial_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )
    else:
        st.info("No report was generated.")
        
    # Display data tables if available
    if st.session_state['data_tables']:
        st.subheader("📊 Financial Data and Ratios")
        for key, value in st.session_state['data_tables'].items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                # Convert list of dictionaries to DataFrame
                df = pd.DataFrame(value)
                # Convert 'None' strings to actual NoneType
                df.replace('None', np.nan, inplace=True)
                # Convert date strings to datetime objects if necessary
                if 'end' in df.columns:
                    df['end'] = pd.to_datetime(df['end'], errors='coerce')
                # Handle special data types in DataFrame (e.g., Timestamp, nan)
                df.replace({pd.NaT: None, np.nan: None}, inplace=True)
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                st.dataframe(df)
            else:
                # Display other data as needed
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

    # Display Fundamental Analysis Plots
    if st.session_state['fundamental_plots']:
        st.subheader("📊 Fundamental Analysis Plots")
        selected_fundamental_plot_name = st.selectbox(
            "Select a fundamental analysis plot to display:",
            st.session_state['fundamental_plot_names'],
            key='fundamental_plot_selector'
        )
        if selected_fundamental_plot_name:
            plot_path = st.session_state['fundamental_plots'][selected_fundamental_plot_name]
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)
            else:
                st.error(f"Plot file not found: {plot_path}")
    else:
        st.info("No fundamental analysis plots were generated.")

    # Display Technical Analysis Plots
    if st.session_state['technical_plots']:
        st.subheader("📈 Technical Analysis Plots")
        selected_technical_plot_name = st.selectbox(
            "Select a technical analysis plot to display:",
            st.session_state['technical_plot_names'],
            key='technical_plot_selector'
        )
        if selected_technical_plot_name:
            plot_path = st.session_state['technical_plots'][selected_technical_plot_name]
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)
            else:
                st.error(f"Plot file not found: {plot_path}")
    else:
        st.info("No technical analysis plots were generated.")

else:
    if st.session_state.get('fundamental_plots') or st.session_state.get('technical_plots'):
        st.warning("No analysis has been performed yet. Please enter a prompt and click 'Analyze'.")

# Sidebar Information
st.sidebar.header("About")
st.sidebar.info("""
**Financial Analysis Agent** performs fundamental and technical analysis to generate relevant plots and data tables.

Developed by [Siva Yellepeddi](https://yourwebsite.com)
""")

# Footer
st.markdown("""
---
© 2024 [Siva Yellepeddi](https://yourwebsite.com) | Powered by [Streamlit](https://streamlit.io/)
""")