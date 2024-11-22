# frontend/utils.py

import streamlit as st
import os
import re
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from autogen_agentchat.messages import ToolCallResultMessage, FunctionExecutionResult

# Function to load custom CSS
def load_css(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

# Function to clean content strings by replacing problematic patterns
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
                                        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                                            # If value is a list of dicts, add it to data_tables
                                            data_tables[key] = value
                                        else:
                                            # For other data types, you might choose to store them differently or ignore
                                            pass

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
