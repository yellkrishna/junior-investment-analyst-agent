# Multi-Agent Junior Investment Analyst

## Table of Contents

- [Introduction](#introduction)

- [Project Overview](#project-overview)

- [Features](#features)

- [Architecture Overview](#architecture-overview)

- [Prerequisites](#prerequisites)

- [Installation](#installation)

- [Setup](#setup)

- [How to Run](#how-to-run)

- [Usage](#usage)

- [Project Structure](#project-structure)

- [Contributing](#contributing)

- [License](#license)

- [Acknowledgments](#acknowledgments)

## Introduction

The **Multi-Agent Junior Investment Analyst** is an AI-powered tool designed to assist investment analysts by automating the fundamental and technical analysis of publicly traded companies. This system simulates collaboration between specialized agents to generate comprehensive financial reports, enabling analysts to quickly appraise a company.

## Project Overview

This project implements a multi-agent system using Microsoft's AutoGen framework to coordinate multiple specialized agents. The agents collaborate to perform tasks such as data retrieval, fundamental analysis, technical analysis, and report generation. The user interacts with a web application built with Streamlit to input prompts and receive detailed financial reports.

## Features

- **Fundamental Analysis**: Extracts key financial statements and calculates essential financial ratios from EDGAR filings.

- **Technical Analysis**: Analyzes stock price trends, patterns, and volatility using historical price and volume data.

- **Google Search**: Gather all required information by doing google searches for both technical and fundamental analysis.

- **Report**: Creates comprehensive financial report based on the Technical, Fundamental and Google researches.

- **Interactive Web Interface**: Provides a user-friendly interface for inputting prompts and viewing reports.

- **Question & Answer Section**: Allows users to ask questions about the generated report using Retrieval Augmented Generation (RAG).

- **Downloadable Reports**: Users can download the generated financial reports in Markdown format.

## Architecture Overview

The system comprises the following agents:

1\. **Fundamental Analysis Agent**:

   - **Data Source**: EDGAR filings.

   - **Responsibilities**:

     - Extract income statements, balance sheets, and cash flow statements.

     - Assess financial health and performance trends.

     - Calculate financial ratios like ROE, debt-to-equity ratio, etc.

2\. **Technical Analysis Agent**:

   - **Data Source**: Historical stock price and volume data.

   - **Responsibilities**:

     - Analyze stock price trends and patterns.

     - Identify technical indicators such as moving averages and support/resistance levels.

2\. **Google Agent**:

   - **Data Source**: Current information about the company from Google.

   - **Responsibilities**:

     - Agent specialized in retrieving a company's stock ticker, benchmark ticker, and comprehensive information via Google searches..

     - Provides top results with snippets and content to support financial analysis for fundamental analysis agent and technical analysis agent and SWOT analysis for report agent..

3\. **Coordination Agent**:

   - **Responsibilities**:

     - Distribute tasks among the fundamental and technical analysis agents.

     - Aggregate analyses into a cohesive final report.

     - Ensure consistency and coherence in formatting and content.

4\. **Report Agent**:

   - **Responsibilities**:

     - Aggregates fundamental analysis data (financial ratios and qualitative insights), technical analysis results, SWOT analysis, and compares technical indicators with benchmarks, and incorporates information from Google searches to produce a cohesive and readable report. Be comprehensive and based on the indicators and fundamentals of the company.

     - Generate a comprehensive financial report in Markdown format.

5\. **Retrieval Augmented Generation (RAG) Agents**:

   - **Assistant Agent**: Provides answers to user queries based on the generated report.

   - **Retrieve User Proxy Agent**: Handles the retrieval of relevant information for Q&A.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System**: Windows, macOS, or Linux.

- **Python**: Version 3.8 or higher.

- **API Keys**:

  - **OpenAI API Key**: Required for language model interactions.

  - **GOOGLE_SEARCH_ENGINE_ID**: Required for doing google searches data.

  - **GOOGLE_API_KEY**: Required for doing google searches data.


- **Git**: For cloning the repository.

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash

git clone https://github.com/yourusername/multi-agent-investment-analyst.git

cd multi-agent-investment-analyst

### 2\. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

bash

Copy code

`python -m venv venv`

Activate the virtual environment:

-   **Windows**:

    bash

    Copy code

    `venv\Scripts\activate`

-   **macOS/Linux**:

    bash

    Copy code

    `source venv/bin/activate`

### 3\. Install Dependencies

Install the required Python packages using `pip`:

bash

Copy code

`pip install -r requirements.txt`

Setup
-----

### 1\. Obtain API Keys

-   **OpenAI API Key**:
    -   Sign up at [OpenAI](https://openai.com/) to get your API key.
-   **Alpha Vantage API Key**:
    -   Sign up at [Alpha Vantage](https://www.alphavantage.co/) to get your free API key.

### 2\. Create a `.env` File

In the root directory of the project, create a `.env` file to store your API keys and configurations.

bash

Copy code

`touch .env`

Add the following content to the `.env` file:

env

Copy code

`OPENAI_API_KEY=your-openai-api-key
ALPHAVANTAGE_API_KEY=your-alpha-vantage-api-key`

**Note**: Replace `your-openai-api-key` and `your-alpha-vantage-api-key` with your actual API keys.

### 3\. Configure `config.py`

Ensure that your `config.py` file reads the API keys from the `.env` file.

python

Copy code

`# config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# LLM Configuration
llm_config = {
    "config_list": [
        {
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
            "model": "gpt-4",
            "use_stream": False,
            "temperature": 0.5,
        }
    ]
}`

How to Run
----------

### 1\. Start the Streamlit App

In the terminal, run the following command from the project root directory:

bash

Copy code

`streamlit run frontend/app.py`

### 2\. Access the Application

After running the above command, Streamlit will provide a local URL (usually `http://localhost:8501/`). Open this URL in your web browser to access the application. Please be patient, it takes a minute for the application to load. 

Usage
-----

1.  **Enter a Prompt**:

    -   In the text area labeled "Enter your financial analysis prompt," input a prompt such as:

        text

        Copy code

        `Write a financial report on Apple Inc.`

    -   Click the **Analyze** button.

2.  **Wait for Analysis**:

    -   The application will display a spinner while the agents perform the analysis.
    -   Once completed, a comprehensive financial report will be displayed.
3.  **View Plots and Data Tables**:

    -   Fundamental and technical analysis plots can be viewed by selecting them from the dropdown menus.
    -   Financial data and ratios are displayed in interactive tables.
4.  **Ask Questions**:

    -   Scroll down to the "Ask a Question" section.
    -   Enter a question related to the generated report.
    -   Submit the question to receive an answer from the assistant agent.
5.  **Download the Report**:

    -   Use the **Download Report** button to download the financial report in Markdown format.

Project Structure
-----------------

arduino

Copy code

`multi-agent-investment-analyst/
│
├── agents/
│   ├── google_search/
│   │     ├── google_search.py
│   │     ├── __init__.py.py
│   ├── fundamental_analysis_agent/
│   │     ├── __init__.py.py
│   │     ├── fundamental_analysis_agent.py
│   │     ├── extract_filing_details.py.py
│   │     ├── match_company_concepts.py
│   │     ├── template_matching.py
│   ├── technical_analysis_agent/
│   │     ├── __init__.py.py
│   │     ├── technical_analysis_agent.py
│   └── coordination_agent.py
│
├── frontend/
│   ├── __init__.py
│   ├── app.py
│   └── utils.py
│
├── assets/
│   ├── company_logo.png
│
├── fundamental_plots/
│   └── ... [Generated reports]
│
├── technical_plots/
│   └── ... [Generated reports]
│
├── Reports/
│   └── ... [Generated reports]
│
├── .streamlit/
│   ├── config.toml
│   └── styles.css
│
├── .gitignore
├── .env
├── config.py
├── requirements.txt
└── README.md`

Contributing
------------

Contributions are welcome! Please follow these steps:

1.  **Fork the repository**.

2.  **Create a new branch**:

    bash

    Copy code

    `git checkout -b feature/YourFeature`

3.  **Make your changes and commit them**:

    bash

    Copy code

    `git commit -m 'Add some feature'`

4.  **Push to the branch**:

    bash

    Copy code

    `git push origin feature/YourFeature`

5.  **Open a pull request**.

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
---------------

-   **Microsoft's AutoGen Framework**: For providing the agent orchestration capabilities.
-   **Streamlit**: For the interactive web application framework.
-   **OpenAI**: For the language models powering the agents.