import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
SEC_API_KEY = os.getenv("SEC_API_KEY")

# Validate that the OpenAI API key is provided
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Corrected model name (assuming "gpt-4-mini" is intended)
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",  
            "api_key": OPENAI_API_KEY,
            "api_type": "openai",
        }
    ]
}