import os
from autogen import config_list_from_json
from autogen.agentchat import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from autogen import retrieve_utils
import autogen
from sentence_transformers import SentenceTransformer
import config 

# SET Model API Key
openai_api_key = config.OPENAI_API_KEY
# Access the RAG configuration
llm_config  = config.llm_config

# Initialize the RetrieveAssistantAgent
assistant = AssistantAgent(
    name="assistant",
    system_message="You are helpful assistant.",
    llm_config=llm_config,
)

document_content = """
Siva is an extremely hard working and dilligent engineer. 
He has 1 master in structural engineering and 2 masters in computer science.
He was born in India and is currently on a work visa in the United States.
"""


# Initialize the RetrieveUserProxyAgent with appropriate configurations
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": [
            document_content
        ],
        "chunk_token_size": 2000,
        "model": llm_config["config_list"][0]["model"],
        "embedding_model": "all-mpnet-base-v2",
        "vector_db": "chroma",
        "overwrite": False,
        "get_or_create": True,
    },
    code_execution_config=False,
)

if __name__ == "__main__":
    # 
    assistant.reset()
    # Initialize the RetrieveUserProxyAgent
    ragproxyagent.initiate_chat(
        assistant,
        message=ragproxyagent.message_generator,
        problem="How many masters does Siva have?",
        n_results=5  # Number of relevant document chunks to retrieve
    )














    

"""# Function to ingest text into the vector database
def ingest_text(text):
    # Split text into chunks
    chunks = autogen.retrieve_utils.split_text_to_chunks(
        text, max_tokens=2000, chunk_mode="multi_lines"
    )

    # Assuming you have a function to generate embeddings
    embeddings = generate_embeddings(chunks, model="all-mpnet-base-v2")
    # Add chunks to the vector database
    vector_db.add_texts(
        texts=chunks,
        collection_name=collection_name,
        embedding_model="all-mpnet-base-v2",
    )


# Interactive loop for user actions
while True:
    action = input("Choose an action: [ingest/ask/exit]: ").strip().lower()

    if action == "ingest":
        text = input("Enter the text to ingest: ").strip()
        if text:
            ingest_text(text, chroma_client, "user_input_texts")
            print("Text ingested successfully.")
    elif action == "ask":
        question = input("Enter your question: ").strip()
        if question:
            ragproxyagent.initiate_chat(
                assistant,
                message=ragproxyagent.message_generator,
                problem=question,
            )
    elif action == "exit":
        break
    else:
        print("Invalid action. Please choose 'ingest', 'ask', or 'exit'.")"""
