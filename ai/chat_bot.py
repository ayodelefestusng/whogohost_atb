# # # ==========================
# # # 🌐 Django & Project Settings (Commented out as not used in standalone script)
# # # ==========================
# from django.conf import settings
# from .models import Prompt,Prompt7

import base64
from collections import UserDict
import io
import json
from functools import partial # 🚨 FIX: IMPORTED partial
# # ==========================
# # 📦 Standard Library
# # ==========================
import os
# from langgraph.checkpoint.postgres import PostgresSaver
# # --- Project-Specific Imports ---
# # AJADI-2
import re
# from pprint import pprint
import sqlite3
from datetime import datetime
from io import BytesIO
# from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

# import matplotlib.pyplot as plt
# # ==========================
import pandas as pd
from django.conf import settings

# # ==========================
# # 📦 Third-Party Core
# ==========================
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
# # # ==========================
# # # 🤖 LangChain Core & Community
# # # ==========================
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import Tool  # Explicitly import Tool
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_community.vectorstores import Chroma # Import Chroma
from langchain_chroma import Chroma
from langchain_deepseek import \
    ChatDeepSeek  # Import ChatDeepSeek for DeepSeek LLM
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_groq import ChatGroq  # For Groq LLM
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver  # Using SqliteSaver as preferred
# # # ==========================
# # # 🔁 LangGraph Imports
# # # ==========================
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
# from matplotlib.ticker import FuncFormatter
from PIL import Image
from pydantic import BaseModel, Field

# # ==========================
# # 🧠 Google Generative AI
# # ==========================
# import google.generativeai as genai
# from google.generativeai import GenerativeModel, configure
# from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load .env file
load_dotenv()
# import matplotlib

# matplotlib.use('Agg') # This prevents Matplotlib from trying to open a GUI window

# import matplotlib

# This must be done BEFORE importing pyplot
# matplotlib.use('Agg')

import base64
import io
import logging
import re
from io import BytesIO

# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.ticker import FuncFormatter
from sqlalchemy import create_engine
from .models import Tenant

from langchain_openai import OpenAIEmbeddings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ==========================
# ⚙️ Configuration & Initialization
# ==========================
# Load API keys from environment variables for security
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_API_KEY = "AIzaSyBV7_Cbak1LhE2bHK_aG4ARaa6anxjBClY"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Ensure this is set in .env if used
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure this is set in .env if used
# PDF_PATH = os.getenv("PDF_PATH", "default.pdf") # Default value for PDF_PATH

# # Set environment variables for LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY if LANGSMITH_API_KEY else ""
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT if LANGSMITH_PROJECT else "Agent_Creation"
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT if LANGSMITH_ENDPOINT else "https://api.smith.langchain.com"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY if GOOGLE_API_KEY else ""
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY if TAVILY_API_KEY else ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY if GROQ_API_KEY else ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""
tavily_search = TavilySearch(max_results=2)
# Configure Google Generative AI
# genai.configure(api_key=GOOGLE_API_KEY)
# safety_settings = {
#     "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#     "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
# }

def safe_json(data):
    """Ensures safe JSON serialization to prevent errors."""
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return json.dumps({})  # Returns an empty JSON object if serialization fails

# chatbot_model =py.chatbot_model
google_model="gemini-2.0-flash"

llm = init_chat_model("google_genai:gemini-flash-latest")
model = llm # Consistent naming

# tenant_id =get_tenant()
# tenant = Tenant.objects.get(tenant_id=tenant_id)  # Get the existing record

# Initialize Embeddings and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest")
# global_vector_store = None

# Call the initialization function
# initialize_vector_store()
def initialize_vector_store(tenant_id: str):
    """Initializes and connects to the Chroma vector store for a specific tenant."""
    
    # 1. Define a persistent directory path for the tenant's vector store
    # This stores the vector data in a dedicated folder (e.g., project_root/chroma_dbs/tenant_123)
    persist_directory = os.path.join(settings.BASE_DIR, "chroma_dbs", tenant_id)
    os.makedirs(persist_directory, exist_ok=True)
    
    # 2. Connect to the persistent Chroma collection
    # The collection name can be static, as the directory is tenant-specific
    vector_store = Chroma(
        collection_name=f"tenant_{tenant_id}_rag",
        embedding_function=embeddings, # Your chosen embedding model
        persist_directory=persist_directory 
    )

    # 3. Check if the store is empty (i.e., this is the first time loading the PDF)
    # The actual retrieval/storage logic needs to be moved here, NOT in the global scope.
    if vector_store._collection.count() == 0:
        # Load the tenant object here to get the file path
        try:
            tenant = Tenant.objects.get(tenant_id=tenant_id)
            file_path = tenant.tenant_kss.path
        except (Tenant.DoesNotExist, AttributeError):
            print("Tenant profile not found or missing path.")
            return None # Return None if initialization fails

        if file_path and os.path.exists(file_path):
            # Load and split PDF logic
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            )
            # Add documents to the persistent store
            vector_store.add_documents(documents=docs)
            print(f"PDF loaded and persisted for tenant {tenant_id}.")
            
    # 4. Return the initialized, persistent vector store instance
    return vector_store

# Initialize SQLDatabase with the specified file path
DB_URI = f"sqlite:///{settings.DATABASES['default']['NAME']}"

# DB_URI = os.getenv("DB_URI")
db = None
try:
    db = SQLDatabase.from_uri(DB_URI)
    print(f"SQLDatabase connected to {DB_URI} successfully.")
except Exception as e:
    print(f"Error connecting to SQLDatabase at {DB_URI}: {e}. SQL query tool will not be available.")

tavily_search = TavilySearch(max_results=2)

# ==========================
# 📊 Pydantic Schemas (Revised for Clarity)
# ==========================

class Answer(BaseModel):
    """The final, structured answer for the user."""
    answer: str = Field(description="Polite, empathetic, and direct response to the user's query.")
    sentiment: int = Field(description="User's sentiment score from -2 (very negative) to +2 (very positive).")
    ticket: List[str] = Field(description="Relevant service channels for unresolved issues (e.g., 'POS', 'ATM').")
    source: List[str] = Field(description="Sources used to generate the answer (e.g., 'PDF Content', 'Web Search').")
    chart_base64: Optional[str] = Field(default=None, description="A base64 encoded PNG image of the generated chart, if any.")

class Summary(BaseModel):
    """Conversation summary schema."""
    summary: str = Field(description="A concise summary of the entire conversation.")
    sentiment: int = Field(description="Overall sentiment of the conversation from -2 to +2.")
    unresolved_tickets: List[str] = Field(description="A list of channels with unresolved issues.")
    all_sources: List[str] = Field(description="All unique sources referenced throughout the conversation.")

class PDFRetrievalInput(BaseModel):
    """Input schema for the pdf_retrieval_tool."""
    query: str = Field(description="The user's query to search for within the PDF document.")

class WebSearchInput(BaseModel):
    """Input schema for the tavily_search_tool."""
    query: str = Field(description="A concise search query for the web.")

class SQLQueryInput(BaseModel):
    """Input schema for the sql_query_tool."""
    query: str = Field(description="The natural language question to be converted into a SQL query.")

# ==========================
# 📊 State Management (Simplified and Centralized)
# ==========================
# class VisualizationInput(BaseModel):
#     """Input schema for the generate_visualization_tool."""
#     query: str = Field(description="The user's natural language request for a chart or visualization, e.g., 'Plot the total sales by region'.")

class State(MessagesState):
    """Manages the conversation state. Uses Pydantic models for structured data."""
    # Tool outputs
    pdf_content: Optional[str] = None
    web_content: Optional[str] = None
    sql_result: Optional[str] = None
    # visualization_result: Optional[Dict[str, Any]] = None # <-- NEW

    attached_content: Optional[str] = None
    last_tool_name: Optional[str] = Field(default=None)

    # Final structured outputs
    final_answer: Optional[Answer] = None
    conversation_summary: Optional[Summary] = None

    # For final logging
    metadatas: Optional[Dict[str, Any]] = None
    tenant_config: Optional[Dict[str, Any]] = Field(default=None) # <-- NEW

# ==========================
# 🛠️ Tools
# ==========================

def get_time_based_greeting():
    """Return an appropriate greeting based on the current time."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12: return "Good morning"
    if 12 <= current_hour < 17: return "Good afternoon"
    return "Good evening"

def retrieve_from_pdf(query: str, vector_store: Chroma) -> dict: 
    """Performs a document query using the bound vector store."""
    if vector_store:
        try:
            results = vector_store.similarity_search(query, k=3)
            content = "\n\n".join([doc.page_content for doc in results])
            return {"pdf_content": content}
        except Exception as e:
            # Added better error logging here
            print(f"Error during PDF search (Chroma): {e}")
            return {"pdf_content": f"Error: Failed to retrieve from PDF: {e}"}
            
    return {"pdf_content": "Error: Document knowledge base not initialized."}

# NOTE: pdf_retrieval_tool definition is intentionally REMOVED from global scope 
# and created dynamically in process_message.

def search_web_func(query: str) -> dict: # 🚨 NOTE: Changed to return dict for ToolMessage
    """Performs web search and returns structured tool output."""
    try:
        search_docs = tavily_search.invoke(query)
        formatted_docs = "\n\n---\n\n".join(
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs.get("results", [])
        )
        return {"web_content": formatted_docs or "No results found from web search."}
    except Exception as e:
        print(f"Web search error: {e}")
        return {"web_content": f"Error during web search: {e}"}

tavily_search_tool = Tool(
    name="tavily_search_tool",
    description="Useful for general questions or questions requiring up-to-date information from the web. Input should be a concise search query.",
    func=search_web_func,
    args_schema=WebSearchInput,
)

# Updated `update_state_after_tool_call` function
def update_state_after_tool_call(state: State) -> dict:
    """
    Updates the specific state field with the output from the last tool call.
    """
    print("--- UPDATING STATE FROM TOOL OUTPUT ---")
    last_message = state["messages"][-1]
    
    # Ensure the last message is a ToolMessage
    if not isinstance(last_message, ToolMessage):
        return {}

    tool_name = state.get("last_tool_name")
    
    # Check if content is a stringified dictionary
    try:
        tool_output_dict = json.loads(last_message.content)
        if tool_name == "pdf_retrieval_tool":
            tool_output = tool_output_dict.get('pdf_content', 'Error: Missing PDF content key.')
        elif tool_name == "tavily_search_tool":
            tool_output = tool_output_dict.get('web_content', 'Error: Missing web content key.')
        else:
            tool_output = last_message.content # Fallback
    except json.JSONDecodeError:
        tool_output = last_message.content # Not JSON, use raw content
    
    print(f"Tool '{tool_name}' returned: {tool_output[:200]}...")

    if tool_name == "pdf_retrieval_tool":
        return {"pdf_content": tool_output}
    elif tool_name == "tavily_search_tool":
        return {"web_content": tool_output}

    return {}

def agent_node(state: State):
    """
    The Router Node: Decides whether to call a tool or generate a final answer.
    """
    print("--- AGENT NODE (ROUTER) ---")
    messages = state["messages"]
    tenant_config = state["tenant_config"] 

    if len(messages) == 1:
        # 🚨 FIX: Handle initial message/greeting turn
        greeting = tenant_config.get("greeting", "How can I help?") 
        new_messages = messages + [AIMessage(content=f"{get_time_based_greeting()}! {greeting}")]
        
        # Special flag to end the flow immediately after greeting
        return {"messages": new_messages, "last_tool_name": "GREETING_SENT"} 
    
    # Get tools used by the graph. Since tools are dynamically set in process_message, 
    # we must use a minimal global list for LLM binding here to prevent errors, 
    # or rely on the LangGraph engine's tool setup if it handles it implicitly.
    # For safety, we use only the available global tool for LLM binding.
    # Note: LangGraph's ToolNode handles execution, the LLM just needs the definition.
    tools_for_llm = [tavily_search_tool] 
    # Assuming pdf_retrieval_tool is not globally defined, we rely on LangGraph's setup 
    # which takes the tools_list from the build_graph call. 
    
    system_prompt = SystemMessage(
        content=tenant_config.get("agent_prompt", "You are a helpful AI assistant.")
    )
    
    # We bind the globally available tool, knowing the graph's ToolNode will use the full, bound list.
    llm_with_tools = llm.bind_tools(tools_for_llm) 
    response = llm_with_tools.invoke([system_prompt] + messages)
    
    last_tool_name = None
    if response.tool_calls:
        last_tool_name = response.tool_calls[0]['name']
        print(f"LLM decided to call tool: {last_tool_name}")
        
    return {"messages": [response], "last_tool_name": last_tool_name}

def generate_final_answer_node(state: State):
    """
    The Generator Node: Creates the final structured answer after gathering all necessary context from tools.
    """
    print("--- GENERATE FINAL ANSWER NODE ---")
  
    messages = state["messages"] 
    tenant_config = state["tenant_config"] 

    # Find the last HumanMessage
    user_query = " "
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
            
    if not user_query:
        user_query = "The user asked an unrecoverable question."
    print("Aleu",user_query)
   
    print ("Lemu",state.get("pdf_content"))

    
    context_parts = []
    if state.get("pdf_content"): context_parts.append(f"PDF Content:\n{state['pdf_content']}")
    if state.get("web_content"): context_parts.append(f"Web Content:\n{state['web_content']}")
    if state.get("attached_content"): context_parts.append(f"Attached Content:\n{state['attached_content']}")
    context = "\n\n".join(context_parts) if context_parts else "No additional context was retrieved."

    print ("Lemu",state.get("pdf_content"))

    # 🚨 FIX: Ensure prompt is defined and safely formatted
    prompt = ""  
    
    # Get the template, using a sane default if the tenant config is missing
    prompt_template = tenant_config.get("final_prompt", 
                                       "Based on the following context, answer the user's question clearly and professionally.\n\nUser Question: {0}\n\nContext:\n{1}") 
    print(f"Using prompt template: {prompt_template[:50]}...")

    
    try:
        # Format the final prompt
        prompt = prompt_template.format(user_query, context)
    except Exception as e:
        # Fallback if formatting fails (e.g., mismatch in {0}, {1}, etc.)
        print(f"Error formatting final prompt: {e}. Falling back to default prompt structure.")
        # Create a safe, guaranteed prompt
        prompt = (f"User Question: {user_query}\n\nContext: {context}\n\n"
                  "Please generate a structured answer based on the provided information.")
    # 🚨 FIX END
    
   
    structured_llm = llm.with_structured_output(Answer)
    final_answer_obj = structured_llm.invoke(prompt)
    
    # Append the human-readable part of the answer to the message history
    new_messages = state["messages"] + [AIMessage(content=final_answer_obj.answer)]
    return {
        "final_answer": final_answer_obj,
        "messages": new_messages
    }

def should_continue_or_end(state: State) -> str:
    """Routes the flow based on tool calls or the special greeting flag."""
    # 🚨 FIX: Handles the 'GREETING_SENT' flag
    if state.get("last_tool_name") == "GREETING_SENT":
        return END
    
    if state["messages"][-1].tool_calls:
        return "tools"
        
    return "generate_final_answer"

def summarize_conversation(state: State):
    print("--- SUMMARIZE CONVERSATION NODE ---")
    
    messages = state.get("messages", [])
    tenant_config = state["tenant_config"] 

    if not messages:
        return {
            "conversation_summary": None,
            "metadatas": {"error": "No messages to summarize."}
        }

    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

    # 🚨 FIX: Provide a safe, format-ready default template
    summarize_prompt_template = tenant_config.get(
        "summary_prompt", 
        "Summarize the conversation below, determining sentiment and any unresolved issues, using only the structured output schema:\n\nConversation:\n{0}"
    ) 
    try:
        summarize_prompt = summarize_prompt_template.format(conversation_history)
    except Exception as e:
        print(f"ERROR: Summary prompt format failed: {e}. Using raw history.")
        summarize_prompt = "Summarize the following raw history: " + conversation_history
    
    # ... (rest of the summarize node logic)

    structured_llm = llm.with_structured_output(Summary)
    summary_obj = structured_llm.invoke(summarize_prompt)
    
    # Create the final metadata dictionary for logging
    final_answer = state.get("final_answer")
    last_user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_question = msg.content
            break

    metadata_dict = {
        "question": last_user_question, # Use the robustly found question
        "answer": final_answer.answer if final_answer else "N/A",
        "sentiment": final_answer.sentiment if final_answer else 0,
        "ticket": final_answer.ticket if final_answer else [],
        "source": final_answer.source if final_answer else [],
        "chart_base64": final_answer.chart_base64 if final_answer else None,
        "summary": summary_obj.summary,
        "summary_sentiment": summary_obj.sentiment,
        "summary_unresolved_tickets": summary_obj.unresolved_tickets,
        "summary_sources": summary_obj.all_sources,
    }

    return {
        "conversation_summary": summary_obj,
        "metadatas": metadata_dict
    }

# Define the condition to check if a tool was called
def should_continue(state: State) -> str:
    """Determines the next step: call a tool or generate the final answer."""
    if state["messages"][-1].tool_calls:
        return "tools"
    return "generate_final_answer"

# ==========================
# 🔄 Graph Workflow
# ==========================

def build_graph(tools_list): # 🚨 FIX: Accepts the dynamic tools list
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(State)
    
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tools", ToolNode(tools=tools_list)) # 🚨 FIX: Uses the dynamic list
    workflow.add_node("update_state", update_state_after_tool_call)

    workflow.add_node("generate_final_answer", generate_final_answer_node)
    workflow.add_node("summarize", summarize_conversation)

    workflow.set_entry_point("agent_node")

    # 🚨 FIX: Use the custom router to handle greetings and tool calls
    workflow.add_conditional_edges(
        "agent_node", 
        should_continue_or_end, 
        {
            "tools": "tools",
            "generate_final_answer": "generate_final_answer",
            END: END 
        }
    )

    workflow.add_edge("tools", "update_state")
    workflow.add_edge("update_state", "agent_node")
    workflow.add_edge("generate_final_answer", "summarize")
    workflow.add_edge("summarize", END)
    
    # Initialize checkpointing with a robust fallback
    memory = None
    try:
        # Create a path for the checkpointing database
        checkpoint_dir = os.path.dirname(settings.DATABASES['default']['NAME'])
        checkpoint_db = os.path.join(checkpoint_dir, 'checkpoints.sqlite')
        
        # Ensure the directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Connect to the SQLite database
        conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
        memory = SqliteSaver(conn=conn)
        print("SQLite checkpointing connected successfully.")
    except Exception as e:
        print(f"Error connecting to SQLite for checkpointing: {e}. Using in-memory saver.")
        # memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)

# Main processing function
def process_message(message_content: str, session_id: str, tenant_id: str, file_path: Optional[str] = None):
    """Main function to process user messages using the LangGraph agent."""

    # --- DYNAMIC INITIALIZATION ---
    try:
        current_tenant = Tenant.objects.get(tenant_id=tenant_id)
    except Tenant.DoesNotExist:
        return {"answer": "Error: Tenant configuration not found.", "chart": None, "metadata": {}}

    # ⚠️ Initialize the tenant-specific, persistent vector store
    tenant_vector_store = initialize_vector_store(tenant_id)
    
    # 🚨 FIX: Create the RAG tool binding and define the tool list here!
    # 1. Bind the vector store instance to the retrieval function
    pdf_retrieval_bound = partial(retrieve_from_pdf, vector_store=tenant_vector_store)

    # 2. Redefine the PDF tool using the bound function
    pdf_retrieval_tool_dynamic = Tool(
        name="pdf_retrieval_tool",
        description="Useful for answering questions from the bank's internal knowledge base (PDFs). Input should be a specific question.",
        func=pdf_retrieval_bound,
        args_schema=PDFRetrievalInput,
    )
    
    # 3. Create the final tools list to pass to the graph builder
    local_tools = [pdf_retrieval_tool_dynamic, tavily_search_tool]
    graph = build_graph(local_tools) # 🚨 FIX: Pass the dynamic tools list

    config = {"configurable": {"thread_id": session_id}}
    
    attached_content = None # Simplified for this example
    # Image processing logic can be added here as in the original code
    if file_path:
        try:
            image = Image.open(file_path)
            image.thumbnail([512, 512]) # Resize for efficiency
            
             # Detect format and set MIME type
            image_format = image.format.upper()
            if image_format not in ["PNG", "JPEG", "JPG"]:
                raise ValueError(f"Unsupported image format: {image_format}")

            mime_type = "jpeg" if image_format in ["JPEG", "JPG"] else "png"

            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format=image_format)
            
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_uri = f"data:image/{mime_type};base64,{img_str}"
            
            if image_uri:
                # prompt = "Describe the content of the picture in detail."
                os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
                
                prompt = "Generate the message in the content of the picture."
                message = HumanMessage(
                content=[
                {"type": "text", "text": prompt, },
                { "type": "image_url","image_url": {"url": image_uri}},])
                # Invoke the model with the message
                response = llm.invoke([message])
                # Invoke the model
                response = llm.invoke([message])
                attached_content = response.content

                print("Attached content from image:", attached_content)

        except Exception as e:
            print(f"Error processing image attachment: {e}")
            attached_content = f"Error: Could not process attached file ({e})"
    elif file_path:
        print(f"Warning: Attached file not found at {file_path}. Skipping image processing.")
    
    # initial_state = {"messages": [HumanMessage(content=message_content)], "attached_content": attached_content}
    # Pass the tenant object/prompts/config into the initial state
    initial_state = {
        "messages": [HumanMessage(content=message_content)], 
        "attached_content": attached_content,
        "tenant_config": {
            "greeting": current_tenant.chatbot_greeting,
            "agent_prompt": current_tenant.agent_node_prompt,
            "final_prompt": current_tenant.final_answer_prompt,
            "summary_prompt": current_tenant.summary_prompt,
        }
    }
    
    output = graph.invoke(initial_state, config)
    print("--- LangGraph workflow completed ---")
    
    # Extract final answer from the structured Pydantic object
    final_answer_obj = output.get('final_answer')
    final_answer_content = final_answer_obj.answer if final_answer_obj else "No final answer was generated."
    if final_answer_obj:
        return {
            "answer": final_answer_content,
            "chart": final_answer_obj.chart_base64, # <-- Pass chart data to the view
            "metadata": output.get("metadatas", {})
        }

    else:
        # Handles cases where the greeting causes the END state without a final_answer object
        last_message = output.get("messages", [AIMessage(content="Internal error.")] )[-1]
        return {
            "answer": last_message.content,
            "chart": None,
            "metadata": {}
        }