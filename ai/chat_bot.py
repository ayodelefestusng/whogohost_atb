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
from xml.dom.minidom import Document

from langchain_core.documents import Document

# import matplotlib.pyplot as plt
# # ==========================
# import pandas as pd
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
from sqlalchemy import Boolean, create_engine
from .models import Conversation, Tenant

from langchain_openai import OpenAIEmbeddings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ==========================
# ⚙️ Configuration & Initialization
# ==========================
# Load API keys from environment variables for security

# GOOGLE_API_KEY = "AIzaSyBV7_Cbak1LhE2bHK_aG4ARaa6anxjBClY"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
# LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
# LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Ensure this is set in .env if used
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure this is set in .env if used
# PDF_PATH = os.getenv("PDF_PATH", "default.pdf") # Default value for PDF_PATH

# # Set environment variables for LangSmith
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY if LANGSMITH_API_KEY else ""
# os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT if LANGSMITH_PROJECT else "Agent_Creation"
# os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT if LANGSMITH_ENDPOINT else "https://api.smith.langchain.com"
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY if GOOGLE_API_KEY else ""
# os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY if TAVILY_API_KEY else ""
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY if GROQ_API_KEY else ""
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""
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

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest")
# global_vector_store = None

# Call the initialization function
# initialize_vector_store()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
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

# class PDFRetrievalInput(BaseModel):
#     """Input schema for the pdf_retrieval_tool."""
#     query: str = Field(description="The user's query to search for within the PDF document.")

# class WebSearchInput(BaseModel):
#     """Input schema for the tavily_search_tool."""
#     query: str = Field(description="A concise search query for the web.")

# class SQLQueryInput(BaseModel):
#     """Input schema for the sql_query_tool."""
#     query: str = Field(description="The natural language question to be converted into a SQL query.")

# ==========================
# 📊 State Management (Simplified and Centralized)
# ==========================
# class VisualizationInput(BaseModel):
#     """Input schema for the generate_visualization_tool."""
#     query: str = Field(description="The user's natural language request for a chart or visualization, e.g., 'Plot the total sales by region'.")

class State(MessagesState):
    """Manages the conversation state. Uses Pydantic models for structured data."""
    user_query: str
    attached_content: Optional[str]




    # Utility 
    next_node: Optional[str] 
    vector_store_paths: Optional[List[str]]
    tenant_config: Optional[Dict[str, Any]] # <-- NEW
    # vector_store:Optional[Document[str, Any]]
    # vector_store_path: Optional[Chroma]
    

    pdf_content: Optional[str] 
    web_content: Optional[str] 
    sql_result: Optional[str]
    # visualization_result: Optional[Dict[str, Any]] = None # <-- NEW
 

    
    final_answer: Optional[Answer]
    conversation_summary: Optional[Summary]

    # For final logging
    metadatas: Optional[Dict[str, Any]] 

    
    
 
    # tools_list: Optional[List[Tool]] 

# ==========================
# 🛠️ Tools
# ==========================

def get_time_based_greeting():
    """Return an appropriate greeting based on the current time."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12: return "Good morning"
    if 12 <= current_hour < 17: return "Good afternoon"
    return "Good evening"


def search_pdf(state: State): 
# def retrieve_from_pdf(query: str, state: State) -> dict: 
    """Performs a document query using the bound vector store."""
    # vector_store = state["vector_store"] 
    
    vector_store_path = state["tenant_config"]["vector_store_path"]
    # vector_store_path = state["vector_store_paths"]

    print(vector_store_path)
    if not os.path.exists(vector_store_path):
        print(f"Vector store path not found for tenant: {vector_store_path}")

        return {"pdf_content": "Error: Vector store path not found."}
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embeddings
    )

    if vector_store:
        try:
            ayula=state["messages"][-1].content
            attached_content=state["attached_content"]
            user_input = f"User Query:\n{ayula}\n\n:Attached File Content:\n{attached_content}"
            results = vector_store.similarity_search(user_input, k=3)
            content = "\n\n".join([doc.page_content for doc in results])
            return {"pdf_content": content}
        except Exception as e:
            # Added better error logging here
            # print(f"Error during PDF search (Chroma): {e}")
            print(f"Error during PDF search for tenant at {vector_store_path}: {e}")
            return {"pdf_content": f"Error: Failed to retrieve from PDF: {e}"}
            
    return {"pdf_content": "Error: Document knowledge base not initialized."}

# NOTE: pdf_retrieval_tool definition is intentionally REMOVED from global scope 
# and created dynamically in process_message.
# pdf_retrieval_tool = Tool(
#     name="pdf_retrieval_tool",
#     description="Useful for answering questions from the bank's internal knowledge base (PDFs). Input should be a specific question.",
#     func=retrieve_from_pdf,
#     args_schema=PDFRetrievalInput,
# )


def search_web(state: State):  
        """Perform web search"""
        
        try:
            
            ayula=state["messages"][-1].content
            attached_content=state["attached_content"]
            user_input = f"User Query:\n{ayula}\n\n:Attached File Content:\n{attached_content}"
           
            search = TavilySearch(max_results=2)
            search_docs = search.invoke(input=user_input)
            # search_docs = tavily_search.invoke(user_input)
            # print ("Web: Response Type:", search_docs)  # Debug print
            
            if any(error in str(search_docs) for error in ["ConnectionError", "HTTPSConnectionPool"]):
                return {"web_content": ""}
                
            formatted_docs = "\n\n---\n\n".join(
                f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
                for doc in search_docs['results']
            )
            return {"web_content": formatted_docs}
        except Exception as e:
            print(f"Web search error: {e}")
            return {"web_content": ""}


pdf_retrieval_tool = Tool(
    name="search_pdf",
    description="Useful for answering questions that require information from the tenant's internal knowledge base (PDFs). Always use this tool first for tenant-specific queries. Input should be the user's question.",
    func=search_pdf,
)

# 2. Define the Web Search tool (bound to the search_web function)
web_search_tool = Tool(
    name="search_web",
    description="Useful for answering general or external knowledge questions that are NOT found in the tenant's internal PDF. Input should be a concise search query.",
    func=search_web,
)

# 3. Create the tools list
local_tools = [pdf_retrieval_tool, web_search_tool]

# Note: We'll pass these tools into the new agent node function.
# The graph compilation part below remains the same for now, but we'll update build_graph.
# We'll modify build_graph to accept this list
# ...
# ==========================
# 🛠️ agent_node function (FIXED)
# ==========================
# 🚨 FIX 2: Agent node now accepts the tools list directly as an argument.
# def agent_node(state: State, tools_for_llm: List[Tool]): 

# Replace your current agent_node function with this:

def run_agent(state: State, tools: List[Tool]):
    """
    The Agent Node: Uses the LLM to decide whether to call a tool or generate a final answer.
    """
    print("--- LLM AGENT NODE (Decision Maker) ---")
    messages = state["messages"]
    
    # Check for initial greeting scenario
    if len(messages) == 1:
        # Check if the message is the first one in the conversation
        if not state.get("summarization_request") == "true":
             tenant_config = state["tenant_config"] 
             greeting = tenant_config.get("chatbot_greeting", "Hello! How can I help you?") 
             new_messages = messages + [AIMessage(content=f"{get_time_based_greeting()}! {greeting}")]
             # Add a flag to the state to skip further processing
             return {"messages": new_messages, "next_node": "GREETING_SENT"} 

    
    # Get the system prompt from tenant config for the agent
    # We'll use a strong default if it's missing
    # agent_prompt_template = state["tenant_config"].get("agent_prompt", 
    #     "You are a helpful AI assistant. You have access to a web search and a PDF knowledge base. Use these tools to gather information before generating a final, confident answer. If you have enough information, respond directly without calling any tool."
    # )
    agent_prompt_template = "You are a helpful AI assistant. You have access to a web search and a PDF knowledge base. Use these tools to gather information before generating a final, confident answer. If you have enough information, respond directly without calling any tool."

    
    # Bind the tools and the system prompt to the LLM
    agent_llm = llm.bind_tools(tools=tools).with_config(
        {"tags": ["agent_decision_maker"], "system_message": agent_prompt_template}
    )

    # Invoke the LLM with the message history
    response = agent_llm.invoke(messages)
    
    # Update the state with the LLM's response (which may contain tool_calls)
    return {"messages": messages + [response]}



def conditional_rule(state: State) -> str:
    """Routes the flow based on tool calls or the special greeting flag."""
    # 🚨 FIX: Handles the 'GREETING_SENT' flag
    if state.get("next_node") == "END":
        return END
    
    elif state.get("summary_request") == "true":
        return "generate_summary"

    return "prepare_answer"

def make_call(state: State):
   pass


def generate_final_answer_node(state: State):
    """
    The Generator Node: Creates the final structured answer after gathering all necessary context from tools.
    """
    print("--- GENERATE FINAL ANSWER NODE ---")
    user_query = state.get("user_query")
    # user_query = state["messages"][-1].content
  
    messages = state["messages"] 
    tenant_config = state["tenant_config"] 

    # Find the last HumanMessage
    # user_query = " "
    # for msg in reversed(messages):
    #     if isinstance(msg, HumanMessage):
    #         user_query = msg.content
    #         break
            
    if not user_query:
        user_query = "The user asked an unrecoverable question."
    print("Aleu",user_query)
  

    
    context_parts = []
    if state.get("pdf_content"): context_parts.append(f"PDF Content:\n{state['pdf_content']}")
    if state.get("web_content"): context_parts.append(f"Web Content:\n{state['web_content']}")
    if state.get("attached_content"): context_parts.append(f"Attached Content:\n{state['attached_content']}")
    context = "\n\n".join(context_parts) if context_parts else "No additional context was retrieved."


    # 🚨 FIX: Ensure prompt is defined and safely formatted
 
    

    prompt_template="""You are Damilola, the AI-powered virtual assistant for ATB. Your role is to deliver professional customer service and insightful data analysis, depending on the user's needs.

You operate in two modes:
1. **Customer Support**: Respond with empathy, clarity, and professionalism. Your goal is to resolve issues, answer questions, and guide users to helpful resources — without technical jargon or internal system references.
2. **Data Analyst**: Interpret data, explain trends, and offer actionable insights. When visualizations are included, describe what the chart shows and what it means for the user.

Your response must be:
- **Final**: No follow-up questions or uncertainty.
- **Clear and Polite**: Use emotionally intelligent language, especially if the user expresses frustration or confusion.
- **Context-Aware**: Avoid mentioning internal systems (e.g., database names or SQL sources) unless explicitly requested.
- **Structured**: Always return your answer in the following JSON format.

User Question:
{user_query}

Available Context:
---
{context}
---

If the context includes 'Visualization Analysis', describe the chart’s content and implications.

Format your response as a JSON object using this schema (omit 'chart_base64'):

Schema:
{{
  "answer": "str: Your clear, concise, and polite response.",
  "sentiment": "int: An integer rating of the user's sentiment (-2 to +2).",
  "ticket": "List[str]: Relevant service channels (e.g., 'email', 'live chat', 'support portal'). Empty list if not applicable.",
  "source": "List[str]: Sources used to generate the answer. Empty list if not applicable."
}}
   """
    


    
    try:
        # Format the final prompt
        prompt = prompt_template.format(user_query, context)
        prompt = prompt_template
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

def summarize_conversation(state: State):
    print("--- SUMMARIZE CONVERSATION NODE ---")
    
    messages = state.get("messages", [])
    conversation_id=state.get("conversation_id", "")

    tenant_config = state["tenant_config"] 

    if not messages:
        return {
            "conversation_summary": None,
            "metadatas": {"error": "No messages to summarize."}
        }

    # conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    messages = state["messages"]
    conversation= Conversation.objects.get(conversation_id=conversation_id)
    conversation_history=conversation.summary
    len_hist=conversation.message_count
    rec=len(messages) -len_hist
    recent_message =state["messages"][-rec]



    summarize_prompt_template1=f"""Please provide a structured summary of the following conversation.
    
    Summary of Previous Conversation:
    {conversation_history}
    Recent Message:{recent_message}

    Provide the output in a JSON format matching this schema:
    {{
        "summary": "A concise summary of the entire conversation.",
        "sentiment": "Overall sentiment score of the conversation (-2 to +2).",
        "unresolved_tickets": ["List of channels with unresolved issues."],
        "all_sources": ["All unique sources referenced in the conversation."]
    }}"""




    

    # 🚨 FIX: Provide a safe, format-ready default template
    summarize_prompt_template = tenant_config.get(
        "summary_prompt", 
        "Summarize the conversation below, determining sentiment and any unresolved issues, using only the structured output schema:\n\nConversation:\n{0}"
    ) 
    try:
        summarize_prompt = summarize_prompt_template.format(conversation_history)
        # summarize_prompt=summarize_prompt_template
  
    except Exception as e:
        print(f"ERROR: Summary prompt format failed: {e}. Using raw history.")
        summarize_prompt = "Summarize the following raw history: " + conversation_history
    
    # ... (rest of the summarize node logic)

    structured_llm = llm.with_structured_output(Summary)
    summary_obj = structured_llm.invoke(summarize_prompt)
    
    # Create the final metadata dictionary for logging

    final_answer = state.get("final_answer")
    user_query = state.get("user_query")
    # for msg in reversed(messages):
    #     if isinstance(msg, HumanMessage):
    #         last_user_question = msg.content
    #         break
    conversation.message_count=len(messages) 
    conversation.save()
    metadata_dict = {
        "question": user_query, # Use the robustly found question
        "answer": final_answer.answer if final_answer else "N/A",
        "sentiment": final_answer.sentiment if final_answer else 0,
        "ticket": final_answer.ticket if final_answer else [],
        "source": final_answer.source if final_answer else [],
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
# Define the condition to check if a tool was called
# def should_continue(state: State) -> str:
#     """Determines the next step: call a tool or generate the final answer."""
#     if state["messages"][-1].tool_calls:
#         return "tools"
#     return "generate_final_answer"

# ==========================
# 🔄 Graph Workflow
# ==========================

# def build_graph(tools_list): # 🚨 FIX: Accepts the dynamic tools list


# Main processing function
# def process_message(message_content: str, session_id: str, tenant_id: str, file_path: Optional[str] = None):

def build_graph(tools: List[Tool]): 
    """Builds and compiles the LangGraph workflow."""
    
    # 1. Define the ToolNode for execution (takes the tools list)
    tool_node = ToolNode(tools=tools)

    workflow = StateGraph(State)
    
    # 2. Add the nodes
    # 'run_agent' is the new agent decision-maker
    workflow.add_node("run_agent", partial(run_agent, tools=tools)) 
    
    # 'tools' is the LangGraph ToolNode executor
    workflow.add_node("tools", tool_node) 
    
    # Tool execution nodes (these run inside the ToolNode) are now removed from nodes list: 
    # workflow.add_node("search_web", search_web)
    # workflow.add_node("search_pdf", search_pdf)

    workflow.add_node("generate_final_answer", generate_final_answer_node)
    workflow.add_node("summarize", summarize_conversation)

    # 3. Define the Entry Point
    workflow.set_entry_point("run_agent")

    # 4. Define the Edges

    # Custom function to route from the initial decision maker
    def route_agent_decision(state: State) -> str:
        """Determines the next step after the LLM agent runs."""
        if state.get("next_node") == "GREETING_SENT":
            return END
        
        if state.get("summarization_request") == "true":
            return "summarize"

        # Check for tool calls made by the LLM (ReAct loop)
        if state["messages"][-1].tool_calls:
            # LLM decided to call a tool
            return "tools"
        
        # LLM decided to answer directly (No tool call)
        return "generate_final_answer"


    # Decision from the Agent Node routes to Tool Execution, Summary, or Final Answer
    workflow.add_conditional_edges(
        "run_agent", 
        route_agent_decision,
        {
            "tools": "tools",  # Route to the ToolNode for execution
            "generate_final_answer": "generate_final_answer",
            "summarize": "summarize",
            END: END,
        }
    )

    # Tool Execution routes back to the Agent Node to decide what to do with the tool results
    workflow.add_edge("tools", "run_agent")
    
    # Final states
    workflow.add_edge("generate_final_answer", END)
    workflow.add_edge("summarize", END)
    
    # ... (Rest of the checkpointing logic remains the same)
    
    # Initialize checkpointing with a robust fallback
    # ... (Keep existing checkpointing code here)

    memory = None # Placeholder for memory initialization
    # ... (Existing SQLite/InMemory setup)
    try:
        # Create a path for the checkpointing database
        checkpoint_dir = os.path.dirname(settings.DATABASES['default']['NAME'])
        checkpoint_db = os.path.join(checkpoint_dir, 'checkpoints.sqlite')
        # ... (Rest of SQLite setup)
        conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
        memory = SqliteSaver(conn=conn)
        print("SQLite checkpointing connected successfully.")
    except Exception as e:
        print(f"Error connecting to SQLite for checkpointing: {e}. Using in-memory saver.")
        memory = InMemorySaver()

    return workflow.compile(checkpointer=memory)






























def process_message(message_content: str, conversation_id: str, tenant_id: str, file_path: Optional[str] = None,summarization_request: Optional[str] = None):

    """Main function to process user messages using the LangGraph agent."""

    # --- DYNAMIC INITIALIZATION ---
    try:
        current_tenant = Tenant.objects.get(tenant_id=tenant_id)
    except Tenant.DoesNotExist:
        return {"answer": "Error: Tenant configuration not found.", "chart": None, "metadata": {}}

    # ⚠️ Initialize the tenant-specific, persistent vector store
    tenant_vector_store = initialize_vector_store(tenant_id)
    user_query =message_content


    # 1. Create the dynamic pdf_retrieval_tool bound to the tenant's vector store
    # --- DYNAMIC INITIALIZATION (NEW) ---
    # 1. Define the PDF retrieval tool (bound to the search_pdf function)
    pdf_retrieval_tool = Tool(
        name="search_pdf",
        description="Useful for answering questions that require information from the tenant's internal knowledge base (PDFs). Always use this tool first for tenant-specific queries. Input should be the user's question.",
        # func=search_pdf, # The func takes 'state', so we don't need args_schema or Pydantic input here.
    )

    # 2. Define the Web Search tool (bound to the the search_web function)
    web_search_tool = Tool(
        name="search_web",
        description="Useful for answering general or external knowledge questions that are NOT found in the tenant's internal PDF. Input should be a concise search query.",
        # func=search_web,
    )
    
    # 3. Create the tools list
    local_tools = [pdf_retrieval_tool, web_search_tool] 

    # 4. Build the graph with the dynamic tools
    graph = build_graph(local_tools) # Pass the tools list
   
   

   
    # # 3. Create the final tools list to pass to the graph builder
    # local_tools = [pdf_retrieval_tool_dynamic, tavily_search_tool]
     # 🚨 FIX 4A: Bind the tools list to the agent_node function
    # bound_agent_node = partial(agent_node, tools_for_llm=local_tools)
     # 🚨 FIX 4B: Pass BOTH the tool list and the bound node to build_graph
    # graph = build_graph(local_tools, bound_agent_node) 
    # graph = build_graph()

    # graph = build_graph(local_tools) # 🚨 FIX: Pass the dynamic tools list

    config = {"configurable": {"thread_id": conversation_id}}
    
    attached_content = None # Simplified for this example
    summarization_request =None
    if summarization_request:
        summarization_request="true"
    else:
        summarization_request="false"



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
                # response = llm.invoke([message])
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
        "user_query": user_query,
        "summarization_request": "true" if summarization_request else "false", # Ensure string value
        "conversation_id":conversation_id,
        "tenant_config": {
            "greeting": current_tenant.chatbot_greeting,
            "agent_prompt": current_tenant.agent_node_prompt,
            "final_prompt": current_tenant.final_answer_prompt,
            "summary_prompt": current_tenant.summary_prompt,
             "vector_store_path": os.path.join(settings.BASE_DIR, "chroma_dbs", tenant_id),



            
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
    
