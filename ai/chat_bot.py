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
from typing import Any, Dict, List, Literal, Optional, Union
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
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
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
from pydantic import BaseModel, Field, ValidationError
from langchain.vectorstores.base import VectorStore


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
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Removed any reference to a 'file' handler to avoid configuration errors.


# logger = logging.getLogger(__name__)
logger = logging.getLogger("ayodele")


def log_info(msg, tenant_id, conversation_id):
    logger.info(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_error(msg, tenant_id, conversation_id):
    logger.error(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_debug(msg, tenant_id, conversation_id):
    logger.debug(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_warning(msg, tenant_id, conversation_id):
    logger.warning(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_exception(e, tenant_id, conversation_id):
    import traceback
    tb = traceback.format_exc()
    logger.error(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] Exception: {e}\n{tb}")

from langchain.tools import StructuredTool

# log_info("LangGraph execution started", tenant_id, conversation_id)
# log_error("Tool failed to return results", tenant_id, conversation_id)
# log_exception(e, tenant_id, conversation_id)

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



from langchain_community.vectorstores import FAISS

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
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Assuming 'settings', 'embeddings', and 'Tenant' are defined elsewhere

import os
# Assuming necessary imports like Chroma, PyPDFLoader, RecursiveCharacterTextSplitter, etc., are here
import os
from langchain_community.vectorstores import FAISS
# Assuming these imports are defined elsewhere in your file:
# from your_project.settings import settings
# from your_project.models import Tenant
# from your_project.llm_setup import embeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

def initialize_vector_store(tenant_id: str):
    """Initializes and connects to the FAISS vector store, loading from disk if available."""

    # 1. Test embedding model (Keep this)
    try:
        test_emb = embeddings.embed_query("Test embedding")
        print("Test embedding length:", len(test_emb))
    except Exception as e:
        print(f"Embedding model failed: {e}")
        return None 

    # --- Setup Persistence Path and File Checks ---
    persist_directory = os.path.join(settings.BASE_DIR, "faiss_dbs", tenant_id) # ⚠️ Renamed for clarity
    faiss_index_path = os.path.join(persist_directory, "index.faiss")
    
    
    # 2. LOAD FROM DISK CHECK (Persistence)
    if os.path.exists(faiss_index_path):
        try:
            print(f"✅ Loading existing FAISS index from disk: {persist_directory}")
            vector_store = FAISS.load_local(
                folder_path=persist_directory, 
                embeddings=embeddings,
                # Required for loading FAISS indices saved from disk
                allow_dangerous_deserialization=True 
            )
            return vector_store
        except Exception as e:
            # Handle corrupted or incompatible index files
            print(f"⚠️ Warning: Failed to load existing FAISS index ({e}). Re-indexing...")
            # Fall through to the creation logic below

    # --- Document File Retrieval ---
    try:
        tenant = Tenant.objects.get(tenant_id=tenant_id)
        file_path = tenant.tenant_kss.path 
        print("File Path Ajadi:", file_path)
    except (Tenant.DoesNotExist, AttributeError):
        print("Tenant profile not found or missing path. Returning empty store.")
        # Return an empty FAISS index
        return FAISS.from_texts([""], embeddings) 

    # --- Create Index (If Load Failed or Index Didn't Exist) ---
    if file_path and os.path.exists(file_path):
        try:
            # --- Document Loading and Splitting ---
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            )

            valid_docs = [doc for doc in docs if doc.page_content.strip()]
            print(f"Loaded chunks (total): {len(docs)}")
            print(f"Valid chunks to embed: {len(valid_docs)}")

            if valid_docs:
                # 3. CREATE FAISS INDEX (Initial creation)
                vector_store = FAISS.from_documents(
                    documents=valid_docs,
                    embedding=embeddings
                )
                print(f"PDF loaded and indexed in FAISS (in-memory) for tenant {tenant_id}.")
                
                # 4. 🔥 CRITICAL STEP: SAVE TO DISK
                os.makedirs(persist_directory, exist_ok=True) # Ensure directory exists
                vector_store.save_local(
                    folder_path=persist_directory,
                    index_name="index" # Saves as index.faiss and index.pkl
                )
                print(f"✅ FAISS index successfully saved to disk at {persist_directory}")
                
                return vector_store
            else:
                print("No valid chunks found to embed.")
        
        except Exception as e:
            print(f"🔴 ERROR during document loading or indexing: {e}")
            
    # Fallback return: Create and return an empty FAISS index if loading/indexing failed
    print("Returning placeholder empty FAISS index.")
    return FAISS.from_texts([""], embeddings)


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

class Summary(BaseModel):
    """Conversation summary schema."""
    summary: str = Field(description="A concise summary of the entire conversation.")
    sentiment: int = Field(description="Overall sentiment of the conversation from -2 to +2.")
    unresolved_tickets: List[str] = Field(description="A list of channels with unresolved issues.")
    all_sources: List[str] = Field(description="All unique sources referenced throughout the conversation.")

class State(MessagesState):
    """Manages the conversation state. Uses Pydantic models for structured data."""

    user_query: str
    attached_content: Optional[str]

    # Core identifiers
    conversation_id: str
    tenant_id: str

    # Tenant configuration and vector store
    tenant_config: Optional[Dict[str, Any]]
    vector_store_path: Optional[str]  # ✅ ADD THIS LINE

    # vector_store: Optional[VectorStore]

    # Tool outputs
    pdf_content: Optional[str]
    web_content: Optional[str]
   
    # Final outputs
    final_answer: Optional[Answer]
    conversation_summary: Optional[Summary]
    metadatas: Optional[Dict[str, Any]]

    # Utility
  
    next_node: Optional[str]
    tool_usage_log: Optional[List[str]]  # Optional tracking
    llm_calls: int
# ==========================
# 🛠️ Tools
# ==========================
def log_tool_usage(state: State, tool_name: str):
    state["tool_usage_log"] = state.get("tool_usage_log") or []
    state["tool_usage_log"].append(tool_name)

def get_time_based_greeting():
    """Return an appropriate greeting based on the current time."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12: return "Good morning"
    if 12 <= current_hour < 17: return "Good afternoon"
    return "Good evening"
import json

import json

from langchain_community.vectorstores import FAISS # Import FAISS

def search_pdf(state: State):
    """Performs a document query using the pre-initialized FAISS vector store."""
    
    # Normalize input if it's a string
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            return {"pdf_content": "Error: Invalid input format for PDF search."}

    tenant_config = state.get("tenant_config", {})
    
    user_query = state.get("user_query", "unknown")
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    
    if not user_query:
        return {"pdf_content": "Error: No query provided for PDF search."}

    log_info("search_pdf tool invoked", tenant_id, conversation_id)

    # 🚨 NOTE: 'vector_store_path' now refers to the directory where FAISS index files are saved.
    vector_store_path = state.get("vector_store_path") 
    print(f"DEBUG: Search Path Loaded: {vector_store_path}")
    if not vector_store_path:
        log_error("Missing vector_store_path in state", tenant_id, conversation_id)
        return {"pdf_content": "Error: Vector store path not provided."}

    try:
        # 1. LOAD FAISS: Use FAISS.load_local() instead of instantiating Chroma
        vector_store = FAISS.load_local(
            folder_path=vector_store_path, 
            embeddings=embeddings,
            # This is CRITICAL for loading FAISS indexes saved from a file
            allow_dangerous_deserialization=True 
        )
        
        # 2. Check Count (FAISS doesn't expose a simple .count(), so we check a known attribute)
        search_count = vector_store.index.ntotal
        print(f"Vector Count loaded for search: {search_count}")

        if search_count == 0:
             return {"pdf_content": "Error: Vector store loaded successfully but is empty (0 documents)."}

        # 3. Perform Similarity Search (same method signature)
        results = vector_store.similarity_search(user_query, k=3)
        content = "\n\n".join([doc.page_content for doc in results])
        # print("Content", content)

        formatted_content = (
            "--- PDF DOCUMENT CONTEXT START ---\n"
            f"{content}\n"
            "--- PDF DOCUMENT CONTEXT END ---"
        )

        log_debug(f"PDF Search Results:\n{formatted_content}", tenant_id, conversation_id)
        return {"pdf_content": formatted_content}

    except Exception as e:
        # This will now catch file-not-found or corrupted index errors
        log_error(f"PDF search failed: {e}", tenant_id, conversation_id)
        # log_exception is assumed to be defined elsewhere for full exception logging
        # log_exception(e, tenant_id, conversation_id) 
        return {"pdf_content": f"Error: Failed to retrieve from PDF: {e}"}
    

    
def search_web(state: State): 
    """Perform web search using Tavily."""
    tenant_config = state.get("tenant_config", {})

    tenant_config = state.get("tenant_config", {})
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    

    log_info("search_web tool invoked", tenant_id, conversation_id)
    try:

        user_query = state["messages"][-1].content if "messages" in state else state.get("__arg1", "")
        attached_content = state.get("attached_content", "")
        query = f"User Query:\n{user_query}\n\nAttached File Content:\n{attached_content}"

        search = TavilySearch(max_results=2)
        search_docs = search.invoke(input=query)

        if isinstance(search_docs, str):
            try:
                search_docs = json.loads(search_docs)
            except json.JSONDecodeError:
                log_error("Web search response is not valid JSON", tenant_id, conversation_id)
                return {"web_content": "Error: Invalid JSON response from web search."}

        if not isinstance(search_docs, dict) or "results" not in search_docs:
            log_error("Web search response format is unexpected", tenant_id, conversation_id)
            return {"web_content": "Error: Unexpected response format from web search."}

        formatted_docs = "\n\n---\n\n".join(
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs['results']
        )
        log_tool_usage(state, "search_web")
        log_debug(f"Web Search Results:\n{formatted_docs}", tenant_id, conversation_id)
        return {"web_content": formatted_docs}

    except Exception as e:
        log_error(f"Web search exception: {e}", tenant_id, conversation_id)
        return {"web_content": f"Error: Web search failed: {e}"}


def landing(state: State) -> str:
    tenant_id = state["tenant_config"].get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    log_info("Landing node activated", tenant_id, conversation_id)

    if state.get("summarization_request") == "true":
        log_info("Routing to summarize", tenant_id, conversation_id)
        return {"next_node":"summarize"}
    
    log_info("Routing to rag", tenant_id, conversation_id)
    return  {"next_node":"rag"}
    
def decide(state: State) -> str:
    tenant_id = state["tenant_config"].get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    log_info("Decide node activated", tenant_id, conversation_id)

    if state.get("next_node") == "tsummarizerue":
        log_info("Routing to summarize", tenant_id, conversation_id)
        return "summarize"
    
    log_info("Routing to rag", tenant_id, conversation_id)
    return "rag"

def rag(state: State) -> State:
    tenant_id = state["tenant_config"].get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    log_info("RAG node activated", tenant_id, conversation_id)
    log_debug(f"State entering RAG: {state}", tenant_id, conversation_id)
    return state  # ✅ Return the full state dictionary


def generate_final_answer(state: State) -> dict:
    """Generates the final structured answer using retrieved context and user query."""
    tenant_config = state["tenant_config"]
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")

    log_info("Final answer node activated", tenant_id, conversation_id)

    user_query = state.get("user_query")
    if not user_query:
        log_warning("Missing user_query — falling back to last human message", tenant_id, conversation_id)
        user_query = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "The user asked an unrecoverable question.")

    log_debug(f"User query: {user_query}", tenant_id, conversation_id)

    context_parts = []
    for key, label in [("pdf_content", "PDF Content"), ("web_content", "Web Content"), ("attached_content", "Attached Content")]:
        if state.get(key):
            context_parts.append(f"{label}:\n{state[key]}")
    context = "\n\n".join(context_parts) if context_parts else "No additional context was retrieved."
   
    log_debug(f"Context used in prompt:\n{context}", tenant_id, conversation_id)

    default_prompt = (
        "You are a polite, professional AI assistant. Respond to the user's question with the provided context.\n"
        "User Question: {user_query}\nContext: {context}\nReturn the answer in the required JSON schema."
    )
    prompt_template = tenant_config.get("final_answer_prompt", default_prompt)

    try:
        prompt = prompt_template.format(user_query=user_query, context=context)
    except Exception as e:
        log_error(f"Prompt formatting failed: {e}", tenant_id, conversation_id)
        prompt = default_prompt.format(user_query=user_query, context=context)

    structured_llm = llm.with_structured_output(Answer)
    final_answer_obj = structured_llm.invoke(prompt)
    print ("Akdkdk",final_answer_obj)

    new_messages = state["messages"] + [AIMessage(content=final_answer_obj.answer)]

    return {
        "final_answer": final_answer_obj,
        "messages": new_messages
    }


def summarize_conversation(state: State):
    tenant_config = state["tenant_config"]
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")

    log_info("Summarize node activated", tenant_id, conversation_id)

    messages = state.get("messages", [])
    if not messages:
        log_warning("No messages to summarize", tenant_id, conversation_id)
        return {
            "conversation_summary": None,
            "metadatas": {"error": "No messages to summarize."}
        }

    try:
        conversation = Conversation.objects.get(conversation_id=conversation_id)
    except Conversation.DoesNotExist:
        log_error("Conversation not found", tenant_id, conversation_id)
        return {
            "conversation_summary": None,
            "metadatas": {"error": "Conversation not found."}
        }

    conversation_history = conversation.summary
    len_hist = conversation.message_count
    rec = len(messages) - len_hist

    if rec > 0 and rec < len(messages):
        recent_message = messages[-rec]
    else:
        recent_message = messages[-1]
        log_warning("Fallback to last message due to invalid rec value", tenant_id, conversation_id)

    summarize_prompt_template = tenant_config.get(
        "summary_prompt",
        "Summarize the conversation below, determining sentiment and any unresolved issues, using only the structured output schema:\n\nConversation:\n{0}"
    )

    try:
        summarize_prompt = summarize_prompt_template.format(conversation_history)
    except Exception as e:
        log_error(f"Summary prompt format failed: {e}. Using raw history.", tenant_id, conversation_id)
        summarize_prompt = "Summarize the following raw history: " + conversation_history

    log_debug(f"Summarization prompt:\n{summarize_prompt}", tenant_id, conversation_id)

    structured_llm = llm.with_structured_output(Summary)
    summary_obj = structured_llm.invoke(summarize_prompt)

    log_debug(f"Summary result:\n{summary_obj}", tenant_id, conversation_id)

    final_answer = state.get("final_answer")
    user_query = state.get("user_query")

    conversation.message_count = len(messages)
    conversation.save()

    # Helper to safely extract attributes from dicts or BaseModel-like objects
    def _get_field(obj, field, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(field, default)
        # Try attribute access first, then dict() if available
        value = getattr(obj, field, None)
        if value is not None:
            return value
        try:
            return obj.dict().get(field, default)
        except Exception:
            return default

    metadata_dict = {
        "question": user_query,
        "answer": _get_field(final_answer, "answer", "N/A"),
        "sentiment": _get_field(final_answer, "sentiment", 0),
        "ticket": _get_field(final_answer, "ticket", []),
        "source": _get_field(final_answer, "source", []),
        "summary": _get_field(summary_obj, "summary", None),
        "summary_sentiment": _get_field(summary_obj, "sentiment", None),
        "summary_unresolved_tickets": _get_field(summary_obj, "unresolved_tickets", None),
        "summary_sources": _get_field(summary_obj, "all_sources", None),
    }

    return {
        "conversation_summary": summary_obj,
        "metadatas": metadata_dict
    }


def build_graph(tenant_id: str, conversation_id: str):
    """Builds and compiles the LangGraph workflow for a given tenant and conversation."""

    workflow = StateGraph(State)

    # --- Nodes ---
    workflow.add_node("landing", landing)
    workflow.add_node("decide", decide)
    workflow.add_node("rag", rag)
    workflow.add_node("search_pdf", search_pdf)
    workflow.add_node("search_web", search_web)
    workflow.add_node("generate_final_answer", generate_final_answer)
    workflow.add_node("summarize", summarize_conversation)

    # --- Routing ---
    workflow.add_edge(START, "landing")
    workflow.add_conditional_edges("landing", decide, ["summarize", "rag"])
    workflow.add_edge("rag", "search_pdf")
    workflow.add_edge("rag", "search_web")
    workflow.add_edge("search_pdf", "generate_final_answer")
    workflow.add_edge("search_web", "generate_final_answer")
    workflow.add_edge("generate_final_answer", END)
    workflow.add_edge("summarize", END)

    # --- Checkpointing ---


    # --- Checkpointing Setup ---
    memory = None
    try:
        checkpoint_dir = os.path.dirname(settings.DATABASES['default']['NAME'])
        checkpoint_db = os.path.join(checkpoint_dir, 'checkpoints.sqlite')
        conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
        memory = SqliteSaver(conn=conn)
        print("SQLite checkpointing connected successfully.")
    except Exception as e:
        print(f"Error connecting to SQLite for checkpointing: {e}. Using in-memory saver.")
        memory = InMemorySaver()

    log_info("LangGraph workflow compiled successfully", tenant_id, conversation_id)
    return workflow.compile(checkpointer=memory)


def process_message(
    message_content: str,
    conversation_id: str,
    tenant_id: str,
    file_path: Optional[str] = None,
    summarization_request: Optional[str] = None
):
    """Main function to process user messages using the LangGraph agent."""

    log_info("Processing message", tenant_id, conversation_id)

    # --- Tenant Configuration ---
    try:
        current_tenant = Tenant.objects.get(tenant_id=tenant_id)
    except Tenant.DoesNotExist:
        log_error("Tenant configuration not found", tenant_id, conversation_id)
        return {
            "answer": "Error: Tenant configuration not found.",
            "chart": None,
            "metadata": {}
        }

    # --- Initialization ---
    persist_directory = os.path.join(settings.BASE_DIR, "faiss_dbs", tenant_id)
    tenant_vector_store = initialize_vector_store(tenant_id)
    print("Vector store initialized.")
    if tenant_vector_store is not None:

        # Use the correct attribute for FAISS count
        try:
            document_count = tenant_vector_store.index.ntotal
            print("Document count:", document_count)
        except AttributeError:
            # Fallback if the object structure is unexpected
            print("Error: Could not determine document count for FAISS.")
    else:
        print("Vector store initialization failed; document count unavailable.")

    
    user_query = message_content
    summarization_flag = str(summarization_request).lower() in ['true', '1']

    # --- Image Processing ---
    attached_content = None
    if file_path:
        try:
            image = Image.open(file_path)
            image.thumbnail((512, 512))

            image_format = image.format.upper() if image.format else None
            if image_format not in ["PNG", "JPEG", "JPG"]:
                raise ValueError(f"Unsupported image format: {image_format}")

            mime_type = "jpeg" if image_format in ["JPEG", "JPG"] else "png"
            buffered = io.BytesIO()
            image.save(buffered, format=image_format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_uri = f"data:image/{mime_type};base64,{img_str}"

            if GOOGLE_API_KEY:
                os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            else:
                raise ValueError("GOOGLE_API_KEY is not set in environment variables")

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
            prompt = "Generate the message in the content of the picture."
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_uri}}
            ])
            response = llm.invoke([message])
            attached_content = response.content
            log_debug(f"Attached content from image: {attached_content}", tenant_id, conversation_id)

        except Exception as e:
            log_error(f"Error processing image attachment: {e}", tenant_id, conversation_id)
            attached_content = f"Error: Could not process attached file ({e})"

    # --- Tenant Config Dictionary ---
    tenant_config_dict = {
        "tenant_id": tenant_id,
        "vector_store_path": persist_directory,
        "chatbot_greeting": current_tenant.chatbot_greeting,
        "agent_node_prompt": current_tenant.agent_node_prompt,
        "final_answer_prompt": current_tenant.final_answer_prompt,
        "summary_prompt": current_tenant.summary_prompt,
      
    }
    # --- Initial State ---
    initial_state = {
    "messages": [HumanMessage(content=message_content)],
    "attached_content": attached_content,
    "user_query": user_query,
    "summarization_request": "true" if summarization_flag else "false",
    "conversation_id": conversation_id,
    "tenant_config": tenant_config_dict,
    "vector_store_path": persist_directory,  # ✅ FIXED
}

    try:
        initial_state_obj = State(**initial_state)
    except ValidationError as e:
        log_error(f"State validation failed: {e}", tenant_id, conversation_id)
        raise

    # --- Graph Execution ---
    graph = build_graph( tenant_id, conversation_id)
    config = {"configurable": {"thread_id": conversation_id}} # Replace with dynamic thread ID if needed

    try:
        output = graph.invoke(initial_state_obj, config=config)
        log_info("LangGraph execution completed successfully", tenant_id, conversation_id)
    except Exception as e:
        log_error(f"LangGraph execution failed: {e}", tenant_id, conversation_id)
        raise

    # --- Final Response ---
    final_answer_obj = output.get('final_answer')
    if final_answer_obj:
        return {
            "answer": final_answer_obj.answer,
            "metadata": output.get("metadatas", {})
        }
    else:
        last_message = output.get("messages", [AIMessage(content="Internal error.")])[-1]
        fallback = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
        log_warning("No final answer object returned. Using fallback message.", tenant_id, conversation_id)
        return {
            "answer": fallback,
            "chart": None,
            "metadata": {}
        }