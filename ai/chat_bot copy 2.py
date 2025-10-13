# # # ==========================
# # # 🌐 Django & Project Settings (Commented out as not used in standalone script)
# # # ==========================
# from django.conf import settings
# from .models import Prompt,Prompt7

import base64
from collections import UserDict
import io
import json
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

import matplotlib.pyplot as plt
# # ==========================
import pandas as pd
from django.conf import settings
from .views import get_tenant
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
from langchain_core.vectorstores import InMemoryVectorStore
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
from matplotlib.ticker import FuncFormatter
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
import matplotlib

matplotlib.use('Agg') # This prevents Matplotlib from trying to open a GUI window

import matplotlib

# This must be done BEFORE importing pyplot
matplotlib.use('Agg')

import base64
import io
import logging
import re
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sqlalchemy import create_engine
from .models import Tenant
from .views import get_tenant



from langchain_openai import OpenAIEmbeddings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ==========================
# ⚙️ Configuration & Initialization
# ==========================
# Load API keys from environment variables for security
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyBV7_Cbak1LhE2bHK_aG4ARaa6anxjBClY"


# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
# LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
# LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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

# Initialize LLM
# User specified ChatGroq with deepseek model


# py = Prompt7.objects.get(pk=1)  # Get the existing record
# google_model = py.google_model


# chatbot_model =py.chatbot_model
google_model="gemini-2.0-flash"
# google_model="gemini-2.5-pro"

# google_model = "gemini-2.5-flash",
# chatbot_model="gemini"



# if chatbot_model =="gpt":
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
# elif chatbot_model == "deepseek":
#     llm = ChatDeepSeek(model="deepseek-chat", temperature=0, max_tokens=None, timeout=None, max_retries=2)
# elif chatbot_model == "gemini":
#     llm = init_chat_model("google_genai:gemini-2.0-flash")
# elif chatbot_model == "groq":
#     llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, max_tokens=None, timeout=None, max_retries=2)



llm = init_chat_model("google_genai:gemini-flash-latest")

model = llm # Consistent naming

tenant_id =get_tenant()
tenant = Tenant.objects.get(tenant_id=tenant_id)  # Get the existing record


# Initialize Embeddings and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest")
global_vector_store = None



def initialize_vector_store():
    """Initializes the vector store by loading and splitting the PDF document."""
    global global_vector_store
    if global_vector_store is None:
        global_vector_store = InMemoryVectorStore(embedding=embeddings)

        try:
            # tenant = Tenant.objects.get(tenant_id=tenant_id)
            file_path = tenant.tenant_profile.path  # Get actual file path from FileField
        except Tenant.DoesNotExist:
            print("Tenant with the given profile does not exist.")
            return
        except Exception as e:
            print(f"Error retrieving tenant profile path: {e}")
            return

        if file_path and os.path.exists(file_path):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    add_start_index=True
                )
                all_splits = text_splitter.split_documents(docs)
                global_vector_store.add_documents(documents=all_splits)
                print("PDF document loaded and processed successfully.")
            except Exception as e:
                print(f"Error loading PDF: {e}. PDF retrieval tool will not work.")
        else:
            print(f"Warning: PDF file not found at {file_path}. PDF retrieval tool will not work.")
# Call the initialization function
initialize_vector_store()

# Initialize SQLDatabase with the specified file path
# DB_FILE_PATH = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\db.sqlite3"
# DB_FILE_PATH = f"sqlite:///{settings.DATABASES['default']['NAME']}"
# DB_URI = f"sqlite:///{DB_FILE_PATH}" 
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
class VisualizationInput(BaseModel):
    """Input schema for the generate_visualization_tool."""
    query: str = Field(description="The user's natural language request for a chart or visualization, e.g., 'Plot the total sales by region'.")

class State(MessagesState):
    """Manages the conversation state. Uses Pydantic models for structured data."""
    # Tool outputs
    pdf_content: Optional[str] = None
    web_content: Optional[str] = None
    sql_result: Optional[str] = None
    visualization_result: Optional[Dict[str, Any]] = None # <-- NEW

    attached_content: Optional[str] = None
    last_tool_name: Optional[str] = Field(default=None)

    # Final structured outputs
    final_answer: Optional[Answer] = None
    conversation_summary: Optional[Summary] = None

    # For final logging
    metadatas: Optional[Dict[str, Any]] = None

# ==========================
# 🛠️ Tools
# ==========================

def get_time_based_greeting():
    """Return an appropriate greeting based on the current time."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12: return "Good morning"
    if 12 <= current_hour < 17: return "Good afternoon"
    return "Good evening"
def retrieve_from_pdf(query: str) -> str:
    """Performs a document query using the initialized vector store."""
    if global_vector_store:
        results = global_vector_store.similarity_search(query, k=3)
        print ("Yelo",results )
        content = "\n\n".join([doc.page_content for doc in results])
        return {"pdf_content": content}
    return {"pdf_content": "Error: Document knowledge base not initialized."}

pdf_retrieval_tool = Tool(
    name="pdf_retrieval_tool",
    description="Useful for answering questions from the bank's internal knowledge base (PDFs). Input should be a specific question.",
    func=retrieve_from_pdf,
    args_schema=PDFRetrievalInput,
)

def search_web_func(query: str) -> str:
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
tools = [pdf_retrieval_tool, tavily_search_tool]


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
    tool_output = last_message.content
    
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
    
    # Handle the very first message with a greeting


    if len(messages) == 1:
        # return {"messages": [AIMessage(content=f"{get_time_based_greeting()}! I am Damilola... How can I help?")]}

        return {"messages": [AIMessage(content=f"{get_time_based_greeting()}! {tenant.chatbot_greeting}")]}
    
    # REVISED PROMPT: More specific on tool usage
    
    # system_prompt = SystemMessage(
    #     content=f"""You are a helpful AI assistant for ATB Bank. Your task is to analyze the user's request and decide if a tool is needed to answer it.
        
    #     You have access to the following tools:
    #     - `pdf_retrieval_tool`: For questions about bank policies, products, or internal knowledge.
    #     - `tavily_search_tool`: For general knowledge or up-to-date information.
        
    #     Based on the conversation history, either call the most appropriate tool to gather information or, if you have enough information already, prepare to answer the user directly.
    #     """
    # )
    system_prompt = SystemMessage(
        content=tenant.agent_node_prompt
    )
    
    llm_with_tools = llm.bind_tools(tools)
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
    user_query = state["messages"][-1].content
    print ("Lemu",state.get("pdf_content"))

    
    context_parts = []
    if state.get("pdf_content"): context_parts.append(f"PDF Content:\n{state['pdf_content']}")
    if state.get("web_content"): context_parts.append(f"Web Content:\n{state['web_content']}")
    # if state.get("sql_result"): context_parts.append(f"SQL Database Result:\n{state['sql_result']}")




    # --- THE FIX: PART 1 ---
    # # Store the chart data in a variable, but only put the TEXT analysis in the LLM context.
    # viz_result = state.get("visualization_result")
    # chart_base64_data = None # Initialize
    # if viz_result:
    #     analysis = viz_result.get('analysis', 'Chart analysis is not available.')   
    #     chart_base64_data = viz_result.get('image_base64') # Store the data here
    #     context_parts.append(f"Visualization Analysis:\n{analysis}") # Add ONLY analysis to context


    # if state.get("attached_content"): context_parts.append(f"Attached Content:\n{state['attached_content']}")
    # context = "\n\n".join(context_parts) if context_parts else "No additional context was retrieved."

    if state.get("attached_content"): context_parts.append(f"Attached Content:\n{state['attached_content']}")
    context = "\n\n".join(context_parts) if context_parts else "No additional context was retrieved."

    # Prompt designed to generate a structured JSON output based on the Answer schema
    # prompt = f"""You are Damilola, the AI-powered virtual assistant for ATB Bank.
    # Your goal is to provide a final, comprehensive, and empathetic answer based on the user's question and the context gathered from your tools.
    
    # User Question: "{user_query}"
    
    # Available Context:
    # ---
    # {context}
    # ---
    
    # Based on all the information above, generate a structured response. You MUST format your response as a JSON object that strictly follows the schema below.
    
    #     Generate a structured JSON response. 
    # - The 'answer' field should summarize the findings. If a chart was generated, describe what the chart shows.
    # - If a chart was generated (indicated by 'Visualization Analysis' in the context), copy the provided chart data into the 'chart_base64' field. Otherwise, leave it as null.
    #     Schema:
    # {{
    #   "answer": "str: A clear, concise, empathetic, and polite response directly addressing the user's question. Use straightforward language.",
    #   "sentiment": "int: An integer rating of the user's sentiment, from -2 (very negative) to +2 (very positive).",
    #   "ticket": "List[str]: A list of service channels relevant to any unresolved issue. Possible values: ['POS', 'ATM', 'Web', 'Mobile App', 'Branch', 'Call Center', 'Other']. Leave empty if not applicable.",
    #   "source": "List[str]: A list of sources used. Possible values: ['PDF Content', 'Web Search', 'SQL Database', 'User Provided Context', 'Internal Knowledge']. Leave empty if no specific source was used."
    # }}
    # """
    

 # --- THE FIX: PART 2 ---
    # Simplify the prompt. The LLM should NOT handle the chart_base64 data.
    y = Tenant.objects.get(pk=1)  # Get the existing record
# retrieved_template1=y.response_prompt 
    prompt1 = f"""You are Damilola, the AI-powered virtual assistant for ATB .
    Your goal is to provide a final, comprehensive, and empathetic answer based on the user's question and the context gathered from your tools.
    
    User Question: "{user_query}"
    
    Available Context:
    ---
    {context}
    ---
    
    Based on all the information above, generate a structured response. If the context includes 'Visualization Analysis', your answer should describe what the chart shows.
    You MUST format your response as a JSON object that strictly follows this schema, omitting the 'chart_base64' field as it will be handled separately.
    
    Schema:
    {{
      "answer": "str: Your clear, concise, and polite response.",
      "sentiment": "int: An integer rating of the user's sentiment (-2 to +2).",
      "ticket": "List[str]: Relevant service channels. Empty list if not applicable.",
      "source": "List[str]: Sources used. Empty list if not applicable."
    }}
    """

    prompt1 = f"""You are Damilola, the AI-powered virtual assistant for ATB. Your role is to deliver professional customer service and insightful data analysis, depending on the user's needs.

You operate in two modes:
1. **Customer Support**: Respond with empathy, clarity, and professionalism. Your goal is to resolve issues, answer questions, and guide users to helpful resources — without technical jargon or internal system references.
2. **Data Analyst**: Interpret data, explain trends, and offer actionable insights. When visualizations are included, describe what the chart shows and what it means for the user.

Your response must be:
- **Final**: No follow-up questions or uncertainty.
- **Clear and Polite**: Use emotionally intelligent language, especially if the user expresses frustration or confusion.
- **Context-Aware**: Avoid mentioning internal systems (e.g., database names or SQL sources) unless explicitly requested.
- **Structured**: Always return your answer in the following JSON format.

User Question:
"{user_query}"

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

    prompt=tenant.final_answer_prompt.format(user_query,context)

    structured_llm = llm.with_structured_output(Answer)
    final_answer_obj = structured_llm.invoke(prompt)
    # if chart_base64_data:
    #     final_answer_obj.chart_base64 = chart_base64_data
    
    # Append the human-readable part of the answer to the message history
    new_messages = state["messages"] + [AIMessage(content=final_answer_obj.answer)]
    return {
        "final_answer": final_answer_obj,
        "messages": new_messages
    }


    
    # return {
    #     "final_answer": final_answer_obj,
    #     "messages": new_messages
    # }

def summarize_conversation(state: State):
    """Generates a final summary of the conversation.
     
    Generates a final summary of the conversation after the answer has been provided.

    This node:
    1. Takes the complete message history from the state.
    2. Uses an LLM with structured output to generate a Summary object.
    3. Compiles a comprehensive 'metadatas' dictionary for logging, combining
       information from the final answer and the new summary.
    4. Updates the state with the 'conversation_summary' and 'metadatas'.
    """
    print("--- SUMMARIZE CONVERSATION NODE ---")
    
    messages = state.get("messages", [])
 
    
    # 1. Prepare the conversation history for the LLM
    # Using .get() provides a default empty list if 'messages' is not in the state
     
    if not messages:
        # If there are no messages, we can't summarize. Return an empty update.
        return {
            "conversation_summary": None,
            "metadatas": {"error": "No messages to summarize."}
        }



    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    
    summarize_prompt = f"""Please provide a structured summary of the following conversation.
    
    Conversation History:
    {conversation_history}
    
    Provide the output in a JSON format matching this schema:
    {{
        "summary": "A concise summary of the entire conversation.",
        "sentiment": "Overall sentiment score of the conversation (-2 to +2).",
        "unresolved_tickets": ["List of channels with unresolved issues."],
        "all_sources": ["All unique sources referenced in the conversation."]
    }}
    """
    
    structured_llm = llm.with_structured_output(Summary)
    summary_obj = structured_llm.invoke(summarize_prompt)
    
    # Create the final metadata dictionary for logging
    final_answer = state.get("final_answer")
    last_user_question = ""
    if len(messages) > 1:
        # The last message is the AI's answer, the one before is the user's prompt for that turn
        last_user_question = messages[-2].content

    metadata_dict = {
        "question": state["messages"][-2].content if len(state["messages"]) > 1 else "",
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

def build_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(State)
    
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tools", ToolNode(tools=tools))
    workflow.add_node("update_state", update_state_after_tool_call)

    workflow.add_node("generate_final_answer", generate_final_answer_node)
    workflow.add_node("summarize", summarize_conversation)

    workflow.set_entry_point("agent_node")

    workflow.add_conditional_edges("agent_node", tools_condition, {"tools": "tools",END: "generate_final_answer",})
    


    # workflow.add_conditional_edges("agent_node", should_continue)
    workflow.add_edge("tools", "update_state")
    workflow.add_edge("update_state", "agent_node")
    # workflow.add_edge("tools", "agent_node")
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
        # from langgraph.checkpoint.memory import InMemorySaver
        print(f"Error connecting to SQLite for checkpointing: {e}. Using in-memory saver.")
        # memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)

# Main processing function
def process_message(message_content: str, session_id: str, tenant_id:int, file_path: Optional[str] = None,):
    """Main function to process user messages using the LangGraph agent."""
    graph = build_graph()
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
                
     
            # configure(api_key=GOOGLE_API_KEY)
            # genai.configure(api_key=GOOGLE_API_KEY)  # Configure the API key
            # modelT = genai.GenerativeModel('gemini-pro-vision') # Specify the vision model
            # modelT = GenerativeModel(model_name="gemini-2.0-flash", generation_config={"temperature": 0.7,"max_output_tokens": 512 })
            # response = modelT.generate_content([image, prompt])
            # attached_content = response.text
            # Extract the text content from the response
        
                # Invoke the model
                response = llm.invoke([message])
                attached_content = response.content

                print("Attached content from image:", attached_content)

# except FileNotFoundError:
#     print(f"Error: File not found at {file_path}")

# except Exception as e:
#     print(f"Error processing image: {e}")
            

        except Exception as e:
            print(f"Error processing image attachment: {e}")
            attached_content = f"Error: Could not process attached file ({e})"
    elif file_path:
        print(f"Warning: Attached file not found at {file_path}. Skipping image processing.")
    
    initial_state = {"messages": [HumanMessage(content=message_content)], "attached_content": attached_content}
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
    # return {
    #     "answer": final_answer_content,
    #     "metadata": output.get("metadatas", {})
    # }
    else:
        return {
            "answer": "I'm sorry, I could not generate a response.",
            "chart": None,
            "metadata": {}
        }


