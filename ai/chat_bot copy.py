# # # ==========================
# # # 🌐 Django & Project Settings (Commented out as not used in standalone script)
# # # ==========================
# from django.conf import settings
# from .models import Prompt,Prompt7

import base64
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


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Ensure this is set in .env if used
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure this is set in .env if used
PDF_PATH = os.getenv("PDF_PATH", "default.pdf") # Default value for PDF_PATH

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

from sqlalchemy import create_engine


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
chatbot_model="gemini"



if chatbot_model =="gpt":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
elif chatbot_model == "deepseek":
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0, max_tokens=None, timeout=None, max_retries=2)
elif chatbot_model == "gemini":
    llm = init_chat_model("google_genai:gemini-2.0-flash")
elif chatbot_model == "groq":
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, max_tokens=None, timeout=None, max_retries=2)



llm = init_chat_model("google_genai:gemini-flash-latest")

# llm = init_chat_model("google_genai:gemini-2.0-flash")
# llms= ChatGroq( model="deepseek-r1-distill-llama-70b",temperature=0, max_tokens=None,timeout=None, max_retries=2,)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0,google_api_key=GOOGLE_API_KEY)
# llm = ChatGoogleGenerativeAI(model=google_model, temperature=0,google_api_key=GOOGLE_API_KEY)
# llm = ChatDeepSeek( model="deepseek-chat",  temperature=0, max_tokens=None, timeout=None,max_retries=2,)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)


# llm = init_chat_model("gpt-4o-mini", model_provider="openai")  # Removed because init_chat_model is not defined
# If you want to use OpenAI, you can use the following (make sure you have the correct import):





# If you want to use Gemini, uncomment the following and comment out ChatGroq
# llm= ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )


# llm = ChatGoogleGenerativeAI(
#     # model="gemini-2.5-flash-preview-04-17",
#     model ="gemini-2.5-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )



model = llm # Consistent naming

# Initialize Embeddings and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest")
global_vector_store = None

def initialize_vector_store():
    """Initializes the vector store by loading and splitting the PDF document."""
    global global_vector_store
    if global_vector_store is None:
        global_vector_store = InMemoryVectorStore(embedding=embeddings)
        
        # Use the PDF_PATH from environment variables or default
        # file_path = PDF_PATH # This should be a full path or handled by Django settings
        file_path = os.path.join(settings.MEDIA_ROOT, 'pdfs', 'ATB Bank Nigeria Groq v2.pdf')
        
        # For local testing without Django settings, you might hardcode or derive it:
        # file_path = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\media\pdfs\ATB Bank Nigeria Groq v2.pdf"

        if file_path and os.path.exists(file_path):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
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
DB_FILE_PATH = f"sqlite:///{settings.DATABASES['default']['NAME']}"
# DB_URI = f"sqlite:///{DB_FILE_PATH}" 
DB_URI = f"sqlite:///{settings.DATABASES['default']['NAME']}"


# DB_URI = os.getenv("DB_URI")
db = None
try:
    db = SQLDatabase.from_uri(DB_URI)
    print(f"SQLDatabase connected to {DB_URI} successfully.")
except Exception as e:
    print(f"Error connecting to SQLDatabase at {DB_URI}: {e}. SQL query tool will not be available.")


# ==========================
# 📝 Pydantic Schemas
# ==========================
# class Answer(BaseModel):
#     answerA: str = Field(..., description="A clear, concise, empathetic, and polite response...")
#     sentimentA: int = Field(..., description="An integer rating of the user's sentiment...")
#     ticketA: list[str] = Field(..., description='A list of specific transaction or service channels...')
#     sourceA: list[str] = Field(..., description='A list of specific sources...')

# kd begin 





# Assume 'llm', 'db', 'global_vector_store', 'model', etc. are initialized elsewhere
# For example:
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# db = SQLDatabase.from_uri("sqlite:///db.sqlite3")
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

# Initialize SQL Agent (Primary Method)
SQL_AGENT = None
if db:
    try:
        SQL_TOOLKIT = SQLDatabaseToolkit(db=db, llm=llm)
        SQL_SYSTEM_PROMPT = """You are an agent designed to interact with a SQL database. Given an input question, create a syntactically correct {dialect} query, execute it, and return the answer.
        - You must query only the necessary columns.
        - You must double-check your query before execution.
        - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP).
        - ALWAYS look at the tables first to understand the schema."""
        
        SQL_AGENT = create_react_agent(
            llm,
            SQL_TOOLKIT.get_tools(),
            prompt=SQL_SYSTEM_PROMPT.format(dialect=db.dialect),
        )
        print("SQL Agent initialized successfully.")
    except Exception as e:
        print(f"Error initializing SQL Agent: {e}. SQL query tool will not be available.")

def execute_sql_query_func(query: str) -> str:
    """Executes a SQL query using the pre-initialized SQL agent and returns the result."""
    if not SQL_AGENT:
        return {"sql_result": "Error: SQL Agent not initialized."}
    try:
        response_generator = SQL_AGENT.stream(
            {"messages": [HumanMessage(content=query)]}, stream_mode="values"
        )
        full_response_content = []
        for chunk in response_generator:
            if 'messages' in chunk and chunk['messages']:
                content = chunk['messages'][-1].content
                if content:
                    full_response_content.append(content)
        
        result = "\n".join(full_response_content) if full_response_content else "No response from SQL agent."
        return {"sql_result": result}
    except Exception as e:
        return {"sql_result": f"Error executing SQL query: {e}"}

sql_query_tool = Tool(
    name="sql_query_tool",
    description="Useful for answering questions requiring data from a SQL database (e.g., 'How many users are there?'). Input should be a natural language question.",
    func=execute_sql_query_func,
    args_schema=SQLQueryInput,
)





def get_column_types(df: pd.DataFrame):
    """Helper function to identify column types for plotting."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    return numeric_cols, categorical_cols, date_cols




# --- FULLY ENHANCED VISUALIZATION TOOL ---
def generate_visualization_func(query: str) -> dict:
    """
    Generates a data visualization based on a natural language query.
    """
    logging.info(f"--- Generating Visualization for query: '{query}' ---")
    analysis_text = "" # Initialize in case of early failure
    try:
        # Step 1: Generate SQL from the natural language query (with few-shot prompt)
        sql_generation_prompt = f"""Given the user's question, create a single, syntactically correct SQL query to retrieve the data needed for a chart.
Do not include any other text or explanation, just the SQL query itself.

Tables available: {db.get_table_info()}

### Example ###
User question: "Show me the total transaction value for each month this year."
SQL Query:
```sql
SELECT
  STRFTIME('%Y-%m', timestamp) AS month,
  SUM(amount) AS total_value
FROM
  ai_transaction
WHERE
  STRFTIME('%Y', timestamp) = STRFTIME('%Y', 'now')
GROUP BY
  month
ORDER BY
  month;
```
### End Example ###

User question: "{query}"
SQL Query:
"""
        raw_sql_query = llm.invoke(sql_generation_prompt).content.strip()

        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_sql_query, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
        else:
            sql_query = raw_sql_query
        
        logging.info(f"Generated SQL: {sql_query}")

        # Step 2: Execute the query with Pandas
        engine = create_engine(DB_URI)
        df = pd.read_sql_query(sql_query, con=engine)
        
        if df.empty:
            logging.warning("Query returned no data.")
            return {"visualization_result": {"analysis": "I found no data to visualize for your request.", "image_base64": None}}

        # --- ENHANCED LOGGING: Log DataFrame details ---
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info_str = buffer.getvalue()
        logging.info(f"--- DataFrame Details ---\nHead:\n{df.head().to_string()}\nInfo:\n{df_info_str}")

        # Step 3: Determine the best chart type
        df_info_for_prompt = f"Data Columns: {df.columns.tolist()}\nData Head:\n{df.head().to_string()}"
        chart_selection_prompt = f"""
        Given the user's original query '{query}' and the following data summary, what is the best chart type to use?
        Your answer must be a single word from this list: 'bar', 'line', 'scatter', 'pie'.

        Data Summary:\n{df_info_for_prompt}
        """
        chart_type = llm.invoke(chart_selection_prompt).content.strip().lower()
        logging.info(f"LLM chose chart type: '{chart_type}'")

        # Step 4: Get textual analysis from the LLM
        analysis_prompt = f"Analyze this data and provide a brief, insightful summary based on the user's original request: '{query}'.\n\nData:\n{df.to_csv(index=False)}"        
        analysis_text = llm.invoke(analysis_prompt).content
        logging.info(f"Generated Analysis: {analysis_text[:200]}...") # Log a snippet

        # Step 5: Generate the plot using intelligent chart selection
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric, categorical, dates = get_column_types(df)
        
        if chart_type == 'bar' and categorical and numeric:
            x_col = categorical[0]
            if len(numeric) > 1: # Handle multi-series bar charts
                df.set_index(x_col)[numeric].plot(kind='bar', ax=ax, figsize=(12, 7))
                ax.set_ylabel("Values")
                ax.legend(title='Metrics')
            else: # Handle single-series bar charts
                y_col = numeric[0]
                df.plot(kind='bar', x=x_col, y=y_col, ax=ax, legend=False)
                ax.set_ylabel(y_col.replace('_', ' ').title())
            ax.set_xlabel(x_col.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')

        elif chart_type == 'line' and (dates or numeric):
            x_col = dates[0] if dates else numeric[0]
            y_cols = [c for c in numeric if c != x_col]
            if not y_cols: y_cols = numeric # Fallback if x is also the only numeric
            df.plot(kind='line', x=x_col, y=y_cols, ax=ax, marker='o')
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel("Value")
            plt.xticks(rotation=45, ha='right')

        elif chart_type == 'scatter' and len(numeric) >= 2:
            x_col, y_col = numeric[0], numeric[1]
            df.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_col.replace('_', ' ').title())

        elif chart_type == 'pie' and categorical and numeric:
            df.set_index(categorical[0])[numeric[0]].plot(
                kind='pie', ax=ax, autopct='%1.1f%%', startangle=90
            )
            ax.set_ylabel('')
        
        else: # Fallback
            logging.warning(f"Could not find a perfect chart match for type '{chart_type}'. Using generic plot.")
            df.plot(ax=ax)
       
        # Formatting common to all charts
        ax.set_title(query.title())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        
        # Step 6: Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        logging.info(f"Successfully generated plot image (Base64 length: {len(image_base64)}).")

        return {
            "visualization_result": {
                "analysis": analysis_text,
                "image_base64": image_base64
            }
        }
    
    except Exception as e:
        # --- ENHANCED LOGGING: Log the full exception traceback ---
        logging.error("Error in visualization tool", exc_info=True)
        analysis_text_on_error = analysis_text if analysis_text else f"Sorry, I encountered an unrecoverable error: {e}"
        return {"visualization_result": {"analysis": analysis_text_on_error, "image_base64": None}}





def get_column_types1(df: pd.DataFrame):
    """Helper function to identify column types for plotting."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    return numeric_cols, categorical_cols, date_cols
# --- NEW VISUALIZATION TOOL ---
def generate_visualization_func1(query: str) -> dict:
    """
    Generates a data visualization based on a natural language query.
    1. Converts the query to SQL.
    2. Executes the SQL to get data.
    3. Generates a textual analysis of the data.
    4. Creates a plot and returns it as a base64 string.
    """
    print(f"--- Generating Visualization for query: '{query}' ---")
    try:
        # Step 1: Generate SQL from the natural language query
        # sql_generation_prompt = f"""Given the user's question, create a syntactically correct SQL query to retrieve the data needed for a chart.
        # Tables available: {db.get_table_info()}
        # User question: "{query}"
        # """
        # sql_query = llm.invoke(sql_generation_prompt).content.strip().replace("```sql", "").replace("```", "")
#         sql_generation_prompt2 = f"""Given the user's question, create a single, syntactically correct SQL query to retrieve the data needed for a chart.
# Do not include any other text or explanation, just the SQL query itself.
# Tables available: {db.get_table_info()}
# User question: "{query}"

# """
        # # sql_query = llm.invoke(sql_generation_prompt).content.strip() # Remove the .replace() calls
        # sql_generation_prompt = f"""Given the user's question, create a single, syntactically correct SQL query to retrieve the data needed for a chart.
        # Do not include any other text or explanation, just the SQL query itself.
        # Tables available: {db.get_table_info()}
        # User question: "{query}"
        # """
        

        # raw_sql_query = llm.invoke(sql_generation_prompt).content.strip()

        # # More robustly find a SQL block
        # match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_sql_query, re.DOTALL)
        # if match:
        #     sql_query = match.group(1).strip()
        # else:
        #     # If no markdown block is found, assume the whole output is the query
        #     sql_query = raw_sql_query


     # Step 1: Generate SQL from the natural language query
        sql_generation_prompt = f"""Given the user's question, create a single, syntactically correct SQL query to retrieve the data needed for a chart.
        Do not include any other text or explanation, just the SQL query itself.
        Tables available: {db.get_table_info()}
        User question: "{query}"
        """
        raw_sql_query = llm.invoke(sql_generation_prompt).content.strip()

        # AMENDED: Use robust regex to find the SQL block
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_sql_query, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
        else:
            sql_query = raw_sql_query


         # --- FIX: Clean the query by removing markdown syntax ---
        # sql_query = raw_sql_query.replace("```sql", "").replace("```", "").strip()
        
        print(f"Generated SQL: {sql_query}")
        # print(f"Generated SQL: {sql_query}")

        # Step 2: Execute the query with Pandas
        # engine = create_engine(DB_URI)
        # df = pd.read_sql_query(sql_query, con=engine)
        # print ("AYEM")
        # if df.empty:
        #     return {"visualization_result": {"analysis": "I found no data to visualize for your request.", "image_base64": None}}


           # Step 2: Execute the query with Pandas
        engine = create_engine(DB_URI)
        df = pd.read_sql_query(sql_query, con=engine)
        print("AYEM")
        if df.empty:
            return {"visualization_result": {"analysis": "I found no data to visualize for your request.", "image_base64": None}}

 # --- NEW: Step 3.5: Determine the best chart type ---
        df_info = f"""
        Data Columns: {df.columns.tolist()}
        Data Types: {df.dtypes.to_dict()}
        Data Head:
        {df.head().to_string()}
        """

        chart_selection_prompt = f"""
        Given the user's original query '{query}' and the following data summary, what is the best chart type to use?
        Your answer must be a single word from this list: 'bar', 'line', 'scatter', 'pie'.

        Data Summary:
        {df_info}
        """
        chart_type = llm.invoke(chart_selection_prompt).content.strip().lower()
        print(f"--- LLM chose chart type: '{chart_type}' ---")


       # Step 4: Get textual analysis from the LLM
        data_str = df.to_csv(index=False)
        analysis_prompt = f"Analyze this data and provide a brief, insightful summary based on the user's original request: '{query}'.\n\nData:\n{data_str}"        
        analysis_text = llm.invoke(analysis_prompt).content

        # --- AMENDED: Step 5: Generate the plot using intelligent chart selection ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric, categorical, dates = get_column_types(df)
        
        # Simple logic to determine plot type (can be improved)
        # if len(df.columns) == 2:
        #     x_col, y_col = df.columns[0], df.columns[1]
        #     if pd.api.types.is_numeric_dtype(df[y_col]):
        #         df.plot(kind='bar', x=x_col, y=y_col, ax=ax, legend=False)
        #         ax.set_ylabel(y_col.replace('_', ' ').title())
        #         ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        #     else: # Fallback for non-numeric y
        #         df[x_col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        if chart_type == 'bar' and categorical and numeric:

            x_col, y_col = categorical[0], numeric[0]
            df.plot(kind='bar', x=x_col, y=y_col, ax=ax, legend=False)
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_col.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')     
            ax.set_title(query.title())
            ax.set_xlabel(x_col.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')
        

        elif chart_type == 'line' and (dates or numeric):
            x_col = dates[0] if dates else numeric[0]
            y_cols = [c for c in numeric if c != x_col]
            df.plot(kind='line', x=x_col, y=y_cols, ax=ax)
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel("Value")
            plt.xticks(rotation=45, ha='right')

        elif chart_type == 'scatter' and len(numeric) >= 2:
            x_col, y_col = numeric[0], numeric[1]
            df.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_col.replace('_', ' ').title())

        elif chart_type == 'pie' and categorical and numeric:
            # Pie chart works best with a single categorical and numeric series
            df.set_index(categorical[0])[numeric[0]].plot(
                kind='pie', ax=ax, autopct='%1.1f%%', startangle=90
            )
            ax.set_ylabel('') # Hide y-label for pie charts
        
        else:
            # Fallback for other data shapes
            df.plot(ax=ax)
       
            # ax.set_title(query.title())
         # Formatting common to all charts

        ax.set_title(query.title())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        # plt.tight_layout()
        
        # Step 5: Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)


        return {
            "visualization_result": {
                "analysis": analysis_text,
                "image_base64": image_base64
            }
        }
    
    
    
    # except Exception as e:
    #     print(f"Error in visualization tool: {e}")
    #     return {"visualization_result": {"analysis": f"Sorry, I encountered an error while creating the visualization: {e}", "image_base64": None}}

    except Exception as e:
            print(f"Error in visualization tool: {e}")
            # Return the analysis if it was generated before the error
            analysis_text_on_error = analysis_text if 'analysis_text' in locals() else f"Sorry, I encountered an error: {e}"
            return {"visualization_result": {"analysis": analysis_text_on_error, "image_base64": None}}




generate_visualization_tool = Tool(
    name="generate_visualization_tool",
    description="Use this tool to create charts, graphs, plots, or any data visualizations. This is the best tool when the user asks to 'plot', 'chart', 'visualize', or 'draw' data.",
    func=generate_visualization_func,
    args_schema=VisualizationInput,
)

# tools = [pdf_retrieval_tool, tavily_search_tool, sql_query_tool, generate_visualization_tool]


# Combine all tools (Corrected logic)
tools = [pdf_retrieval_tool, tavily_search_tool,generate_visualization_tool]
if SQL_AGENT:
    tools.append(sql_query_tool)






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
    elif tool_name == "sql_query_tool":
        return {"sql_result": tool_output}
    elif tool_name == "generate_visualization_tool":
        # The tool's output is a stringified JSON. We need to parse it.

        try:
            # Find the start and end of the JSON object in the raw output
            start_index = tool_output.find('{')
            end_index = tool_output.rfind('}') + 1
            if start_index != -1 and end_index != -1:
                json_string = tool_output[start_index:end_index]
                parsed_output = json.loads(json_string)
                viz_data = parsed_output.get("visualization_result")
                if viz_data:
                    return {"visualization_result": viz_data}
            return {} # Return empty dict if no valid JSON is found
        except json.JSONDecodeError as e:
            print(f"Error parsing visualization tool output: {e}")
            return {}
        # try:
        #     parsed_output = json.loads(tool_output)
        #     # The tool returns a dictionary with one key: "visualization_result"
        #     viz_data = parsed_output.get("visualization_result")
        #     if viz_data:
        #         return {"visualization_result": viz_data}
        # except json.JSONDecodeError as e:
        #     print(f"Error parsing visualization tool output: {e}")
        #     return {}
    
    return {}




# def agent_node1(state: State):
#     """
#     The Router Node: Decides whether to call a tool or generate a final answer.
#     This node's prompt is focused on routing, not on generating the final answer.
#     """
#     print("--- AGENT NODE (ROUTER) ---")
#     messages = state["messages"]
    
#     # Handle the very first message with a greeting
#     if len(messages) == 1:
#         return {"messages": [AIMessage(content=f"{get_time_based_greeting()}! I am Damilola, your AI-powered virtual assistant. Welcome to ATB Bank. How can I help you today?")]}
    
#     system_prompt = SystemMessage(
#         content=f"""You are Damilola, a helpful AI assistant for ATB Bank. Your role is to decide the next step in the conversation.
        
#         You have access to the following tools: {', '.join([t.name for t in tools])}.
        
#         1. Review the user's latest message in the context of the conversation history.
#         2. If the user's question can be answered using one of your tools, call the most appropriate tool with the correct input.
#         3. If you have already used a tool and have enough information to answer the user's question, respond directly.
#         4. Do not generate the final answer here. Your job is to either call a tool or indicate that you're ready to answer.
        
#         Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#         """
#     )
    
#     llm_with_tools = llm.bind_tools(tools)
#     response = llm_with_tools.invoke([system_prompt] + messages)
    
#     return {"messages": [response]}
def agent_node(state: State):
    """
    The Router Node: Decides whether to call a tool or generate a final answer.
    """
    print("--- AGENT NODE (ROUTER) ---")
    messages = state["messages"]
    
    # Handle the very first message with a greeting
    if len(messages) == 1:
        return {"messages": [AIMessage(content=f"{get_time_based_greeting()}! I am Damilola... How can I help?")]}
    
    # REVISED PROMPT: More specific on tool usage
    system_prompt = SystemMessage(
        content=f"""You are a helpful AI assistant for ATB Bank. Your task is to analyze the user's request and decide if a tool is needed to answer it.
        
        You have access to the following tools:
        - `pdf_retrieval_tool`: For questions about bank policies, products, or internal knowledge.
        - `tavily_search_tool`: For general knowledge or up-to-date information.
        - `sql_query_tool`: For questions about specific data, like user counts or transaction volumes.
        - **`generate_visualization_tool`**: **Use this tool ONLY when the user explicitly asks to 'plot', 'chart', 'graph', or 'visualize' data. This is your primary tool for creating visual representations from database data.**
        
        Based on the conversation history, either call the most appropriate tool to gather information or, if you have enough information already, prepare to answer the user directly.
        """
    )
    
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_prompt] + messages)
    
    last_tool_name = None
    if response.tool_calls:
        last_tool_name = response.tool_calls[0]['name']
        print(f"LLM decided to call tool: {last_tool_name}")
        
    return {"messages": [response], "last_tool_name": last_tool_name}




def agent_node2(state: State):
    """
    The Router Node: Decides whether to call a tool or generate a final answer.
    """
    print("--- AGENT NODE (ROUTER) ---")
    messages = state["messages"]
    
    # Handle the very first message with a greeting
    if len(messages) == 1:
        return {"messages": [AIMessage(content=f"{get_time_based_greeting()}! I am Damilola... How can I help?")]}
    
    # A more focused system prompt for routing
    system_prompt = SystemMessage(
        content=f"""You are a helpful AI assistant for ATB Bank. Your task is to analyze the user's request and decide if a tool is needed to answer it.
        
        You have access to the following tools:
        - `pdf_retrieval_tool`: For questions about bank policies, products, or internal knowledge.
        - `tavily_search_tool`: For general knowledge or up-to-date information.
        - `sql_query_tool`: For questions about specific data, like user counts.
        
        Based on the conversation history, either call the most appropriate tool to gather information or, if you have enough information already, prepare to answer the user directly.
        """
    )
    
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_prompt] + messages)
    
    ### MODIFIED ###
    # If a tool is called, we store its name in the state to use later.
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
    if state.get("sql_result"): context_parts.append(f"SQL Database Result:\n{state['sql_result']}")




    # # NEW: Handle visualization result
    # viz_result = state.get("visualization_result")
    # chart_base64 = None
    # if viz_result:
    #     analysis = viz_result.get('analysis', 'Chart analysis is not available.')   
    #     chart_base64 = viz_result.get('image_base64')
    #     context_parts.append(f"Visualization Analysis:\n{analysis}")

    # --- THE FIX: PART 1 ---
    # Store the chart data in a variable, but only put the TEXT analysis in the LLM context.
    viz_result = state.get("visualization_result")
    chart_base64_data = None # Initialize
    if viz_result:
        analysis = viz_result.get('analysis', 'Chart analysis is not available.')   
        chart_base64_data = viz_result.get('image_base64') # Store the data here
        context_parts.append(f"Visualization Analysis:\n{analysis}") # Add ONLY analysis to context


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

    prompt = f"""You are Damilola, the AI-powered virtual assistant for ATB. Your role is to deliver professional customer service and insightful data analysis, depending on the user's needs.

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



    structured_llm = llm.with_structured_output(Answer)
    final_answer_obj = structured_llm.invoke(prompt)
    if chart_base64_data:
        final_answer_obj.chart_base64 = chart_base64_data
    
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
def process_message(message_content: str, session_id: str, file_path: Optional[str] = None):
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


















# kd ed 
# # ==========================
# # ▶️ Main Execution
# # ==========================
# if __name__ == "__main__":

#     # Function to set up a sample SQLite database
#     def setup_sample_db(db_file_path):
#         conn_local = None
#         try:
#             # Ensure the directory exists
#             os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
#             # Connect to the database file
#             conn_local = sqlite3.connect(db_file_path)
#             cursor = conn_local.cursor()

#             # Create a sample table
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS customers (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     name TEXT NOT NULL,
#                     email TEXT UNIQUE NOT NULL,
#                     age INTEGER,
#                     city TEXT
#                 );
#             """)

#             # Insert some sample data (only if not already present)
#             cursor.execute("INSERT OR IGNORE INTO customers (id, name, email, age, city) VALUES (1, 'Alice Smith', 'alice@example.com', 30, 'New York');")
#             cursor.execute("INSERT OR IGNORE INTO customers (id, name, email, age, city) VALUES (2, 'Bob Johnson', 'bob@example.com', 24, 'Los Angeles');")
#             cursor.execute("INSERT OR IGNORE INTO customers (id, name, email, age, city) VALUES (3, 'Charlie Brown', 'charlie@example.com', 35, 'New York');")
#             cursor.execute("INSERT OR IGNORE INTO customers (id, name, email, age, city) VALUES (4, 'Diana Prince', 'diana@example.com', 29, 'London');")
            
#             conn_local.commit()
#             print(f"Sample 'customers' table created and populated in {db_file_path}.")
#         except Exception as e:
#             print(f"Error setting up sample database: {e}")
#         finally:
#             if conn_local:
#                 conn_local.close()

#     # Setup the sample database before running the agent
#     setup_sample_db(DB_FILE_PATH)

#     # Example user queries
#     # user_query = "What are the bank's account opening requirements and which services are available?"
#     user_query = "How many customers do we have?" # Example query for SQL tool
#     # user_query = "What is the capital of France?" # Example query for web search tool

#     session_id = "test_session_1"
    
#     # No file attached for this example
#     attached_file_path = None 

#     try:
#         final_output = process_message(user_query, session_id, attached_file_path)
        
#         print("\n==================\nFinal Answer:")
#         print(final_output['messages'])
#         print("\nMetadata (Summary):")
#         pprint(final_output['metadata'])

#     except Exception as e:
#         print(f"\nAn error occurred during execution: {e}")









importants= """
gross
net
charge
amount
Basic
Transport
Housing
NHF
NHIS
NSITF
tax
pension
employerPension
deduction
OtherAllowance
_id
fullname
id
employeeID.phone
employeeID.accountNumber
employeeID.pencomID
employeeID.annualSalary
employeeID.jobRole.name
employeeID.accountName
meta.annualGross
meta.sumBasicHousingTransport
meta.earnedIncome
meta.earnedIncomeAfterRelief
meta.sumRelief
"""

def important ():
    key_fields = [line.strip() for line in importants.strip().splitlines()]
    return key_fields


desired_columnsT2= [
     "type", "payment_status", "gross", "net", "charge", "amount", "_id", "Basic", "Housing", "Transport",
    "Leave", "Medical", "Bread", "Shoe", "Car", "House", "Nepa", "allOtherItems", "notPayableDetails",
    "Leave_schedule", "Medical_schedule", "Bread_schedule", "Shoe_schedule", "Car_schedule", "Nepa_schedule",
    "Basic_type", "Transport_type", "Housing_type", "Leave_type", "Medical_type", "Bread_type", "Shoe_type",
    "Car_type", "House_type", "Nepa_type", "reliefs", "NHF", "NHIS", "NSITF", "benefits", "benefit", "tax",
    "pension", "employerPension", "bonuses", "bonus", "deduction", "OtherAllowance", "allowance", "year",
    "paymentDate", "percentage", "month", "companyID", "CashAdvance", "leaveAllowance", "cashReimbursement",
    "__v", "createdAt", "updatedAt", "fullname", "id", "employeeID.companyLicense.assignedAt",
    "employeeID.companyLicense.licenseId", "employeeID.jobRole.jobRoleId", "employeeID.jobRole.name",
    "employeeID.employeeConfirmation.confirmed", "employeeID.employeeConfirmation.status",
    "employeeID.employeeConfirmation.date", "employeeID.termination.prorated.status",
    "employeeID.termination.status", "employeeID.termination.medicalBalance", "employeeID.termination.leaveBalance",
    "employeeID.termination.isPaid", "employeeID.termination.otherAllowancesBalance", "employeeID.termination.date",
    "employeeID.termination.reason", "employeeID.employeeType", "employeeID.employeeSubordinates",
    "employeeID.mentees", "employeeID.aboutEmployee", "employeeID.facebookUrl", "employeeID.twitterUrl",
    "employeeID.linkedInUrl", "employeeID.employeeTribe", "employeeID.allowanceType", "employeeID.annualSalary",
    "employeeID.costOfHire", "employeeID.hourlyRate", "employeeID.isActive", "employeeID.isConfirmed",
    "employeeID.dependents", "employeeID._id", "employeeID.employeeEmail", "employeeID.firstName",
    "employeeID.middleName", "employeeID.lastName", "employeeID.religion", "employeeID.gender", "employeeID.bankName",
    "employeeID.phone", "employeeID.accountNumber", "employeeID.salaryScheme._id", "employeeID.salaryScheme.name",
    "employeeID.salaryScheme.items", "employeeID.salaryScheme.country", "employeeID.salaryScheme.companyID",
    "employeeID.salaryScheme.employeeContribution", "employeeID.salaryScheme.employerContribution",
    "employeeID.salaryScheme.createdAt", "employeeID.salaryScheme.updatedAt", "employeeID.salaryScheme.__v",
    "employeeID.branchID._id", "employeeID.branchID.branchKeyName", "employeeID.branchID.branchName",
    "employeeID.branchID.companyID", "employeeID.branchID.createdOn", "employeeID.branchID.modifiedOn",
    "employeeID.branchID.createdAt", "employeeID.branchID.updatedAt", "employeeID.branchID.id",
    "employeeID.branchID.__v", "employeeID.staffID", "employeeID.employeeHireDate", "employeeID.dateOfBirth",
    "employeeID.employeeCadreStep", "employeeID.employeeCadre", "employeeID.createdOn", "employeeID.modifiedOn",
    "employeeID.companyID", "employeeID.companyFID", "employeeID.createdAt", "employeeID.updatedAt", "employeeID.id",
    "employeeID.__v", "employeeID.dailyPay", "employeeID.employeeManager", "employeeID.myx3ID", "employeeID.competency",
    "employeeID.departmentID", "employeeID.employementType", "employeeID.workArrangement", "employeeID.teamID",
    "employeeID.divisionID", "employeeID.accountName", "employeeID.bankCode", "employeeID.recipientCode",
    "employeeID.talentNominations", "employeeID.addonLicenses", "employeeID.leaveCategory", "employeeID.city",
    "employeeID.payRateType", "employeeID.businessUnitID", "employeeID.employeeCategory", "employeeID.lastHireDate",
    "employeeID.maritalStatus", "meta.annualGross", "meta.sumBasicHousingTransport", "meta.earnedIncome",
    "meta.earnedIncomeAfterRelief", "meta.sumRelief", "pfa", "taxAuthority", "employeeID.employeeConfirmation.processId",
    "employeeID.pencomID", "employeeID.profileImgUrl", "employeeID.nhfPIN", "employeeID.pfa", "employeeID.taxAuthority",
    "employeeID.taxID", "employeeID.promotionDate", "employeeID.bankCountry", "employeeID.branchCode",
    "employeeID.employeeTitle", "payslipPDFView.person.profileImgUrl", "payroll_id", "employee_ID", "Other Items"
]

desired_columnsT= [
     "type", "payment_status", "gross", "net", "charge", "amount", "_id", "Basic", "Housing", "Transport",
    "Leave", "Medical", "Bread", "Shoe", "Car", "House", "Nepa", "allOtherItems", "notPayableDetails",
    "Leave_schedule", "Medical_schedule", "Bread_schedule", "Shoe_schedule", "Car_schedule", "Nepa_schedule",
    "Basic_type", "Transport_type", "Housing_type", "Leave_type", "Medical_type", "Bread_type", "Shoe_type",
    "Car_type", "House_type", "Nepa_type", "reliefs", "NHF", "NHIS", "NSITF", "benefits", "benefit", "tax",
    "pension", "employerPension", "bonuses", "bonus", "deduction", "OtherAllowance", "allowance", "year",
    "paymentDate", "percentage", "month", "companyID", "CashAdvance", "leaveAllowance", "cashReimbursement",
    "__v", "createdAt", "updatedAt", "fullname", "id", "employeeID.companyLicense.assignedAt",
    "employeeID.companyLicense.licenseId", "employeeID.jobRole.jobRoleId", "employeeID.jobRole.name",
    "employeeID.employeeConfirmation.confirmed", "employeeID.employeeConfirmation.status",
    "employeeID.employeeConfirmation.date", "employeeID.termination.prorated.status",
    "employeeID.termination.status", "employeeID.termination.medicalBalance", "employeeID.termination.leaveBalance",
    "employeeID.termination.isPaid", "employeeID.termination.otherAllowancesBalance", "employeeID.termination.date",
    "employeeID.termination.reason", "employeeID.employeeType", "employeeID.employeeSubordinates",
    "employeeID.mentees", "employeeID.aboutEmployee", "employeeID.facebookUrl", "employeeID.twitterUrl",
    "employeeID.linkedInUrl", "employeeID.employeeTribe", "employeeID.allowanceType", "employeeID.annualSalary",
    "employeeID.costOfHire", "employeeID.hourlyRate", "employeeID.isActive", "employeeID.isConfirmed",
    "employeeID.dependents", "employeeID._id", "employeeID.employeeEmail", "employeeID.firstName",
    "employeeID.middleName", "employeeID.lastName", "employeeID.religion", "employeeID.gender", "employeeID.bankName",
    "employeeID.phone", "employeeID.accountNumber", "employeeID.salaryScheme._id", "employeeID.salaryScheme.name",
    "employeeID.salaryScheme.items", "employeeID.salaryScheme.country", "employeeID.salaryScheme.companyID",
    "employeeID.salaryScheme.employeeContribution", "employeeID.salaryScheme.employerContribution",
    "employeeID.salaryScheme.createdAt", "employeeID.salaryScheme.updatedAt", "employeeID.salaryScheme.__v",
    "employeeID.branchID._id", "employeeID.branchID.branchKeyName", 
    "employeeID.branchID.companyID", "employeeID.branchID.createdOn", "employeeID.branchID.modifiedOn",
    "employeeID.branchID.createdAt", 
    "employeeID.branchID.__v", "employeeID.staffID", "employeeID.employeeHireDate", "employeeID.dateOfBirth",
    "employeeID.employeeCadreStep", "employeeID.employeeCadre", "employeeID.createdOn",
    "employeeID.companyID", "employeeID.companyFID", "employeeID.createdAt", "employeeID.updatedAt", "employeeID.id",
    "employeeID.__v","employeeID.employeeManager", "employeeID.myx3ID", 
    "employeeID.departmentID", "employeeID.employementType", 
     "employeeID.city",
    "employeeID.payRateType", "employeeID.businessUnitID", "employeeID.employeeCategory", "employeeID.lastHireDate",
    "meta.annualGross", "meta.sumBasicHousingTransport",

]



def desire():
    desire = desired_columnsT
    return desire 

# Define your multiply function
def atb1(a, b):
    data1_raw = a
    data_raw = b
    systemprompt = f"""
You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets I will provide: a "previous period":{data1_raw} dataset and a "current period":{data_raw} dataset.
Your analysis must follow these steps:
Identify Employee Status: For every employee ID across both datasets, determine their status as one of the following:
Continuing: Appears in both datasets.
New: Appears only in the current period dataset.
Departed: Appears only in the previous period dataset.
Calculate Variances: Compute the monetary variance (NGN) for Gross Pay, Tax, and Pension for each employee and for the overall totals.
Identify Key Drivers: Analyze the variances to find the main reasons for the changes. Specifically look for:
Changes in headcount (new hires vs. departures).
Pay raises or decreases for continuing employees.
Unusual changes, such as a change in a deduction (like tax) without a corresponding change in gross pay. This is a critical insight to identify.
You must structure your output as a professional report using Markdown formatting with the following exact sections:
1. Executive Summary:
Start with the single most important number: the total variance in Gross Pay.
State whether this variance is favorable (a cost decrease) or unfavorable (a cost increase) from the company's perspective.
Briefly state the primary reason for this variance (e.g., "driven by headcount changes").
2. Overall Payroll Summary:
Create a summary table comparing the totals of the two periods.
The table columns must be: Metric, Previous Period, Current Period, Variance (NGN), and Variance (%).
Include rows for Gross Pay, Total Tax, and Total Pension.
3. Detailed Variance Analysis:
Create a sub-section titled 3.1. Headcount Changes that lists the departed and new employees and the gross pay impact of each group.
Create a sub-section titled 3.2. Variances for Continuing Employees that explicitly calls out any employees with changes in pay or deductions, specifying the exact variance amount.
4. Reconciliation of Gross Pay Variance:
Provide a simple table that clearly shows how the individual key drivers (e.g., Departures, New Hires, Pay Raises) sum up to the total Gross Pay variance. This proves your analysis is correct.
5. Conclusion & Recommendations:
Conclude with clear, actionable recommendations based on your findings. For example: "Verify the authorization for [Employee]'s pay raise" or "Investigate the reason for the tax change for [Employee], as their gross pay was unchanged."
Ensure the tone is professional, objective, and data-driven. Use currency formatting (e.g., N5,200) throughout the report."""
    responseY = llm.invoke([
        systemprompt,
        HumanMessage(content="Please review")
    ])
    return responseY.content



# retrieved_template1 = py.variance_prompt
# systemprompt=retrieved_template1

def atb(old, new,llmv,retrieved_template6):
    old = old
    new = new
    systemprompt = f"""

You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets provided in JSON format:

"Previous Payroll Period": {old}

"Current Payroll Period": {new}

Your analysis must be meticulous, data-driven, and presented in a clear, professional Markdown report.

Analysis Instructions

You must follow these steps precisely:

1. Identify Employee Status:
Use employeeID._id as the unique identifier for each employee across both datasets. Classify each employee into one of the following categories:

Continuing: The employeeID._id exists in both the previous and current datasets.

New: The employeeID._id exists only in the current dataset.

Departed: The employeeID._id exists only in the previous dataset.

Suspicious: Flag any continuing employee as "Suspicious" if ANY of the following conditions are met. Comparison must be exact and case-sensitive.

fullname has changed.

employeeID.bankName has changed (e.g., "Zenith Bank" vs. "Zenith Banks").

employeeID.accountNumber has changed.

employeeID.phone has changed.

Suspicious (Duplicate ID): If an employeeID._id appears more than once within the payslips array of the Current Payroll Period, it must be flagged as a critical data integrity issue.

Note: If a file contains multiple top-level payroll objects, consolidate all payslips into a single list for each period before starting the analysis.

2. Calculate Monetary Variances:
For each employee and for the overall totals, compute the monetary variance (Current - Previous) in NGN for the following fields:

gross (Gross Pay)

tax (Tax)

pension (Employee Pension Contribution)

3. Identify Key Drivers of Variance:
Analyze the data to determine the root causes of any financial changes. Your analysis must explicitly connect variances to:

Headcount Changes: The financial impact of new hires and departures.

Pay Changes: Changes in gross pay for continuing employees.

Anomalies & Data Quality: The financial impact of suspicious records, especially duplicate entries.

Required Output Format (Markdown)

Generate the report using the exact structure and formatting below.

1. Executive Summary

Start with a headline figure: the total variance in Gross Pay.

State whether the variance is favorable (cost decrease) or unfavorable (cost increase).

Briefly summarize the primary drivers (e.g., headcount changes, significant pay adjustments, data anomalies).

2. Overall Payroll Summary

Provide a Markdown table comparing the aggregate values:

Generated markdown
| Metric        | Previous Period | Current Period | Variance (NGN) | Variance (%) |
|---------------|-----------------|----------------|----------------|--------------|
| Gross Pay     |                 |                |                |              |
| Total Tax     |                 |                |                |              |
| Total Pension |                 |                |                |              |

3. Detailed Variance Analysis
3.1 Headcount Changes

List new and departed employees and their financial impact.

Departed Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total Departures** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END

New Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total New Hires** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
3.2 Variances for Continuing & Suspicious Employees

Create a table for all continuing employees. Highlight significant changes and flag suspicious records.

Generated markdown
| Employee Name | Employee ID | Gross Pay Variance (NGN) | Notes & Flags |
|---------------|-------------|--------------------------|---------------|
| ...           |             |                          | 🔴 **Suspicious (Identity Change):** Bank name changed from 'Old Bank' to 'New Bank'. |
| ...           |             |                          | 🔴 **Suspicious (Duplicate ID):** Employee ID appears X times in the current period. |
| ...           |             |                          | **Significant Pay Change:** Describe the change (e.g., Housing increased by NXX). |
| ...           |             |                          | No significant variance. |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
4. Reconciliation of Gross Pay Variance

Summarize the drivers contributing to the total Gross Pay variance in a reconciliation table.

Generated markdown
| Driver                               | Count | Value Impact (NGN) |
|--------------------------------------|-------|--------------------|
| New Hires                            |       |                    |
| Departures                           |       |                    |
| Pay Changes (Continuing Employees)   |       |                    |
| Suspicious Anomalies (e.g., Duplicates) |       |                    |
| **Total Gross Pay Variance**         |       |                    |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
5. Conclusion & Recommendations

Provide clear, numbered, and actionable recommendations based on your findings. Prioritize critical issues.

🔴 URGENT: Investigate Duplicate Employee ID: Detail the specific employee and the risk of double payment.

🔴 URGENT: Verify Bank Detail Change: Detail the specific employee and the potential fraud risk.

Review Pay Increase Authorization: Specify the employee and the amount that needs verification.

Data Cleansing Protocol: Recommend a future action to prevent similar data integrity issues.

Final Instructions:

Format all monetary values with the Nigerian currency symbol and two decimal places (e.g., N5,200.00).

Maintain a professional, objective, and data-driven tone. Your primary goal is to act as a diligent analyst, highlighting not just the numbers but the underlying data quality issues and operational risks they represent.
"""

    systemprompt1= retrieved_template6.format(old=old,new=new)

    responseY = llmv.invoke([
        systemprompt1,
        HumanMessage(content="Please review")
    ])
    return responseY.content

systemprompt = """

You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets provided in JSON format:

"Previous Payroll Period": {old}

"Current Payroll Period": {new}

Your analysis must be meticulous, data-driven, and presented in a clear, professional Markdown report.

Analysis Instructions

You must follow these steps precisely:

1. Identify Employee Status:
Use employeeID._id as the unique identifier for each employee across both datasets. Classify each employee into one of the following categories:

Continuing: The employeeID._id exists in both the previous and current datasets.

New: The employeeID._id exists only in the current dataset.

Departed: The employeeID._id exists only in the previous dataset.

Suspicious: Flag any continuing employee as "Suspicious" if ANY of the following conditions are met. Comparison must be exact and case-sensitive.

fullname has changed.

employeeID.bankName has changed (e.g., "Zenith Bank" vs. "Zenith Banks").

employeeID.accountNumber has changed.

employeeID.phone has changed.

Suspicious (Duplicate ID): If an employeeID._id appears more than once within the payslips array of the Current Payroll Period, it must be flagged as a critical data integrity issue.

Note: If a file contains multiple top-level payroll objects, consolidate all payslips into a single list for each period before starting the analysis.

2. Calculate Monetary Variances:
For each employee and for the overall totals, compute the monetary variance (Current - Previous) in NGN for the following fields:

gross (Gross Pay)

tax (Tax)

pension (Employee Pension Contribution)

3. Identify Key Drivers of Variance:
Analyze the data to determine the root causes of any financial changes. Your analysis must explicitly connect variances to:

Headcount Changes: The financial impact of new hires and departures.

Pay Changes: Changes in gross pay for continuing employees.

Anomalies & Data Quality: The financial impact of suspicious records, especially duplicate entries.

Required Output Format (Markdown)

Generate the report using the exact structure and formatting below.

1. Executive Summary

Start with a headline figure: the total variance in Gross Pay.

State whether the variance is favorable (cost decrease) or unfavorable (cost increase).

Briefly summarize the primary drivers (e.g., headcount changes, significant pay adjustments, data anomalies).

2. Overall Payroll Summary

Provide a Markdown table comparing the aggregate values:

Generated markdown
| Metric        | Previous Period | Current Period | Variance (NGN) | Variance (%) |
|---------------|-----------------|----------------|----------------|--------------|
| Gross Pay     |                 |                |                |              |
| Total Tax     |                 |                |                |              |
| Total Pension |                 |                |                |              |

3. Detailed Variance Analysis
3.1 Headcount Changes

List new and departed employees and their financial impact.

Departed Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total Departures** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END

New Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total New Hires** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
3.2 Variances for Continuing & Suspicious Employees

Create a table for all continuing employees. Highlight significant changes and flag suspicious records.

Generated markdown
| Employee Name | Employee ID | Gross Pay Variance (NGN) | Notes & Flags |
|---------------|-------------|--------------------------|---------------|
| ...           |             |                          | 🔴 **Suspicious (Identity Change):** Bank name changed from 'Old Bank' to 'New Bank'. |
| ...           |             |                          | 🔴 **Suspicious (Duplicate ID):** Employee ID appears X times in the current period. |
| ...           |             |                          | **Significant Pay Change:** Describe the change (e.g., Housing increased by NXX). |
| ...           |             |                          | No significant variance. |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
4. Reconciliation of Gross Pay Variance

Summarize the drivers contributing to the total Gross Pay variance in a reconciliation table.

Generated markdown
| Driver                               | Count | Value Impact (NGN) |
|--------------------------------------|-------|--------------------|
| New Hires                            |       |                    |
| Departures                           |       |                    |
| Pay Changes (Continuing Employees)   |       |                    |
| Suspicious Anomalies (e.g., Duplicates) |       |                    |
| **Total Gross Pay Variance**         |       |                    |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
5. Conclusion & Recommendations

Provide clear, numbered, and actionable recommendations based on your findings. Prioritize critical issues.

🔴 URGENT: Investigate Duplicate Employee ID: Detail the specific employee and the risk of double payment.

🔴 URGENT: Verify Bank Detail Change: Detail the specific employee and the potential fraud risk.

Review Pay Increase Authorization: Specify the employee and the amount that needs verification.

Data Cleansing Protocol: Recommend a future action to prevent similar data integrity issues.

Final Instructions:

Format all monetary values with the Nigerian currency symbol and two decimal places (e.g., N5,200.00).

Maintain a professional, objective, and data-driven tone. Your primary goal is to act as a diligent analyst, highlighting not just the numbers but the underlying data quality issues and operational risks they represent.
"""



systemprompt11="""
 "Please perform a payroll variance analysis comparing two JSON files: {old} (the old payroll data) and {new} (the recent payroll data). The report should capture the following details:

New Employees: List all employees present in the new file but not in the old file, including their gross pay, net pay, and full account details (bank, account number, account name).

Salary Changes: Identify any employees present in both files whose gross, net, or charge amounts have changed. Specify the old and new values for each change.

Delisted Employees: List all employees present in the old file but not in the new file, including their last recorded gross pay, net pay, and full account details.

Changed in Account Details: For employees present in both files, identify any changes in their bank name, account number, or account name. Specify the old and new account details.

Any Other Significant Change: Provide a summary of the overall financial impact, including the total variance in gross payroll, net payroll, and charges between the old and new files.






"""
def aluke():
   return systemprompt


def get_payslips_from_json(json_file_path,desired_columns):
    # json_file_path is request.FILES.get('old') or request.FILES.get('new
    
    """
    Extracts payslips from a JSON file and returns a DataFrame with selected fields.
    
    Args:
        json_file_path (str): Path to the JSON file containing payroll data.
        
    Returns:
        pd.DataFrame: DataFrame containing the extracted payslips with selected fields.
    """
   
    data = json.load(json_file_path)
    if not isinstance(data, list):
     data = [data]
    #Notes
    all_payslips = []
    for payroll in data:
        payroll_id = payroll['_id']
        payslips = payroll.get('payslips', [])

        # Normalize payslips into a DataFrame
        df = pd.json_normalize(payslips)

        # Add payroll_id and payslip_id columns
        df['payroll_id'] = payroll_id
        df['employee_ID'] = df['_id']

        all_payslips.append(df)
    # Combine all into one DataFrame
    final_df = pd.concat(all_payslips, ignore_index=True)
    # desired_columns= desired_columns

    available_columns = [col for col in desired_columns if col in final_df.columns]
    # Trim safely using only available columns
    trimmed_df = final_df[available_columns]

    
    # Filter the final DataFrame
    # trimmed_df = final_df[desired_columns]
    json_output = trimmed_df.to_json(orient="records", indent=4)
    # # Save the final DataFrame to a CSV file
    # csv_file_path = r"C:\Users\Pro\Downloads\payslips_outputfullaboki.csv"
    # final_df.to_csv(csv_file_path, index=False)
        
    # df = pd.DataFrame(all_payslips)
    return json_output




        
# y = Prompt.objects.get(pk=1)  # Get the existing record
# retrieved_template1=y.response_prompt 
# response_prompt = retrieved_template1.format(
#     greeting=greeting,
#     ayula=state["messages"][-1].content,
#     attached_content=attached_content,  # Assuming no attached content for now
#     context=context,
#     pdf_text=pdf_text,
#     web_text=web_text,
#     query_answer=query_answer,

# )
# print("--- PROMPT ---",response_prompt)



# sys_msg = SystemMessage(content=response_prompt)

# model_with_structure = model.with_structured_output(Answer) 



# # Prepare key fields
# # key_fields = [line.strip() for line in importants.strip().splitlines()]
# key_fields=important()

# # Load raw JSON
# json_file_pathr = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\myapp\raw.json"
# with open(json_file_pathr, 'r') as f:
#     datar = json.load(f)

# # Normalize JSON and extract desired fields
# payslips_dfr = pd.json_normalize(datar["payslips"])
# initial_json = payslips_dfr[key_fields].to_json(orient="records", indent=4)



# # Load raw JSON

# json_file_patht = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\myapp\revised.json"
# with open(json_file_patht, 'r') as f:
#     datat = json.load(f)

# # Normalize JSON and extract desired fields
# payslips_dft =pd.json_normalize(datat["payslips"])
# treated_json = payslips_dft[key_fields].to_json(orient="records", indent=4)



# # Pass the JSON string to your function
# yemo = atb(initial_json,treated_json)
# print(yemo)