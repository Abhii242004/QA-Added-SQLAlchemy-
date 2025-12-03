import json
import time
import requests
import re
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

# NEW IMPORTS FOR SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from dotenv import load_dotenv
import os

# CRITICAL FIX for SQLite version incompatibility on Linux slim images
# This imports the modern, bundled pysqlite3 and forces the standard 'sqlite3'
# module to use it, satisfying ChromaDB's requirement (version >= 3.35.0).
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# --------------------------------------------------------------------------

from chromadb import PersistentClient, Settings
from fastembed.text.text_embedding import TextEmbedding

# --- Configuration and Initialization ---

# Load environment variables from .env file
load_dotenv()

# Access your API key
API_KEY = os.environ.get("GEMINI_KEY")
if not API_KEY:
    # Fallback/Placeholder message if running outside the intended environment
    print("Warning: GEMINI_KEY environment variable not found. Using empty string.")

# API Endpoint and Model
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
DB_DIR = "chroma_test_db"
COLLECTION_NAME = "qa_knowledge_base"

# Initialize FastAPI app
app = FastAPI(
    title="Monolithic QA Automation & RAG Service",
    description="Backend for AI-powered QA and test script generation with integrated SQL metadata tracking."
)

# --- SQLAlchemy Configuration and Model (New Relational DB) ---

# SQLite database file path for metadata
SQLALCHEMY_DATABASE_URL = "sqlite:///./uploaded_files.db"

# Create the SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} # Required for SQLite
)

# Base class for the model definitions
Base = declarative_base()

class UploadedFile(Base):
    """
    SQLAlchemy Model for storing metadata about ingested documents.
    """
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    # A unique identifier for the document, linking to ChromaDB's ID
    chroma_id = Column(String, unique=True, index=True, nullable=False) 
    document_type = Column(String, default="text") # e.g., 'text', 'html', 'pdf'
    source = Column(String, default="api_upload")
    content_snippet = Column(Text) # Store a small snippet for display
    char_count = Column(Integer)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)

# Create the database table(s) on startup
Base.metadata.create_all(bind=engine)

# SessionLocal class to create DB sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get a database session (used in endpoints)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- End SQLAlchemy Configuration ---

# --- Pydantic Schemas ---

class UploadDocsPayload(BaseModel):
    """Payload for ingesting new documents."""
    docs: List[str] = Field(..., description="List of text documents to ingest.")
    html: Optional[str] = Field(None, description="Optional raw HTML content to extract text from.")

class QueryPayload(BaseModel):
    """Payload for query-based operations."""
    query: str = Field(..., description="The user query or description.")

class SeleniumPayload(BaseModel):
    """Payload for generating Selenium scripts."""
    test_case: Dict[str, Any] = Field(..., description="The structured test case output from /generate_test_cases.")

class TestStep(BaseModel):
    """A single step in the structured test case."""
    step_number: int
    action: str = Field(..., description="The action to perform (e.g., 'Click', 'Enter text', 'Verify').")
    element_identifier: str = Field(..., description="CSS Selector, XPath, or human-readable description of the UI element.")
    test_data: Optional[str] = Field(None, description="Data to input (e.g., 'user@example.com', 'password123').")
    expected_result: str = Field(..., description="The expected outcome after performing the action.")

class GeneratedTestCase(BaseModel):
    """The structured output for a test case."""
    test_case_id: str = Field(..., description="A unique identifier for the test case, e.g., 'TC_LOGIN_001'.")
    title: str = Field(..., description="A concise title for the test case.")
    description: str = Field(..., description="A brief overview of the test objective.")
    priority: str = Field("Medium", description="Priority level: High, Medium, or Low.")
    steps: List[TestStep]

class DocumentMetadata(BaseModel):
    """Pydantic model for displaying document metadata."""
    id: int
    chroma_id: str
    document_type: str
    source: str
    char_count: int
    upload_timestamp: datetime
    content_snippet: str
    
# JSON Schema for Structured Generation (used in the API call)
TEST_CASE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "test_case_id": {"type": "STRING", "description": "A unique identifier for the test case, e.g., 'TC_PROMO_007'."},
        "title": {"type": "STRING"},
        "description": {"type": "STRING"},
        "priority": {"type": "STRING", "enum": ["High", "Medium", "Low"]},
        "steps": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "step_number": {"type": "INTEGER"},
                    "action": {"type": "STRING"},
                    "element_identifier": {"type": "STRING"},
                    "test_data": {"type": "STRING"},
                    "expected_result": {"type": "STRING"}
                },
                "propertyOrdering": ["step_number", "action", "element_identifier", "test_data", "expected_result"]
            }
        }
    },
    # CRITICAL FIX: Ensure test_case_id is explicitly ordered first to remind the LLM
    "propertyOrdering": ["test_case_id", "title", "description", "priority", "steps"] 
}

# --- ChromaDB Setup with fastembed ---

def get_chroma_client_and_collection():
    """Initializes and returns the ChromaDB client and collection."""
    try:
        model_name = TextEmbedding.list_supported_models()[0]['model']
        print(f"ChromaDB will use fastembed model: {model_name}")
        
        client = PersistentClient(
            path=DB_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
            )
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        return client, collection
    except Exception as e:
        print(f"Error initializing ChromaDB or fastembed: {e}")
        raise RuntimeError(f"Database initialization failed: {e}")

# Global variables for DB (initialized lazily)
chroma_client = None
knowledge_base = None

@app.on_event("startup")
def startup_db_client():
    """Initialize the database client on application startup."""
    global chroma_client, knowledge_base
    if not chroma_client:
        chroma_client, knowledge_base = get_chroma_client_and_collection()
    print(f"ChromaDB initialized. Knowledge base size: {knowledge_base.count()} documents.")


# --- Gemini API Helper Function (with retry logic) ---

def call_gemini_api_with_retry(payload: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
    """Handles API calls with exponential backoff and improved key check."""
    if not API_KEY:
        # Improved error message if the key is missing
        raise HTTPException(
            status_code=500, 
            detail="Gemini API Key is missing. Please ensure the 'GEMINI_KEY' environment variable is set."
        )
        
    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={API_KEY}",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error on attempt {attempt + 1}: {e}")
            if response.status_code == 429 and attempt < max_retries - 1:
                delay = 2 ** attempt
                time.sleep(delay)
                continue
            # For 400 errors (which often indicate an invalid API key or payload issue)
            if response.status_code >= 400 and response.status_code < 500:
                    raise HTTPException(status_code=response.status_code, detail=f"Gemini API Error: Check API Key and Payload. Response: {response.text}")
            
            raise HTTPException(status_code=response.status_code, detail=f"Gemini API Error: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request Exception on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                time.sleep(delay)
                continue
            raise HTTPException(status_code=500, detail="Gemini API request failed.")
    
    raise HTTPException(status_code=500, detail="Gemini API failed after multiple retries.")


# --- Endpoints ---

@app.get("/")
def read_root():
    """Basic health check and environment info."""
    db_count = knowledge_base.count() if knowledge_base else "N/A (ChromaDB not initialized)"
    return {
        "status": "ok",
        "message": "QA Automation Service Running with ChromaDB and SQLAlchemy",
        "chroma_document_count": db_count,
        "model_used": MODEL_NAME
    }

---

## 💾 Document Ingestion and Metadata Tracking

@app.post("/upload_docs")
def upload_docs(payload: UploadDocsPayload, db: Session = Depends(get_db)):
    """
    Ingests raw text and/or HTML content into the ChromaDB vector store 
     AND saves metadata to the SQL database using SQLAlchemy.
    """
    global knowledge_base

    if not knowledge_base:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized.")

    documents_to_embed = []
    file_metadata_list = []
    current_chroma_count = knowledge_base.count()
    id_counter = 0

    # 1. Process documents and gather metadata
    if payload.docs:
        for doc in payload.docs:
            # Generate a unique Chroma ID based on the count at the start of the transaction
            chroma_id = f"doc_{current_chroma_count + id_counter}"
            documents_to_embed.append(doc)
            file_metadata_list.append({
                "chroma_id": chroma_id,
                "document_type": "text",
                "source": "raw_text_list",
                "content_snippet": doc[:100].replace('\n', ' ') + "...",
                "char_count": len(doc)
            })
            id_counter += 1

    if payload.html:
        try:
            soup = BeautifulSoup(payload.html, 'lxml')
            # Extract main readable text, ignoring scripts, styles, etc.
            text_content = soup.get_text(separator='\n', strip=True)
            if text_content:
                chroma_id = f"doc_{current_chroma_count + id_counter}"
                documents_to_embed.append(f"HTML Content:\n{text_content}")
                file_metadata_list.append({
                    "chroma_id": chroma_id,
                    "document_type": "html",
                    "source": "html_upload",
                    "content_snippet": text_content[:100].replace('\n', ' ') + "...",
                    "char_count": len(text_content)
                })
                id_counter += 1
        except Exception as e:
            print(f"Error processing HTML: {e}")
            raise HTTPException(status_code=400, detail="Invalid HTML content provided.")

    if not documents_to_embed:
        return {"status": "warning", "message": "No valid documents or text extracted."}
    
    # 2. Store Metadata in SQL Database (Transaction Start)
    try:
        new_files = [
            UploadedFile(
                chroma_id=meta["chroma_id"],
                document_type=meta["document_type"],
                source=meta["source"],
                content_snippet=meta["content_snippet"],
                char_count=meta["char_count"]
            ) 
            for meta in file_metadata_list
        ]
        db.add_all(new_files)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"SQLAlchemy error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save document metadata to SQL DB.")

    # 3. Store Embeddings in ChromaDB (Vector DB)
    try:
        ids = [meta["chroma_id"] for meta in file_metadata_list]
        knowledge_base.add(
            documents=documents_to_embed,
            ids=ids,
            # Chroma metadata uses the same ID for easy lookup
            metadatas=[{"source": meta["source"], "id": meta["chroma_id"]} for meta in file_metadata_list] 
        )
        return {
            "status": "success",
            "message": "Documents ingested into the knowledge base and metadata saved.",
            "chunks_added": len(documents_to_embed),
            "chroma_db_count_after": knowledge_base.count()
        }
    except Exception as e:
        print(f"ChromaDB add error: {e}")
        # NOTE: A more complex system would also remove the SQL entry here if Chroma failed.
        raise HTTPException(status_code=500, detail="Failed to add documents to vector store.")

---

## 📚 Document Listing (SQL Query)

@app.get("/list_docs", response_model=List[DocumentMetadata])
def list_docs(db: Session = Depends(get_db)):
    """
    Retrieves a list of all ingested document metadata from the SQL database.
    """
    # Query all records from the UploadedFile table
    files = db.query(UploadedFile).all()
    # Pydantic validation handles the conversion to the desired output format
    return files

---

## 🤖 AI Generation Endpoints

@app.post("/generate_test_cases", response_model=GeneratedTestCase)
def generate_test_cases(payload: QueryPayload):
    """
    Generates a structured test case (JSON) based on the user's query and RAG context.
    """
    global knowledge_base

    if not knowledge_base:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    query = payload.query

    # 1. RAG Retrieval Step: Find relevant context
    try:
        results = knowledge_base.query(
            query_texts=[query],
            n_results=3, # Retrieve top 3 relevant documents
            include=['documents']
        )
        context = "\n---\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else "No relevant context found."
    except Exception as e:
        print(f"ChromaDB query error: {e}")
        raise HTTPException(status_code=500, detail="RAG retrieval failed.")

    # 2. LLM Generation Step (Structured Output)
    
    # System Instruction: Guide the model's persona and output format
    system_prompt = (
        "You are an expert QA Engineer. Your task is to generate a comprehensive, structured test case "
        "in the exact JSON format provided in the schema. **You must include the 'test_case_id' field**, "
        "using a descriptive format like 'TC_FEATURE_XXX' (e.g., 'TC_LOGIN_001'). "
        "Use the provided RAG Context to inform the precise details for UI interactions "
        "(e.g., specific form fields, expected messages). Ensure 'element_identifier' is a precise description."
    )
    
    # User Prompt: The specific instruction for the LLM
    user_query = (
        f"Based on the following knowledge context, generate a detailed test case for the user request:\n\n"
        f"USER REQUEST: {query}\n\n"
        f"KNOWLEDGE CONTEXT:\n{context}"
    )

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": TEST_CASE_SCHEMA
        }
    }

    api_response = call_gemini_api_with_retry(payload)
    
    try:
        # The result text contains the JSON string
        json_string = api_response['candidates'][0]['content']['parts'][0]['text']
        test_case_data = json.loads(json_string)
        
        # Validate the generated data against the Pydantic model
        return GeneratedTestCase(**test_case_data)
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        # Include the raw text of the response in the error detail for easier debugging
        raw_text_snippet = json_string[:500] if 'json_string' in locals() else 'Response text not available.'
        print(f"Failed to parse or validate JSON response: {e}")
        print(f"Raw API response JSON: {api_response}")
        raise HTTPException(status_code=500, detail=f"LLM failed to return valid structured JSON. Validation Error: {e}. Raw LLM Output start: {raw_text_snippet}")


@app.post("/generate_selenium")
def generate_selenium(payload: SeleniumPayload):
    """
    Converts a structured test case (JSON) into a runnable Python Selenium script.
    Uses Google Search grounding for up-to-date Selenium syntax/best practices.
    """
    # 1. Format the JSON test case into a readable string
    test_case_str = json.dumps(payload.test_case, indent=2)

    # 2. LLM Generation Step (Code Generation with Grounding)
    
    # System Instruction: Guide the model for Python code generation and best practices
    system_prompt = (
        "You are an expert Python programmer specializing in Selenium WebDriver automation. "
        "Your task is to take a structured test case (provided as a JSON object) and convert it "
        "into a complete, runnable Python script using the `selenium` library. "
        "The script must be self-contained and ready to execute. "
        "For 'element_identifier' fields, assume they are descriptive labels (e.g., 'Login button', 'Email input field'). "
        "You must use the `By.CSS_SELECTOR` or `By.ID` where possible, or use explicit waits with descriptive element IDs/names if available. "
        "Use `webdriver_manager` to automatically set up the Chrome driver. "
        "Provide only the complete, executable Python code block."
    )
    
    # User Prompt: The specific instruction for the LLM
    user_query = (
        f"Convert the following structured test case into a complete, runnable Python script "
        f"that uses Selenium WebDriver. Use best practices for locating elements and handling waits. "
        f"The test case is:\n\n{test_case_str}"
    )
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}] # Enable grounding for current best practices
    }

    api_response = call_gemini_api_with_retry(payload)
    
    try:
        script_text = api_response['candidates'][0]['content']['parts'][0]['text']
        
        # --- FIX: Use regex to extract content between the first ```python and the next ``` ---
        match = re.search(r'```python\s*(.*?)\s*```', script_text, re.DOTALL)
        
        if match:
            script = match.group(1).strip()
        else:
            # If no code fence is found, assume the entire output is the script
            script = script_text.strip()
            if script.lower().startswith("error"):
                 raise ValueError("LLM generation failed or returned an error message.")
            
        return {"script": script}
    except KeyError:
        # If any part of the expected structure is missing
        raise HTTPException(status_code=500, detail="LLM failed to return a valid script structure.")
    except Exception as e:
        print(f"Script extraction failed: {e}. Raw response: {script_text[:100]}...")
        # If extraction or parsing fails, return the raw text for debugging
        return {"script": f"Error: Failed to generate script. Raw LLM output: {script_text}", "detail": str(e)}
