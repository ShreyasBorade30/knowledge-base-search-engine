##Fixed

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path
import shutil
from rag_engine import RAGEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Base RAG API",
    description="API for document ingestion and question answering using RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

rag_engine = RAGEngine(groq_api_key=GROQ_API_KEY)

# Create upload directory
UPLOAD_DIR = Path("./data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    sources: Optional[List[str]] = None
    context_used: Optional[int] = None
    message: Optional[str] = None

# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Knowledge Base RAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload documents",
            "POST /query": "Query the knowledge base",
            "GET /stats": "Get knowledge base statistics",
            "DELETE /clear": "Clear knowledge base"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF or TXT)
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF and TXT files are supported"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest document
        result = rag_engine.ingest_document(str(file_path), file.filename)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base with a question
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = rag_engine.query(request.question, request.top_k)
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get knowledge base statistics
    """
    try:
        stats = rag_engine.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.delete("/clear")
async def clear_knowledge_base():
    """
    Clear all documents from the knowledge base
    """
    try:
        result = rag_engine.clear_knowledge_base()
        
        # Also clear uploaded files
        for file in UPLOAD_DIR.glob("*"):
            if file.is_file():
                file.unlink()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "groq_api_configured": bool(GROQ_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

