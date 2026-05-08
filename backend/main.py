import uuid
from typing import List, Dict
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_processor import extract_text
from chunking import chunk_text
from qa_engine import get_best_answer
from simplifier import simplify_answer

# In-memory session store: session_id -> list of text chunks
session_store: Dict[str, List[str]] = {}
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")

app = FastAPI(title="Document QA Backend")

# Configure CORS to allow requests from the Streamlit frontend (default port 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    session_id: str
    question: str

class AskResponse(BaseModel):
    answer: str
    simplified_answer: str
    score: float
    source_chunk_index: int
    source_text: str

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...)
):
    """
    Upload one or more PDF files, extract text, chunk it, and store in session.
    Returns status and total number of chunks created.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    full_text = ""
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            # You could also allow other types, but per spec we handle PDFs
            continue
        # Read file bytes
        file_bytes = await file.read()
        # Extract text using pdf_processor
        text = extract_text(file_bytes)
        full_text += "\n\n" + text  # Separate documents with newlines
    
    if not full_text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No text could be extracted from the uploaded PDF(s). "
                "If the file is scanned or image-only, configure OPENROUTER_API_KEY "
                "to enable OCR fallback in the backend."
            ),
        )
    
    # Chunk the combined text
    chunks = chunk_text(full_text)
    if not chunks:
        chunks = [full_text.strip()]
    
    # Store in session
    session_store[session_id] = chunks
    
    return {
        "status": "ok",
        "total_chunks": len(chunks)
    }

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answer a question based on chunks stored in the session.
    Returns answer, simplified answer, confidence, and source info.
    """
    session_id = request.session_id
    question = request.question
    
    # Retrieve chunks for the session
    chunks = session_store.get(session_id)
    if chunks is None:
        return AskResponse(
            answer="Please upload documents first",
            simplified_answer="Please upload documents first",
            score=0.0,
            source_chunk_index=-1,
            source_text=""
        )
    
    # Get best answer from QA engine
    best = get_best_answer(question, chunks)
    
    # If no answer found or score is too low
    if best["answer"] is None or best["score"] < 0.1:
        return AskResponse(
            answer="Answer not found in documents",
            simplified_answer="Answer not found in documents",
            score=best["score"],
            source_chunk_index=best["chunk_index"],
            source_text=best["source_text"] if best["source_text"] else ""
        )
    
    # Simplify the answer
    simplified = simplify_answer(best["answer"])
    
    return AskResponse(
        answer=best["answer"],
        simplified_answer=simplified,
        score=best["score"],
        source_chunk_index=best["chunk_index"],
        source_text=best["source_text"]
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
