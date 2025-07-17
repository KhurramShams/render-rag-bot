from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from pathlib import Path                           # real filesystem Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pdf_utils import (
    load_environment, initialize_pinecone, initialize_embeddings, initialize_llm,
    validate_pdf, process_pdf_and_split, store_chunks_in_pinecone,
    get_pdf_hash, is_document_already_indexed, query_llm_with_rag
)
from langchain_pinecone import PineconeVectorStore
import logging

# --- FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your WordPress domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load environment and initialize global resources
load_dotenv()
try:
    PINECONE_API_KEY, OPENAI_API_KEY = load_environment()
    pc = initialize_pinecone(PINECONE_API_KEY)
    embedding_function = initialize_embeddings(OPENAI_API_KEY)
    llm = initialize_llm(OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Startup error: {e}")
    raise

# --- In-memory cache for vector stores
# --- In-memory cache for vector stores (limited to 5 entries)
vector_store_cache = {}
MAX_CACHE_SIZE = 5

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API"}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a PDF.")

    file_bytes = await file.read()
    pdf_hash = get_pdf_hash(file_bytes)

    # Save PDF to ./documents/
    documents_dir = Path("./documents")
    documents_dir.mkdir(exist_ok=True)
    file_path = documents_dir / file.filename
    with file_path.open("wb") as f:
        f.write(file_bytes)

    # Check if already indexed
    try:
        index = pc.Index("rag-index")
        already_indexed = is_document_already_indexed(index, pdf_hash)
    except Exception as e:
        logger.error(f"Pinecone error: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone error: {e}")

    if already_indexed:
        return {"stored": False, "hash": pdf_hash, "msg": "Document already indexed.", "chunks": 0}

    # Validate PDF
    is_valid, msg, extracted_text = validate_pdf(file_bytes)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    # Chunk and store
    try:
        chunks = process_pdf_and_split(file_bytes)
        vector_store = store_chunks_in_pinecone(
            chunks=chunks,
            embedding_function=embedding_function,
            pdf_hash=pdf_hash
        )
        
        # Limit cache size
        if len(vector_store_cache) >= MAX_CACHE_SIZE:
            vector_store_cache.pop(next(iter(vector_store_cache)))
        vector_store_cache[pdf_hash] = vector_store

        return {"stored": True, "hash": pdf_hash, "msg": "Stored embeddings in Pinecone.", "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing embeddings: {e}")

class AskRequest(BaseModel):
    hash: str
    question: str

@app.post("/ask/")
async def ask(request: AskRequest):
    pdf_hash = request.hash
    question = request.question.strip()
    if not pdf_hash:
        raise HTTPException(status_code=400, detail="Missing PDF hash.")
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Try cache, else load from Pinecone
    vector_store = vector_store_cache.get(pdf_hash)
    if not vector_store:
        try:
            vector_store = PineconeVectorStore(
                index_name="rag-index",
                embedding=embedding_function,
                namespace=None
            )
            vector_store_cache[pdf_hash] = vector_store
        except Exception as e:
            logger.error(f"Vector store error: {e}")
            raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

    try:
        answer = query_llm_with_rag(question, vector_store, pdf_hash, llm)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"}
    )
