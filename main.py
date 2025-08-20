from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status, Request
from pathlib import Path                   
from fastapi.params import Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pdf_utils import (
    load_environment, initialize_pinecone, initialize_embeddings, initialize_llm,
    validate_pdf, process_pdf_and_split, store_chunks_in_pinecone,
    get_pdf_hash, is_document_already_indexed, query_llm_with_agent
)
from langchain_pinecone import PineconeVectorStore
import logging
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
import secrets
from message_data import save_chat_history,get_chat_history

load_dotenv()

# --- FastAPI app setup
app = FastAPI(
    title="RAG API",
    description="RAG API for customer support agent.",
    version="0.1",
    docs_url=None,
    redoc_url=None,           # disables ReDoc UI
    openapi_url="/rag-openapi.json"
)
security = HTTPBasic()

# ✅ Fixed credentials
DOCS_USERNAME = os.getenv("DOCS_USERNAME")
DOCS_PASSWORD = os.getenv("DOCS_PASSWORD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your WordPress domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)  

COMPANY_NAMESPACES = {
    "DigiRoam": "tenant_digiroam",
    "DigiCom": "tenant_digicom",
    "DigiTech": "tenant_digitech",
}


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

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, DOCS_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, DOCS_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

# ✅ Custom protected docs route
@app.get("/api-endpoint-links", include_in_schema=False)
async def custom_swagger_ui(_: bool = Depends(verify_credentials)):
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Protected API Docs"
    )
@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API"}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), company_id: str = Form(...)):
    
    if company_id not in COMPANY_NAMESPACES:
        raise HTTPException(status_code=400, detail="Invalid company ID")

    namespace = COMPANY_NAMESPACES[company_id]
    
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
            company_id=company_id,
            pdf_hash=pdf_hash
        )

        return {"stored": True, "hash": pdf_hash, "msg": "Stored embeddings in Pinecone.", "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing embeddings: {e}")

class AskRequest(BaseModel):
    company_id: str
    question: str
    session_id: str   # <-- new field for memory
    hash: Optional[str] = None
    
    
@app.post("/ask/")
async def ask(request: AskRequest):
    company_id = request.company_id
    pdf_hash = request.hash
    session_id = request.session_id
    question = request.question.strip()
    
    
    filter_conditions = {}  # Always defined
     
    if request.hash:
        filter_conditions  = {"doc_hash": {"$eq": request.hash}}

    if company_id not in COMPANY_NAMESPACES:
        raise HTTPException(status_code=400, detail="Invalid company ID")
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    namespace = COMPANY_NAMESPACES[request.company_id]
    print(namespace)
    
    filter_conditions = {
        "company_id": {"$eq": request.company_id}
    }
    if pdf_hash:
        filter_conditions["doc_hash"] = {"$eq": pdf_hash}
    
    # load message history
    history = get_chat_history (session_id)
    
    # Step 1: Check if hash exists in Pinecone
    try:
        index = pc.Index("rag-index")
        result = index.query(
            vector=[0.0] * 1536, 
            top_k=1,
            namespace=namespace,
            filter=filter_conditions,
            )
        if len(result.matches) == 0:
                raise HTTPException(status_code=400, detail="No Data.")
    except Exception as e:
        logger.error(f"API Hash Invalid / No Data found for API Hash : {e}")
        raise HTTPException(status_code=500, detail=f"API Hash Invalid Or Company ID Invalid")

    # Step 2: Load Pinecone vector store (real-time)
    try:
        new_vector_store = PineconeVectorStore(
            index_name="rag-index",
            embedding=embedding_function,
            namespace=namespace,
        )
    except Exception as e:
        logger.error(f"Vector store error: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

    # Step 3: Query with the agent
    try:
        answer = query_llm_with_agent(
            query=question,
            embedding_function=embedding_function,
            openai_api_key=OPENAI_API_KEY,
            pdf_hash=pdf_hash,
            namespace=namespace,
            top_k=5,
            extra_filter={"company_id": {"$eq": request.company_id}},  # New param
            history=history
        )
        
        save_chat_history(session_id, "user", question)
        save_chat_history(session_id, "assistant", answer)
        
        return {"answer": answer}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")


class DeleteRequest(BaseModel):
    hash: Optional[str] = None
    company_id: Optional[str] = None

@app.delete("/delete_pdf_vectors/")
async def delete_pdf_vectors(request: DeleteRequest):
    
    pdf_hash = request.hash.strip() if request.hash else None
    company_id = request.company_id.strip() if hasattr(request, "company_id") and request.company_id else None

    if not pdf_hash and not company_id:
        raise HTTPException(status_code=400, detail="You must provide either a pdf_hash or a company_id.")

    try:
        index = pc.Index("rag-index")

        # Case 1: Delete vectors for a specific PDF
        if pdf_hash:
            namespace = None
            if company_id:
                if company_id not in COMPANY_NAMESPACES:
                    raise HTTPException(status_code=400, detail="Invalid company ID")
                namespace = COMPANY_NAMESPACES[company_id]

            result = index.query(
                vector=[0.0] * 1536,  # dummy vector for filtering
                top_k=1,
                namespace=namespace,
                filter={"doc_hash": {"$eq": pdf_hash}}
            )
            if len(result.matches) == 0:
                raise HTTPException(status_code=400, detail="API Hash Invalid / No Data found for API Hash")

            index.delete(
                delete_all=False,
                namespace=namespace,
                filter={"doc_hash": {"$eq": pdf_hash}}
            )
            logger.info(f"Deleted vectors for pdf_hash: {pdf_hash} in namespace: {namespace or 'default'}")
            return {"deleted": True, "hash": pdf_hash, "msg": "PDF vectors deleted successfully."}

        # Case 2: Delete all vectors for a company
        elif company_id:
            if company_id not in COMPANY_NAMESPACES:
                raise HTTPException(status_code=400, detail="Invalid company ID")
            
            namespace = COMPANY_NAMESPACES[company_id]

            index.delete(
                delete_all=True,  # deletes everything in the namespace
                namespace=namespace
            )
            logger.info(f"Deleted ALL vectors for company_id: {company_id} (namespace: {namespace})")
            return {"deleted": True, "company_id": company_id, "msg": "All company vectors deleted successfully."}

    except Exception as e:
        logger.error(f"Error deleting vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting vectors: {e}")



@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"}
    )
    
