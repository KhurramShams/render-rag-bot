from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import hashlib
import pdfplumber

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables and return API keys."""
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        raise ValueError("Missing API keys. Please set PINECONE_API_KEY and OPENAI_API_KEY.")
    return PINECONE_API_KEY, OPENAI_API_KEY

def initialize_pinecone(api_key, index_name="rag-index"):
    """Initialize Pinecone client and create index if it doesn't exist."""
    try:
        pc = Pinecone(api_key=api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Created index: {index_name}")
        else:
            logger.info(f"Index {index_name} already exists")
        return pc
    except Exception as e:
        logger.error(f"Error initializing Pinecone client: {str(e)}")
        raise

def initialize_embeddings(api_key):
    """Initialize OpenAI embeddings."""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def initialize_llm(api_key):
    """Initialize ChatOpenAI LLM."""
    try:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0.7
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

# --- helper to ensure the correct fitz/PyMuPDF is loaded
# def _get_pymupdf():
#     """
#     Return the real PyMuPDF module. If the wrong 'fitz' package is installed,
#     raise an ImportError with a helpful hint.
#     """
#     try:
#         import fitz  # PyMuPDF installs under the name 'fitz'
#         if not hasattr(fitz, "open"):          # wrong package gives this away
#             raise ImportError(
#                 "Found a stub 'fitz' package without 'open()'. "
#                 "Uninstall it and install PyMuPDF:  pip uninstall -y fitz && pip install --upgrade PyMuPDF"
#             )
#         return fitz
#     except ModuleNotFoundError:
#         raise ImportError(
#             "PyMuPDF not installed. Install it with:  pip install PyMuPDF"
#         )


def store_chunks_in_pinecone(chunks, embedding_function, index_name="rag-index", pdf_hash="unknown"):
    try:
        metadatas = [{"doc_hash": pdf_hash, "chunk_id": i} for i in range(len(chunks))]
        vector_store = PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embedding_function,
            index_name=index_name,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(chunks)} chunks in Pinecone")
        return vector_store
    except Exception as e:
        logger.error(f"Error storing embeddings in Pinecone: {str(e)}")
        raise

def validate_pdf(file_content) -> tuple[bool, str, str]:
    try:
        # fitz = _get_pymupdf() 
        with pdfplumber.open(file_content) as doc:
            page_count = len(doc.pages)
        
        if page_count > 5:
            return False, f"PDF has {page_count} pages. Maximum allowed is 5.", ""

        full_text = ""
        for page in doc.pages:
                text = page.extract_text() or ""
                full_text += text
        word_count = len(full_text.split())

        if word_count > 10000:
            return False, f"PDF has {word_count} words. Maximum allowed is 10,000.", ""

        return True, "PDF is valid.", full_text
    
    except Exception as e:
        return False, f"Error reading PDF: {str(e)}", ""

def process_pdf_and_split(file_content, chunk_size=500, chunk_overlap=50):
    try:
        # Step 1: Read PDF with PyMuPDF
        with pdfplumber.open(file_content) as doc:
            full_text = ""
            for page in doc.pages:
                text = page.extract_text() or ""
                full_text += text
        # Step 2: Split using LangChain's RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?"]
        )

        chunks = splitter.split_text(full_text)
        return chunks
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

def create_rag_prompt_template():
    template = """
You are a helpful assistant. Use only the following context to answer the user's question.
If the answer cannot be found in the context, respond with "I don't know based on the provided information."

Context:
{context}

User Question:
{query}

Answer:
"""
    return ChatPromptTemplate.from_template(template)

def query_llm_with_rag(query, vector_store, llm, top_k=5):
    try:
        # Retrieve relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

        retrieved_docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."
        
        # Create prompt and chain
        prompt_template = create_rag_prompt_template()

        chain = prompt_template | llm | StrOutputParser()
        
        # Run the chain
        response = chain.invoke({"query": query, "context": context})
        return response.strip()
    except Exception as e:
        return f"Error querying LLM: {str(e)}"
      
def get_pdf_hash(file_bytes:bytes)->str:
    return hashlib.sha256(file_bytes).hexdigest()

def is_document_already_indexed(index, pdf_hash):
    try:
        # Use metadata filter to search by doc_hash
        results = index.query(
            vector=[0.0] * 1536,
            top_k=1,
            filter={"doc_hash": {"$eq": pdf_hash}}
        )
        return len(results.matches) > 0
    except Exception as e:
        print(f"Error checking existing doc: {e}")
        return False
