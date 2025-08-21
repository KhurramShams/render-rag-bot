from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
import logging
from langchain_core.prompts import ChatPromptTemplate
import hashlib
import pdfplumber
from io import BytesIO
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import json
from pathlib import Path



load_dotenv()

COMPANY_NAMESPACES = {
    "DigiRoam": "tenant_digiroam",
    "DigiCom": "tenant_digicom",
    "DigiTech": "tenant_digitech",
}

COMPANY_PROMPTS = {
    "DigiRoam": (
        "You are an customer support agent for Paragon RoamDigi. "
        "Always answer questions accurately and concisely using the provided context. "
        "If the context lacks specific details add some information from your site to fullfill user customer question. "
        "Keep responses short (2–3 sentences) and professional."
    ),
    "DigiCom": (
        "You are the customer support agent for Paragon DigiCom. "
        "Your role is to help users with accurate, context-based answers about Paragon DigiCom using the provided context. "
        "If the context lacks specific details add some information from your site to fullfill user customer question. "
        "Keep responses short, clear, and user-friendly."
    ),
    "DigiTech": (
        "You are an customer support agent for Paragon DigiTech. "
        "Use the provided context to answer questions about DigiTech’s services and products. "
        "If the context lacks specific details add some information from your site to fullfill user customer question. "
        "Keep answers focused, helpful, and limited to 2–3 sentences."
    )
}

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

def store_chunks_in_pinecone(chunks, embedding_function, company_id,index_name="rag-index", pdf_hash="unknown"):
    try:
        if company_id not in COMPANY_NAMESPACES:
            raise ValueError(f"Invalid company_id: {company_id}")

        namespace = COMPANY_NAMESPACES[company_id]
        print(namespace)
        
        # Always include company_id in metadata
        metadatas = [
            {"doc_hash": pdf_hash, "company_id": company_id, "chunk_id": i}
            for i in range(len(chunks))
        ]

        
        vector_store = PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embedding_function,
            index_name=index_name,
            namespace=namespace, 
            metadatas=metadatas
        )

        storage_dir = Path("./local_chunks")
        storage_dir.mkdir(exist_ok=True)
        file_path = storage_dir / f"{company_id}.json"

        if file_path.exists():
            existing = json.loads(file_path.read_text())
        else:
            existing = []

        existing.extend(chunks)
        file_path.write_text(json.dumps(existing, indent=2))

        logger.info(f"Stored {len(chunks)} chunks in Pinecone under namespace '{namespace}'")
        return vector_store
    except Exception as e:
        logger.error(f"Error storing embeddings in Pinecone: {str(e)}")
        raise

def validate_pdf(file_content) -> tuple[bool, str, str]:
    try:
        with pdfplumber.open(BytesIO(file_content)) as doc:
            page_count = len(doc.pages)
        
        if page_count > 10:
            return False, f"PDF has {page_count} pages. Maximum allowed is 10.", ""

        full_text = ""
        for page in doc.pages:
                text = page.extract_text() or ""
                full_text += text
        word_count = len(full_text.split())

        if word_count > 15000:
            return False, f"PDF has {word_count} words. Maximum allowed is 15,000.", ""

        return True, "PDF is valid.", full_text
    
    except Exception as e:
        return False, f"Error reading PDF: {str(e)}", ""

def process_pdf_and_split(file_content, chunk_size=1000, chunk_overlap=200):
    try:
        # Step 1: Read PDF with pdfplumber
        with pdfplumber.open(BytesIO(file_content)) as doc:
            full_text = ""
            for page in doc.pages:
                text = page.extract_text() or ""
                full_text += text
        # Step 2: Split using LangChain's RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?"]
        )

        chunks = splitter.split_text(full_text)
        return chunks
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")


def load_docs_for_company(company_id):
    file_path = Path(f"./local_chunks/{company_id}.json")
    if file_path.exists():
        return json.loads(file_path.read_text())
    return []

def query_llm_with_agent(query, embedding_function, openai_api_key, pdf_hash, namespace, top_k=3, extra_filter=None , history=[], company_id=''):
    
    try:

        system_prompt = COMPANY_PROMPTS.get(
            company_id,
            "You are a helpful customer suppport assistant. Answer accurately using only the provided context."
        )
        search_kwargs = {"k": top_k}

        filter_conditions = {"company_id": {"$eq": company_id}}

        if pdf_hash:
            filter_conditions["doc_hash"] = {"$eq": pdf_hash}

        search_kwargs["filter"] = filter_conditions
        
        vector_retriever = PineconeVectorStore(
            index_name="rag-index",
            embedding=embedding_function,
            namespace=namespace
        ).as_retriever(search_kwargs=search_kwargs)
        
        # Build conversation from history
        conversation_history = []
        for h in history:
            conversation_history.append((h["role"], h["content"]))
        
        
        docs_texts = load_docs_for_company(company_id)
        bm25_retriever = BM25Retriever.from_texts(docs_texts)

        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

        print(system_prompt)
        print(conversation_history)
        print("data", retriever)
        
        def search_documents(q: str) -> str:
            docs = retriever.invoke(q)
            if not docs:
                raise ValueError("No data found for this company and hash.")
            return "\n\n".join([doc.page_content for doc in docs])

        tools = [
            Tool.from_function(
                func=search_documents,
                name="search_documents",
                description="Useful for answering questions about uploaded PDF documents."
            )
        ]
        
        # Build the agent
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=openai_api_key,
            temperature=0.7,
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", f"Conversation so far:\n{conversation_history}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        try:
            response = agent_executor.invoke({"input": query})
        except ValueError as e:
            if "NO_CONTEXT_FOUND" in str(e):
                return "No relevant information found for this company or document."
            raise
        return response.get("output", "No response generated.")

    except Exception as e:
        return f"Error querying LLM agent: {str(e)}"

      
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
