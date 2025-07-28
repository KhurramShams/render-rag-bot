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
from io import BytesIO
from langchain.agents import AgentExecutor, tool, create_openai_functions_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
import requests
from bs4 import BeautifulSoup

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

def create_rag_prompt_template():
    template = """
You are an customer support chatbot for Paragon Digicom Software Company. Use the context to answer the question. If the context lacks specific details add some information from your site to fullfill user customer question.

Context:
{context}

User Question:
{query}

Answer:
"""
    return ChatPromptTemplate.from_template(template)

def query_llm_with_agent(query, vector_store, openai_api_key, pdf_hash, top_k=5):
    try:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k, "filter": {"doc_hash": {"$eq": pdf_hash}}}
        )
        def search_documents(q: str) -> str:
            """Search the PDF for relevant content"""
            docs = retriever.invoke(q)
            if not docs:
                return "No relevant information found in the document."
            return "\n\n".join([doc.page_content for doc in docs])

        # Step 2: Fetch homepage content (always used)
        def fetch_paragon_homepage() -> str:
            url = "https://paragondigicom.com/"
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator="\n").strip()
                return text[:2000] + "..."
            except Exception as e:
                return f"Error fetching homepage: {str(e)}"

        homepage_content = fetch_paragon_homepage()

        tools = [
            Tool.from_function(
                func=search_documents,
                name="search_documents",
                description="Useful for answering questions about the uploaded PDF document."
            )
        ]

        # Build the agent
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=openai_api_key,
            temperature=0.7,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a customer support agent for Paragon Digicom Software Company. Use the context to answer the question. If the context lacks specific details add some information from your site to fullfill user customer question. Keep your answers concise and focused. Limit responses to 2-3 sentences unless necessary."),
           ("user", "{input}\n\nCompany Website:\n" + homepage_content),
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
        response = agent_executor.invoke({"input": query})
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
