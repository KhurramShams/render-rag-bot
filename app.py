import streamlit as st
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pdf_utils import validate_pdf
from pdf_utils import process_pdf_and_split
from langchain.chains import RetrievalQA
from pdf_utils import load_environment, initialize_pinecone, initialize_embeddings, initialize_llm, store_chunks_in_pinecone, query_llm_with_rag, get_pdf_hash, is_document_already_indexed
import logging


load_dotenv()

# Define API keys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    PINECONE_API_KEY, OPENAI_API_KEY = load_environment()
except Exception as e:
    st.error(f"Error loading environment: {str(e)}")
    logger.error(f"Error loading environment: {str(e)}")
    st.stop()

# Initialize Pinecone
try:
    pc = initialize_pinecone(PINECONE_API_KEY)
    st.success("Pinecone client initialized successfully")
except Exception as e:
    st.error(f"Error initializing Pinecone: {str(e)}")
    st.stop()

# Initialize embeddings
try:
    embedding_function = initialize_embeddings(OPENAI_API_KEY)
except Exception as e:
    st.error(f"Error initializing embeddings: {str(e)}")
    st.stop()

# Initialize LLM
try:
    llm = initialize_llm(OPENAI_API_KEY)
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.stop()

# Initialize PineconeVectorStore
try:
    vector_store = PineconeVectorStore(
    index_name="rag-index",
    embedding=embedding_function
    )
except Exception as e:
    st.error(f"Error initializing DataBase: {str(e)}")
    st.stop()

#------------------------------------- Streamlit Ui Design
Display=True
st.set_page_config(page_title="Your RAG Assistant", page_icon=":material/smart_toy:",layout="centered")

if Display:
    # Desging App
    st.title("Your RAG Assistant v.0.5")
    st.write("(Beta Version)")
    st.divider()
    
    file = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=False
    )
           
    # Show validation result
    if file:

        # Reset file pointer after validation
        file.seek(0)
        file_content = file.read()

        pdf_hash=get_pdf_hash(file_content)

        if not is_document_already_indexed(pc.Index("rag-index"), pdf_hash):

        # Store PDF validation result
            if file and "pdf_validated" not in st.session_state:
                is_valid, msg, extracted_text = validate_pdf(file_content)
                st.session_state.pdf_validated = is_valid
                st.session_state.pdf_msg = msg
                st.session_state.pdf_text = extracted_text if is_valid else ""

            if st.session_state.pdf_validated:
                st.success("✅ " + st.session_state.pdf_msg)
                chunks = process_pdf_and_split(file_content)
                print(len(chunks))
                if st.checkbox("🔍 View Chunks for Debugging"):
                    for i, c in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.code(c[:500], language="markdown")
                try:
                    vector_store=store_chunks_in_pinecone(chunks=chunks,embedding_function=embedding_function, pdf_hash=pdf_hash)
                    st.success("✅ Successfully stored embeddings in Pinecone.")
                    store=False
                except Exception as e:
                    st.error(f"❌ Error storing embeddings in Pinecone: {str(e)}")
            else:
                st.error("❌ Error in pdf validation." + st.session_state.pdf_msg)
        else:
            st.success("✅ Document already indexed.")
            

    if file and st.session_state.pdf_validated:
        question = st.text_area("Ask me! (Max 200 characters)", max_chars=200)

        if st.button("Submit", help="Click to get an answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking... Please wait ⏳"):
                    try:
                        answer=query_llm_with_rag(question,vector_store,llm)
                        st.subheader("Answer:")
                        st.write(answer)
                        st.divider()
                    except Exception as e:
                        st.error(f"❌ Error retrieving data from Pinecone: {str(e)}")
                    
    else:
        st.info("Upload a valid PDF to ask questions.")
