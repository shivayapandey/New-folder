# 3. Home.py (main page)
import streamlit as st

def main():
    st.set_page_config(
        page_title="RAG Pipeline Evaluation",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 RAG Pipeline Evaluation using BeyondLLM")
    
    st.markdown("""
    ## Welcome to the RAG Pipeline Evaluation Tool!
    
    This application helps you evaluate your RAG (Retrieval-Augmented Generation) pipeline using multiple metrics:
    
    1. **Document Indexing** 📚
       - Upload and process PDF documents
       - Create embeddings and vector store
       - Prepare documents for retrieval
    
    2. **Response Generation & Evaluation** 🎯
       - Generate responses using the indexed documents
       - Evaluate context relevancy
       - Measure answer relevancy
       - Calculate groundedness scores
    
    ### Getting Started
    1. Navigate to the **Document Indexing** page in the sidebar
    2. Upload your PDF documents and create the index
    3. Proceed to the **Response Evaluation** page
    4. Enter your GROQ API key and start generating responses
    
    ### Note
    Make sure you have your GROQ API key ready for the evaluation phase.
    """)

if __name__ == "__main__":
    main()