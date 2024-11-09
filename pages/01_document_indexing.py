# 1. pages/01_document_indexing.py
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_core.documents import Document
import tempfile

def index_documents():
    st.title("Document Indexing for RAG Pipeline")

    # Sidebar for uploading documents
    st.sidebar.header("Upload Documents for Indexing")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Documents", 
        type="pdf", 
        accept_multiple_files=True
    )

    # Index documents when button is pressed
    if st.button("Index Documents"):
        if uploaded_files:
            with st.spinner("Loading and processing documents..."):
                # Load and process PDF documents
                docs = []
                for uploaded_file in uploaded_files:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file.seek(0)
                        # Read PDF
                        pdf_reader = PdfReader(tmp_file.name)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        if text.strip():  # Only add non-empty documents
                            docs.append(Document(page_content=text))

                if not docs:
                    st.error("No valid text content found in the uploaded PDFs.")
                    return

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=756,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                )
                chunks = text_splitter.split_documents(docs)

                try:
                    # Load embeddings and create vector store
                    embeddings = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-base-en-v1.5",
                        model_kwargs={'device': 'cpu'}
                    )
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    
                    # Save the vector store in session for later retrieval
                    st.session_state['vectorstore'] = vectorstore
                    st.success("✅ Indexing complete! You may now proceed to the response generation and evaluation page.")
                    
                    # Display some statistics
                    st.info(f"""
                    📊 Indexing Statistics:
                    - Number of documents processed: {len(docs)}
                    - Number of chunks created: {len(chunks)}
                    - Average chunk size: {sum(len(chunk.page_content) for chunk in chunks)/len(chunks):.0f} characters
                    """)
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")
        else:
            st.warning("⚠️ Please upload PDF files for indexing.")

if __name__ == "__main__":
    index_documents()
