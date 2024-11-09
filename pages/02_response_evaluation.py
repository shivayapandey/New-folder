# 2. pages/02_response_evaluation.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import re
import numpy as np
import pysbd
from requests.exceptions import ConnectionError

# Helper functions for evaluation
def extract_number(response):
    match = re.search(r'\b(10|[0-9])\b', response)
    return float(match.group(0)) if match else np.nan

def sent_tokenize(text: str):
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)

def get_context_relevancy(llm, query, context):
    CONTEXT_RELEVANCE = """On a scale of 0-10, how relevant is this context to answering the question? 
    Question: {question}
    Context: {context}
    Score: """
    
    total_score = 0
    score_count = 0
    for content in context:
        score_response = llm.invoke(CONTEXT_RELEVANCE.format(
            question=query, 
            context=str(content)
        ))
        score = float(extract_number(score_response.content))
        if not np.isnan(score):
            total_score += score
            score_count += 1
    return round(total_score / score_count, 1) if score_count > 0 else 0

def get_answer_relevancy(llm, query, response):
    ANSWER_RELEVANCE = """On a scale of 0-10, how relevant is this response to the question asked?
    Question: {question}
    Response: {response}
    Score: """
    
    response_text = response if isinstance(response, str) else str(response)
    answer_response = llm.invoke(ANSWER_RELEVANCE.format(
        question=query, 
        response=response_text
    ))
    return float(extract_number(answer_response.content))

def get_groundedness_score(llm, response, context):
    GROUNDEDNESS = """On a scale of 0-10, how well is this statement supported by the given context?
    Statement: {statement}
    Context: {context}
    Score: """
    
    total_score = 0
    score_count = 0
    statements = sent_tokenize(response)
    for statement in statements:
        groundedness_response = llm.invoke(GROUNDEDNESS.format(
            statement=str(statement), 
            context=" ".join(map(str, context))
        ))
        score = float(extract_number(groundedness_response.content))
        if not np.isnan(score):
            total_score += score
            score_count += 1
    return round(total_score / score_count, 1) if score_count > 0 else 0

def evaluate_response():
    st.title("Response Generation and Evaluation")

    # Sidebar inputs
    st.sidebar.header("API Configuration")
    groq_api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")
    
    # Main query input
    query = st.text_input("Enter your query:", placeholder="Type your question here...")

    # Check if vectorstore exists
    if 'vectorstore' not in st.session_state:
        st.error("⚠️ Please index documents first on the indexing page.")
        return

    vectorstore = st.session_state['vectorstore']
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # Generate and evaluate response
    if st.button("Generate and Evaluate Response"):
        if not groq_api_key or not query:
            st.warning("⚠️ Please enter both API key and query.")
            return

        try:
            with st.spinner("Generating and evaluating response..."):
                # Initialize language model
                llm = ChatGroq(
                    model="llama3-8b-8192",
                    groq_api_key=groq_api_key,
                    temperature=0.1
                )

                # Define prompt template
                template = """Answer the following question based on the provided context. Be concise and specific.
                
                Context: {context}
                Question: {query}
                
                Answer: """
                
                prompt = ChatPromptTemplate.from_template(template)

                # Define RAG chain
                rag_chain = (
                    {"context": retriever, "query": RunnablePassthrough()}
                    | prompt
                    | llm
                )

                # Retrieve documents
                retrieved_docs = retriever.invoke(query)
                context = [str(doc.page_content) for doc in retrieved_docs]

                # Generate response
                response = rag_chain.invoke(query)
                response_text = response.content if hasattr(response, 'content') else str(response)

                # Display response
                st.subheader("Generated Response:")
                st.write(response_text)

                # Calculate and display scores
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    context_score = get_context_relevancy(llm, query, context)
                    st.metric("Context Relevancy", f"{context_score}/10")
                
                with col2:
                    answer_score = get_answer_relevancy(llm, query, response_text)
                    st.metric("Answer Relevancy", f"{answer_score}/10")
                
                with col3:
                    groundedness_score = get_groundedness_score(llm, response_text, context)
                    st.metric("Groundedness", f"{groundedness_score}/10")

                # Display retrieved context
                with st.expander("View Retrieved Context"):
                    for i, ctx in enumerate(context, 1):
                        st.markdown(f"**Context {i}:**\n{ctx}\n---")

        except ConnectionError as e:
            st.error(f"❌ Connection Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    evaluate_response()