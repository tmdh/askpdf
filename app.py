import os
import streamlit as st
import torch
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

torch.classes.__path__ = []

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    collection_name="pdf_database",
    embedding_function=embedding_model,
    persist_directory="./pdf_db"
)

def add_to_db(pdf_docs):
    for uploaded_file in pdf_docs:

        # Store the file in filesystem
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        doc_metadata = [data[i].metadata for i in range(len(data))]
        doc_content = [data[i].page_content for i in range(len(data))]

        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            tokens_per_chunk=100,
            chunk_overlap=50
        )
        st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)
        
        db.add_documents(st_chunks)

        os.remove(temp_file_path)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_chain(query):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k':5})
    PROMPT_TEMPLATE = """
You are a highly knowledgeable assistant specializing in answering from documents.
Answer the question based only on the following context:
{context}

Answer the question based on the above context:
{question}

Use the provided context to answer the user's question accurately and concisely.
Don't justify your answers.
Don't give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar. 
"""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chat_model = ChatOpenAI(
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-chat-v3-0324:free"
    )

    output_parser = StrOutputParser()

    rag_chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt_template | chat_model | output_parser

    response = rag_chain.invoke(query)
    return response


def main():
    st.set_page_config(page_title="AskPDF", page_icon=":microscope:")
    st.header("AskPDF")
    
    query = st.text_area(
        ":bulb: Enter your query about your uploaded PDFs:"
    )

    if st.button("Submit"):
        if not query:
            st.warning("Please ask a question")
        
        else:
            with st.spinner("Thinking..."):
                st.write(run_rag_chain(query))
    
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Upload"):
            if not pdf_docs:
                st.warning("Please select a PDF")
            else:
                with st.spinner("Processing your documents..."):
                    add_to_db(pdf_docs)
                    st.success("Documents uploaded successfully")
        
    st.sidebar.write("Built with ❤️ by [tmdh](https://github.com/tmdh)")


if __name__ == "__main__":
    main()