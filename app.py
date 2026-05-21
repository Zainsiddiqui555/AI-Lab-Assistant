import streamlit as st
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.question_answering import load_qa_chain
import tempfile
import os

st.set_page_config(
    page_title="AI Laboratory Assistant",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 AI Laboratory Assistant")
st.write("Upload PDFs and ask questions")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    st.warning("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF before processing.")
    else:
        all_docs = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)
            os.remove(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        st.session_state.vectorstore = vectorstore

        st.success("Documents Processed")

question = st.text_input("Ask Question")

if st.button("Ask"):
    if st.session_state.vectorstore is None:
        st.warning("Please process PDFs first before asking a question.")
    elif not question:
        st.warning("Please enter a question first.")
    else:
        docs = st.session_state.vectorstore.similarity_search(
            question,
            k=4
        )

        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )

        chain = load_qa_chain(
            llm,
            chain_type="stuff"
        )

        response = chain.run(
            input_documents=docs,
            question=question
        )

        st.write(response)
