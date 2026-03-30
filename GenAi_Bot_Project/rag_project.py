from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# Session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = ChatOllama(model="llama3.2:latest")

# Folder to save uploaded PDFs
PDF_DIR = Path("doc_files")
PDF_DIR.mkdir(exist_ok=True)


# Function to process uploaded PDFs
def process_pdfs(uploaded_files):
    all_docs = []

    # Save and load each uploaded PDF
    for uploaded_file in uploaded_files:
        file_path = PDF_DIR / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        all_docs.extend(docs)

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(all_docs)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = InMemoryVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    st.session_state.vector_store = vector_store


# Function to answer user questions
def get_answer(query):
    vector_store = st.session_state.vector_store
    llm = st.session_state.llm

    if vector_store is None:
        return "Please upload a PDF first."

    # Retrieve relevant chunks from PDF
    docs = vector_store.similarity_search(query, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Prompt for model
    prompt = f"""
You are a helpful assistant.
Answer only from the given PDF context.
If the answer is not in the context, say:
"I couldn't find that in the uploaded PDF."

PDF Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content


# PDF uploader
uploaded_files = st.file_uploader(
    "Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("Processing PDF(s)..."):
        process_pdfs(uploaded_files)
    st.success("PDF processed successfully.")

# Show old chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask a question about your PDF")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.spinner("Thinking..."):
        answer = get_answer(query)

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Reset button
if st.button("Reset Chat"):
    st.session_state.vector_store = None
    st.session_state.messages = []
    st.rerun()