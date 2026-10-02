from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_retriever(codebase_docs):
    # 1. Initialize local embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")


    # 3. Split and ingest into free local vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.create_documents(codebase_docs)
    vector_store = FAISS.from_documents(docs, embeddings)

    # 4. Expose the retriever as a tool the agent can call
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return retriever