import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from /{docs_path}")

    if not os.path.exists:
        raise FileNotFoundError("The directory does not exist")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    if len(documents)==0:
        raise FileNotFoundError("The directory does not contain any files")
    
    return documents

def split_documents(documents,chunk_size=1000,chunk_overlap=0):
    print(f"Splitting {len(documents)} documents to chunks")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents()

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents = chunks,
        persist_directory=persist_directory,
        embedding=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
    )
    
    print(f"Vectorstore created and stored in {persist_directory}")
    return vectorstore

def main():
    docs_path="/docs"
    persist_directory="db/chroma_db"
    
    if os.path.exists(persist_directory):
        print("Vector store already exists. Document processing not required")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(
            persist_directory=persist_directory,
            embedding=embedding_model,
            collection_metadata={"hnsw:space":"cosine"}
        )
        return vectorstore
    
    print("Persist directory does not exist. Initialize vector store")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks,persist_directory)
    
    return vectorstore

if __name__ == "__main__":
    main()