from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.", 
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.", 
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments."
]

documents = [Document(page_content=chunk,metadata={"source":f"Document {i}"}) for i,chunk in enumerate(chunks)]

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = Chroma.from_documents(
    documents = documents,
    embedding=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k":3})
test_query = input("Enter query: ")

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever,bm25_retriever],
    weights=[0.7,0.3]
)

retrieved_chunks = hybrid_retriever.invoke(test_query)

for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")
print()