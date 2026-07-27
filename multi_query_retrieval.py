from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel

load_dotenv()

persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model='gpt-4o',temperature=0)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

class QueryVariations(BaseModel):
    queries: List[str]

original_query = "How does tesla make money"
print(f"Original Query: {original_query}")

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

retriever = db.as_retriever(search_kwargs={"k":5})

for query in query_variations:
    print(f"Results for {query}:")
    docs = retriever.invoke(query)
    for i,doc in enumerate(docs,1):
        print(f"doc{i}")
        print(doc.page_content[:150])