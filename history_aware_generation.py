from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# query = input("👉: ")

model = ChatOpenAI(model="gpt-4o")

chat_history = []

def ask_question(user_question):
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and single. Just retirn the rewritten question.")
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        result = model.invoke(messages)
        query = result.content.strip()
    else:
        query = user_question
    
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":3,
            "score_threshold":0.3
        }
    )
    relevant_docs = retriever.invoke(query)
    combined_input = f"""
    Based on the following documents, please answer this question: {query}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    print(f"🤖: {answer}")

def start_chat():
    print("Ask me a question. Type 'exit' to quit")

    while True:
        question = input("👉: ")
        if question.lower()=='exit':
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()