from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Step 1: Load local LLaMA model
llm = ChatOllama(model="llama3", temperature=0)

# Step 2: Prompt with placeholder for chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Step 3: Chain the prompt and model
base_chain = prompt | llm

# Step 4: Session-based message history manager
session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Step 5: Wrap the base chain with memory logic
chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Step 6: Simple terminal loop
print("Chat with LLaMA (with memory). Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    result = chat_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "local_user"}}
    )

    print("LLaMA:", result.content)
