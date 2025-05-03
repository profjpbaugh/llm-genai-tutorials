from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")

print("Chat with LLaMA (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = llm.invoke(user_input)
    print("LLaMA:", response.content)
