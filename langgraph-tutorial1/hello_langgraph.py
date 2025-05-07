from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama

# 1. Define the shape of the state using TypedDict
class EchoState(TypedDict):
    input: str
    output: str

# 2. Load the local LLaMA3 model
llm = ChatOllama(model="llama3")

# 3. Define a node that sends the input to the LLM
def ask_llm(state: EchoState) -> EchoState:
    user_input = state["input"]
    response = llm.invoke(f"Respond to this input: {user_input}").content
    return {"input": user_input, "output": response}

# 4. Build the graph
builder = StateGraph(EchoState)
builder.add_node("talk_to_llm", ask_llm)
builder.set_entry_point("talk_to_llm")
builder.set_finish_point("talk_to_llm")  # Required to finalize
graph = builder.compile()

# 5. Run it
user_input = input("Type something to the LLM: ")
result = graph.invoke({"input": user_input})
print("\nModel said:\n" + result["output"])
