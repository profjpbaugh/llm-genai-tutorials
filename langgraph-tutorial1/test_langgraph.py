from langgraph.graph import StateGraph
from typing import TypedDict

# 1. Define the shape of the state using TypedDict
class EchoState(TypedDict):
    input: str
    output: str

# 2. Define a simple echo node
def echo_node(state: EchoState) -> EchoState:
    print("Inside echo_node")
    return {"output": f"You said: {state['input']}", "input": state["input"]}

# 3. Build the graph with the state schema
builder = StateGraph(EchoState)
builder.add_node("echo", echo_node)
builder.set_entry_point("echo")
builder.set_finish_point("echo")  # Required in newer LangGraph versions
graph = builder.compile()

# 4. Run the graph
result = graph.invoke({"input": "Hello, LangGraph!"})
print(result)
