from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama

# 1. Define state shape
class ChatState(TypedDict):
    input: str
    output: str
    turn: int  # Keep track of how many interactions have occurred

# 2. Load local model
llm = ChatOllama(model="llama3", temperature=0)

# 3. Node: Ask the LLM
def llm_node(state: ChatState) -> ChatState:
    input_text = state["input"]
    response = llm.invoke(f"A person says: {input_text}. How would you respond?").content

    return {
        "input": input_text,
        "output": response,
        "turn": state["turn"],  # Keep turn unchanged here
    }

# 4. Node: Increment turn counter
def increment_turn(state: ChatState) -> ChatState:
    # This node keeps the same input/output but adds +1 to turn count
    return {
        "input": state["input"],
        "output": state["output"],
        "turn": state["turn"] + 1,
    }

# 5. Build the graph
builder = StateGraph(ChatState)
builder.add_node("ask_llm", llm_node)
builder.add_node("increment_turn", increment_turn)
builder.set_entry_point("ask_llm")
builder.add_edge("ask_llm", "increment_turn")
builder.set_finish_point("increment_turn")
graph = builder.compile()

# 6. Run the graph
initial_turn = 0
while True:
    user_input = input("\nType something (or 'exit'): ")
    if user_input.lower() == "exit":
        break

    result = graph.invoke({"input": user_input, "turn": initial_turn, "output": ""})
    print(f"\n[Model]: {result['output']}")
    print(f"[Turn Count]: {result['turn']}")

    initial_turn = result["turn"]  # Carry turn forward for next round
