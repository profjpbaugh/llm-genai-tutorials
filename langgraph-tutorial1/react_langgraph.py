import re
from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama

# 1. Define the full ReAct state structure
class ReActState(TypedDict):
    input: str
    thought: str
    action: str
    observation: str
    final_answer: str

# 2. Define a simple calculator tool
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# 3. Load the model
llm = ChatOllama(model="llama3", temperature=0)

# 4. Node: Ask for Thought and Action
def planner_node(state: ReActState) -> ReActState:
    prompt = f"""You are a helpful assistant using the ReAct pattern. 
Always respond with the following format — use these labels **exactly**:

Thought: <your reasoning here>
Action: calculator[math expression here]

For example:
Thought: I need to add 2 and 2.
Action: calculator[2 + 2]

Now respond to this question:
Question: {state['input']}
"""

    response = llm.invoke(prompt).content
    thought_match = re.search(r"Thought:\s*(.*)", response)
    action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", response)

    return {
        "input": state["input"],
        "thought": thought_match.group(1).strip() if thought_match else "",
        "action": action_match.group(0).strip() if action_match else "",
        "observation": "",
        "final_answer": "",
    }

# 5. Node: Execute the tool
def action_node(state: ReActState) -> ReActState:
    match = re.search(r"(\w+)\[(.*?)\]", state["action"])
    if match:
        tool, param = match.groups()
        if tool == "calculator":
            result = calculator(param)
        else:
            result = f"Unknown tool: {tool}"
    else:
        result = "Invalid action format."

    return {
        **state,
        "observation": result
    }

# 6. Node: Generate the final answer
def final_node(state: ReActState) -> ReActState:
    prompt = f"""You previously reasoned:

{state['thought']}

You used this tool:
{state['action']}

And got this observation:
{state['observation']}

Now respond with your final answer.
Final Answer:"""

    result = llm.invoke(prompt).content
    return {
        **state,
        "final_answer": result.strip()
    }

# 7. Build the graph
builder = StateGraph(ReActState)
builder.add_node("plan", planner_node)
builder.add_node("act", action_node)
builder.add_node("final", final_node)
builder.set_entry_point("plan")
builder.add_edge("plan", "act")
builder.add_edge("act", "final")
builder.set_finish_point("final")
graph = builder.compile()

# 8. Run it
while True:
    user_input = input("\nAsk a math question (or 'exit'): ")
    if user_input.lower() == "exit":
        break

    result = graph.invoke({
        "input": user_input,
        "thought": "",
        "action": "",
        "observation": "",
        "final_answer": ""
    })

    print("\n--- ReAct Breakdown ---")
    print("Thought:", result["thought"])
    print("Action:", result["action"])
    print("Observation:", result["observation"])
    print("Final Answer:", result["final_answer"])
