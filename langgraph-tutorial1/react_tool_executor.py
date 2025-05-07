import re
from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama

# 1. Define the state structure
class ReActState(TypedDict):
    input: str
    thought: str
    action: str
    observation: str
    final_answer: str

# 2. Define tool functions
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = {
    "calculator": calculator
}

# 3. Load the model
llm = ChatOllama(model="llama3", temperature=0)

# 4. Node: Plan with Thought + Action
def planner_node(state: ReActState) -> ReActState:
    prompt = f"""You are a helpful assistant using the ReAct pattern.
You must solve the entire question using a single Action.

Use this format exactly:
Thought: <your full reasoning>
Action: calculator[entire expression]

Example:
Thought: I need to divide 50 by the result of 9+ 1.
Action: calculator[50 / (9 + 1)]

Question: {state['input']}
"""

    response = llm.invoke(prompt).content
    print("DEBUG planner response:", response)

    thought_match = re.search(r"Thought:\s*(.*)", response)
    action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", response)

    return {
        "input": state["input"],
        "thought": thought_match.group(1).strip() if thought_match else "[No Thought]",
        "action": action_match.group(0).strip() if action_match else "[No Action]",
        "observation": "",
        "final_answer": "",
    }

# 5. Node: Centralized tool executor
def tool_executor_node(state: ReActState) -> ReActState:
    action_str = state["action"]
    match = re.search(r"(\w+)\[(.*?)\]", action_str)

    if not match:
        return {**state, "observation": "Invalid action format."}

    tool, param = match.groups()
    tool = tool.strip()
    param = param.strip()

    if tool not in tools:
        return {**state, "observation": f"Unknown tool: {tool}"}

    try:
        result = tools[tool](param)
        return {**state, "observation": result}
    except Exception as e:
        return {**state, "observation": f"Error executing tool: {e}"}

# 6. Node: Final summary
def final_node(state: ReActState) -> ReActState:
    prompt = f"""You previously reasoned:

{state['thought']}

You used this tool:
{state['action']}

You observed:
{state['observation']}

Now respond with your final answer.
Final Answer:"""

    result = llm.invoke(prompt).content
    cleaned = result.strip()
    if cleaned.lower().startswith("final answer:"):
        cleaned = cleaned[len("final answer:"):].strip()

    return {**state, "final_answer": cleaned}

# 7. Build the graph
builder = StateGraph(ReActState)
builder.add_node("plan", planner_node)
builder.add_node("run_tool", tool_executor_node)
builder.add_node("final", final_node)

builder.set_entry_point("plan")
builder.add_edge("plan", "run_tool")
builder.add_edge("run_tool", "final")
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
