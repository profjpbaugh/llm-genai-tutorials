# math_agent.py

import re
from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama

# 1. Define shared state type
class ReActState(TypedDict):
    input: str
    thought: str
    action: str
    observation: str
    final_answer: str

# 2. Sample calculator tool
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# 3. Tools registry
tools = {
    "calculator": calculator
}

# 4. Reusable MathAgent class
class MathAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ReActState)

        builder.add_node("plan", self._planner_node)
        builder.add_node("run_tool", self._tool_executor_node)
        builder.add_node("final", self._final_node)

        builder.set_entry_point("plan")
        builder.add_edge("plan", "run_tool")
        builder.add_edge("run_tool", "final")
        builder.set_finish_point("final")

        return builder.compile()

    def _planner_node(self, state: ReActState) -> ReActState:
        prompt = f"""You are a helpful assistant using the ReAct pattern.
You must solve the entire question using a single Action.

Format:
Thought: <reasoning>
Action: calculator[expression]

Example:
Thought: I need to divide 100 by the result of 4 + 1.
Action: calculator[100 / (4 + 1)]

Question: {state['input']}
"""
        response = self.llm.invoke(prompt).content
        print("DEBUG planner response:", response)

        thought_match = re.search(r"Thought:\s*(.*)", response)
        action_match = re.search(r"Action:\s*(?:([\w_]+)\[(.*?)\]|\[([\w_]+)\s+(.*?)\])", response)

        return {
            "input": state["input"],
            "thought": thought_match.group(1).strip() if thought_match else "[No Thought]",
            "action": (
                f"{(action_match.group(1) or action_match.group(3))}[{(action_match.group(2) or action_match.group(4)).strip()}]"
                if action_match else "[No Action]"
                ),
            "observation": "",
            "final_answer": "",
        }

    def _tool_executor_node(self, state: ReActState) -> ReActState:
        match = re.search(r"(\w+)\[(.*?)\]", state["action"])
        if not match:
            return {**state, "observation": "Invalid action format. Use: tool[param]"}

        tool, param = match.groups()
        tool = tool.strip()
        param = param.strip()

        if tool not in tools:
            return {
                **state,
                "observation": f"Unknown tool: '{tool}'. Available tools: {', '.join(tools.keys())}"
            }

        try:
            result = tools[tool](param)
            return {**state, "observation": result}
        except Exception as e:
            return {**state, "observation": f"Tool error: {e}"}


    def _final_node(self, state: ReActState) -> ReActState:
        prompt = f"""The original user question was: {state['input']}

You reasoned:
{state['thought']}

You used this tool:
{state['action']}

You observed:
{state['observation']}

Now provide your final answer.
Final Answer:"""

        result = self.llm.invoke(prompt).content.strip()

        if result.lower().startswith("final answer:"):
            result = result[len("final answer:"):].strip()

        return {**state, "final_answer": result}

    def run(self, user_input: str) -> ReActState:
        return self.graph.invoke({
            "input": user_input,
            "thought": "",
            "action": "",
            "observation": "",
            "final_answer": ""
        })
