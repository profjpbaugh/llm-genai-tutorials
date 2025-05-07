# info_agent.py
from tools import doc_search
from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
import re

class InfoState(TypedDict):
    input: str
    thought: str
    final_answer: str

class InfoAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.graph = self._build_graph()
        self.tools = {
            "doc_search": doc_search
        }


    def _build_graph(self):
        builder = StateGraph(InfoState)

        builder.add_node("think_and_answer", self._respond)
        builder.set_entry_point("think_and_answer")
        builder.set_finish_point("think_and_answer")

        return builder.compile()

    def _respond(self, state: InfoState) -> InfoState:
        prompt = f"""You are an information assistant.

    You can answer questions using:
    - doc_search[query] → for company-specific information

    Use this format exactly:
    Thought: ...
    Action: tool_name[parameter]

    Example:
    Thought: I need to check the remote work policy.
    Action: doc_search[remote work policy]

    Question: {state['input']}
    """

        response = self.llm.invoke(prompt).content
        print("DEBUG info planner response:", response)

        thought_match = re.search(r"Thought:\s*(.*)", response)
        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", response)

        if not action_match:
            return {
                "input": state["input"],
                "thought": "[No thought]",
                "final_answer": "Invalid tool format."
            }

        tool_name, param = action_match.groups()

        if tool_name not in self.tools:
            return {
                "input": state["input"],
                "thought": thought_match.group(1).strip() if thought_match else "[No thought]",
                "final_answer": f"Unknown tool: {tool_name}"
            }

        observation = self.tools[tool_name](param)

        final_prompt = f"""You were asked: {state['input']}

    You reasoned:
    {thought_match.group(1).strip() if thought_match else ''}

    You used:
    Action: {tool_name}[{param}]
    Observation: {observation}

    Now provide a final answer.
    Final Answer:"""

        final = self.llm.invoke(final_prompt).content.strip()
        return {
            "input": state["input"],
            "thought": thought_match.group(1).strip() if thought_match else "[No thought]",
            "final_answer": final
        }


    def run(self, user_input: str) -> InfoState:
        return self.graph.invoke({
            "input": user_input,
            "thought": "",
            "final_answer": ""
        })
