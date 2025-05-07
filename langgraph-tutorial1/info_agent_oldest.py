# info_agent.py

from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama

class InfoState(TypedDict):
    input: str
    thought: str
    final_answer: str

class InfoAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(InfoState)

        builder.add_node("think_and_answer", self._respond)
        builder.set_entry_point("think_and_answer")
        builder.set_finish_point("think_and_answer")

        return builder.compile()

    def _respond(self, state: InfoState) -> InfoState:
        prompt = f"""You are an information assistant.

Question: {state['input']}

Respond with:
Thought: <why you know this or how you'd look it up>
Final Answer: <concise, friendly fact-based answer>
"""
        response = self.llm.invoke(prompt).content
        lines = response.strip().split("Final Answer:")
        thought = lines[0].replace("Thought:", "").strip() if len(lines) > 0 else ""
        final = lines[1].strip() if len(lines) > 1 else "[No Final Answer]"

        return {
            "input": state["input"],
            "thought": thought,
            "final_answer": final
        }

    def run(self, user_input: str) -> InfoState:
        return self.graph.invoke({
            "input": user_input,
            "thought": "",
            "final_answer": ""
        })
