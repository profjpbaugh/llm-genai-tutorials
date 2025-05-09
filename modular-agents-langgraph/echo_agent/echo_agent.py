from typing import TypedDict
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate

class EchoState(TypedDict):
    input: str
    output: str

class EchoAgent:
    def __init__(self):
        self.prompt = PromptTemplate.from_template("You said: {text}") 
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(EchoState)
        builder.add_node("echo", self._echo_node)
        builder.set_entry_point("echo")
        builder.set_finish_point("echo")
        return builder.compile()

    def _echo_node(self, state: EchoState) -> EchoState:
        message = self.prompt.format(text=state["input"])
        return {**state, "output": f"You said: {state['input']}"}

    def run(self, user_input: str) -> EchoState:
        return self.graph.invoke({"input": user_input, "output": ""})
