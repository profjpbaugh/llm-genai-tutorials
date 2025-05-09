from typing import TypedDict
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate

class MiniAgentState(TypedDict):
    input: str
    modified: str
    output: str

class MiniAgent:
    def __init__(self):
        self.final_prompt = PromptTemplate.from_template(">>> {text} <<<")
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(MiniAgentState)
        builder.add_node("shoutify", self._shoutify)
        builder.add_node("finalize", self._finalize)
        builder.set_entry_point("shoutify")
        builder.add_edge("shoutify", "finalize")
        builder.set_finish_point("finalize")
        return builder.compile()

    def _shoutify(self, state: MiniAgentState) -> MiniAgentState:
        return {**state, "modified": state["input"].upper()}

    def _finalize(self, state: MiniAgentState) -> MiniAgentState:
        output = self.final_prompt.format(text=state["modified"])
        return {**state, "output": output}

    def run(self, user_input: str) -> MiniAgentState:
        return self.graph.invoke({
            "input": user_input,
            "modified": "",
            "output": ""
        })
