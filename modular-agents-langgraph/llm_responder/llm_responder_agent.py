from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

class LLMResponderState(TypedDict):
    input: str
    llm_response: str

class LLMResponderAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3")
        self.prompt = PromptTemplate.from_template(
            "You are a helpful assistant. Respond concisely to: {query}"
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(LLMResponderState)
        builder.add_node("respond", self._respond_with_llm)
        builder.set_entry_point("respond")
        builder.set_finish_point("respond")
        return builder.compile()

    def _respond_with_llm(self, state: LLMResponderState) -> LLMResponderState:
        prompt_text = self.prompt.format(query=state["input"])
        response = self.llm.invoke(prompt_text).content
        return {**state, "llm_response": response}

    def run(self, user_input: str) -> LLMResponderState:
        return self.graph.invoke({"input": user_input, "llm_response": ""})
