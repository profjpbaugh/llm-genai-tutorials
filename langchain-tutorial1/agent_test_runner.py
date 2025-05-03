from langchain_ollama import ChatOllama
from AgentCore import AgentCore

# Define tools
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Invalid expression: {e}"

def facts(topic: str) -> str:
    return f"Fun fact about {topic}: It is surprisingly interesting!"

tools = {
    "calculator": calculator,
    "facts": facts
}

# Agent behavior description (ReAct-style)
description = """
You are a helpful assistant who uses tools.

Available tools:
- calculator[expression]
- facts[topic]

Use the following format:
Thought: ...
Action: tool_name[parameters]
Observation: ...
Final Answer: ...
"""

# Load local LLaMA model
llm = ChatOllama(model="llama3", temperature=0)

# Create the agent
agent = AgentCore(model=llm, tools=tools, description=description)

# User interaction loop
print("Ask me something (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    agent.run(user_input)
