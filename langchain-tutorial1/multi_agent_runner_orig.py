from langchain_ollama import ChatOllama
from AgentCore import AgentCore
from Orchestrator import Orchestrator

# Define tools
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

def facts(topic: str) -> str:
    return f"Here's a fun fact about {topic}!"

# Descriptions
math_description = """
You are a specialized math agent.

You MUST use the following tool:

- calculator[expression]

Use this EXACT format and syntax:

Thought: <your reasoning>
Action: calculator[expression]
Observation: <result of tool>
Final Answer: <conclusion>

DO NOT write "calculator expression".
DO NOT skip the brackets [].
DO NOT use quotes.
DO NOT omit the Action step.

If you fail to follow this format, your answer will not be accepted.
"""


info_description = """
You are an info agent.

You MUST use one of these tools to answer:
- facts[topic]

Respond ONLY using this exact ReAct format:

Thought: <your internal reasoning>
Action: facts[topic]
Observation: <result of running the tool>
Final Answer: <your complete answer>

Do NOT make up new tool names.
Do NOT add quotes around the parameter.
Do NOT skip the Action step.
Do NOT say anything outside of the ReAct format.
"""

# Create models and agents
llm = ChatOllama(model="llama3", temperature=0)
math_agent = AgentCore(llm, {"calculator": calculator}, math_description)
info_agent = AgentCore(llm, {"facts": facts}, info_description)

# Orchestrator
router = Orchestrator({
    "math": math_agent,
    "info": info_agent
})

# Interactive loop
print("Ask something (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    router.run(user_input)
