from langchain_ollama import ChatOllama
from AgentCore import AgentCore
from Orchestrator import Orchestrator
from tools import doc_search

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
Do NOT use parentheses for tools — use square brackets like tool_name[param].


If you fail to follow this format, your answer will not be accepted.
"""


info_tools = {
    "facts": facts,
    "doc_search": doc_search
}

info_description = """
You are an information agent.

You have access to these tools:
- facts[topic] → Use this for general or public knowledge.
- doc_search[query] → Use this to look up employee and company-specific policies, documents, and internal guides.

Only use one tool per question.

Follow this exact ReAct format:
Thought: ...
Action: tool_name[parameter]
Observation: ...
Final Answer: ...

NEVER use the facts tool for company-specific policies or internal documents.
NEVER invent tools.
Do NOT use quotes.
Do NOT skip the Action step.
Do NOT use parentheses for tools — use square brackets like tool_name[param].
Do NOT put brackets around the entire action.
The format is: Action: tool[param] — NOT Action: [tool param]

FORMAT EXAMPLES (copy this pattern exactly):

Action: doc_search[remote work policy]
Action: facts[octopuses]


If you fail to follow this format, your answer will not be accepted.
"""



# Create models and agents
llm = ChatOllama(model="llama3", temperature=0)
math_agent = AgentCore(llm, {"calculator": calculator}, math_description)
info_agent = AgentCore(llm, info_tools, info_description)

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
