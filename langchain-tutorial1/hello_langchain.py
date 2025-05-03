from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# 1. Create a dynamic prompt with a variable
prompt = PromptTemplate.from_template("Tell me a fun fact about {topic}")

# 2. Use local LLaMA3 through Ollama
llm = ChatOllama(model="llama3")

# 3. Parse the output as plain text
output_parser = StrOutputParser()

# 4. Chain the prompt → model → output parser
chain = prompt | llm | output_parser

# 5. Run the chain with an input
result = chain.invoke({"topic": "octopuses"})
print(result)
