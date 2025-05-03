from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

prompt = PromptTemplate.from_template(
    "Write a short story about a person named {name} who loves {hobby}."
)

llm = ChatOllama(model="llama3")
parser = StrOutputParser()
chain = prompt | llm | parser

result = chain.invoke({"name": "Lena", "hobby": "skydiving"})
print(result)
