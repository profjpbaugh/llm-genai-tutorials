from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

prompt = PromptTemplate.from_template(
    """
    Generate a structured list of three interesting facts about {topic}.

    Format the response like this:
    - Fact 1: ...
    - Fact 2: ...
    - Fact 3: ...
    """
)

llm = ChatOllama(model="llama3")
parser = StrOutputParser()
chain = prompt | llm | parser

result = chain.invoke({"topic": "space travel"})
print(result)
