import re
from langchain_ollama import ChatOllama

class AgentCore:
    def __init__(self, model, tools: dict[str, callable], description: str):
        self.llm = model
        self.tools = tools
        self.description = description

    def _extract_action(self, text: str):
        import re
        # Match Action: tool[param] or fallback Action: [tool param]
        match = re.search(r"Action:\s*(?:([\w_]+)\[(.*?)\]|\[([\w_]+)\s+(.*?)\])",
                        text, 
                        re.DOTALL)
        if match:
            tool = match.group(1) or match.group(3)
            raw_param = match.group(2) or match.group(4)
            param = raw_param.strip().strip('"').strip("'")
            return tool, param
        return None, None



    def _execute_tool(self, tool_name: str, param: str):
        if tool_name not in self.tools:
            return (
                f"The tool '{tool_name}' is not available.\n"
                f"Available tools: {', '.join(self.tools.keys())}"
            )
        try:
            return self.tools[tool_name](param)
        except Exception as e:
            return f"Tool '{tool_name}' failed to execute. Error: {e}"


    def run(self, user_input: str):
        prompt = f"{self.description}\n\nQuestion: {user_input}"
        response = self.llm.invoke(prompt).content

        print("\n--- Raw Model Response ---")
        print(response)

        tool_name, param = self._extract_action(response)

        if tool_name:
            result = self._execute_tool(tool_name, param)
            print(f"\n--- Tool Used ---\n{tool_name}({param}) = {result}")
        else:
            print("\nNo valid Action step found.")
            print("Hint: Expected format is: Action: tool[param]")
            print("     Example: Action: doc_search[vacation policy]")

