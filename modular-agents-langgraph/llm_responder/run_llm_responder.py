from llm_responder_agent import LLMResponderAgent

agent = LLMResponderAgent()

while True:
    user_input = input("Ask something (or 'exit'): ")
    if user_input.lower() == "exit":
        break

    result = agent.run(user_input)
    print("\nLLMResponder Output:")
    print(result["llm_response"])
