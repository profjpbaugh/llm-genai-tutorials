# run_math_agent.py

from math_agent import MathAgent

agent = MathAgent()

while True:
    user_input = input("\nAsk a math question (or 'exit'): ")
    if user_input.lower() == "exit":
        break

    result = agent.run(user_input)

    print("\n--- ReAct Breakdown ---")
    print("Thought:", result["thought"])
    print("Action:", result["action"])
    print("Observation:", result["observation"])
    print("Final Answer:", result["final_answer"])
