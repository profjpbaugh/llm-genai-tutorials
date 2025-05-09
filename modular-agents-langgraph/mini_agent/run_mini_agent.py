from mini_agent import MiniAgent

agent = MiniAgent()

while True:
    user_input = input("Say something (or 'exit'): ")
    if user_input.lower() == "exit":
        break

    result = agent.run(user_input)
    print("MiniAgent Output:", result["output"])
