from echo_agent import EchoAgent

agent = EchoAgent()

while True:
    user_input = input("Say something (or 'exit'): ")
    if user_input.lower() == "exit":
        break

    result = agent.run(user_input)
    print("EchoAgent:", result["output"])
