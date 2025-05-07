# run_router.py

from router_agent import RouterAgent

router = RouterAgent()

while True:
    user_input = input("\nAsk something (or 'exit'): ")
    if user_input.lower() == "exit":
        break

    result = router.run(user_input)

    print(f"\n--- Routed to: {result['agent']} ---")
    print("Thought:", result.get("thought", "[None]"))
    if result['agent'] == "MathAgent":
        print("Action:", result["action"])
        print("Observation:", result["observation"])
    print("Final Answer:", result["final_answer"])
