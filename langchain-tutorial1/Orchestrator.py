class Orchestrator:
    def __init__(self, agents: dict[str, object]):
        self.agents = agents

    def route(self, user_input: str) -> str:
        lowered = user_input.lower()
        if any(word in lowered for word in ["math", "calculate", "+", "-", "*", "/"]):
            return "math"
        else:
            return "info"

    def run(self, user_input: str):
        agent_name = self.route(user_input)
        print(f"[Orchestrator] Routed to agent: {agent_name}")
        agent = self.agents.get(agent_name)
        if agent:
            agent.run(user_input)
        else:
            print("No agent found for the request.")
