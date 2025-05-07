# router_agent.py

from math_agent import MathAgent
from info_agent import InfoAgent

class RouterAgent:
    def __init__(self):
        self.math_agent = MathAgent()
        self.info_agent = InfoAgent()

    def route(self, user_input: str) -> str:
        lowered = user_input.lower()

        math_keywords = ["calculate", "+", "-", "*", "/", "evaluate", "math", "solve", "equation"]
        info_keywords = ["policy", "benefits", "pto", "company", "fact", "tell me", "explain"]

        if any(kw in lowered for kw in math_keywords):
            return "math"
        elif any(kw in lowered for kw in info_keywords):
            return "info"
        else:
            return "info"  # default fallback


    def run(self, user_input: str) -> dict:
        route = self.route(user_input)
        print(f"[Router] Routing to: {route}")

        if route == "math":
            result = self.math_agent.run(user_input)
            return {
                "agent": "MathAgent",
                "thought": result["thought"],
                "action": result["action"],
                "observation": result["observation"],
                "final_answer": result["final_answer"]
            }
        else:
            result = self.info_agent.run(user_input)
            return {
                "agent": "InfoAgent",
                "thought": result["thought"],
                "final_answer": result["final_answer"]
            }
