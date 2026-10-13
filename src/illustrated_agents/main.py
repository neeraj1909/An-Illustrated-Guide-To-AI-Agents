import json
import urllib.request
from dataclasses import asdict 

from agent import TinyAgent
from llm import LLM


llm = LLM(model="gemma4:e4b")
agent = TinyAgent(llm=llm)
output = agent.run("What is 2 + 2?")

print(f"{output}")

print(agent.trajectory.runs)

# # Re-initialize the LLM with the updated class
# llm = LLM(model="gemma4:e4b")

# # Generate a `Response` dataclass
# response = llm.generate([{"role": "user", "content": "Hi! How's life?"}])
# print(json.dumps(asdict(response), indent=2, ensure_ascii=False))
