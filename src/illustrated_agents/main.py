import json
import os
import urllib.request
from dataclasses import asdict 
from pathlib import Path

from dotenv import load_dotenv

from agent import TinyAgent
from llm import LLM


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


llm = LLM(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
    # Open WebUI exposes an OpenAI-compatible endpoint at /api.
    # For direct Ollama, use http://<host>:11434/v1 instead.
    base_url=os.getenv("LLM_BASE_URL", "http://100.86.159.44:38080/api"),
    api_key=os.getenv("OPENWEBUI_API_KEY", "no_key"),
)
agent = TinyAgent(llm=llm)
output = agent.run("What is 2 + 2?")

print(f"{output}")

print(agent.trajectory.runs)

# # Re-initialize the LLM with the updated class
# llm = LLM(model="gemma4:e4b")

# # Generate a `Response` dataclass
# response = llm.generate([{"role": "user", "content": "Hi! How's life?"}])
# print(json.dumps(asdict(response), indent=2, ensure_ascii=False))
