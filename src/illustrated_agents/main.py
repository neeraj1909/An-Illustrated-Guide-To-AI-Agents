import json
import os
import urllib.request
from dataclasses import asdict 
from pathlib import Path

from dotenv import load_dotenv

from agent import TinyAgent
from llm import LLM
from memory import Memory, TrimmingMemory

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


llm = LLM(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
    # Open WebUI exposes an OpenAI-compatible endpoint at /api.
    # For direct Ollama, use http://<host>:11434/v1 instead.
    base_url=os.getenv("LLM_BASE_URL", "http://100.86.159.44:38080/api"),
    api_key=os.getenv("OPENWEBUI_API_KEY", "no_key"),
)

memory = TrimmingMemory()
agent = TinyAgent(llm=llm, memory=memory)

# Run some interactions
response_1 = agent.run("Hi! I'm reading 'An Illustrated Guide to AI Agents'.")
response_2 = agent.run("There are many flamingos in this book, why?")
response_3 = agent.run("What is 1+1?")
response_4 = agent.run("What is 2+2?")

print(f"Trimmed Conversation: {agent.memory.get_messages()}")
