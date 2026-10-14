import json
import os
import urllib.request
from dataclasses import asdict 
from pathlib import Path

from dotenv import load_dotenv

from agent import TinyAgent
from llm import LLM
from memory import Memory, TrimmingMemory, SummarizationMemory

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


llm = LLM(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
    # Open WebUI exposes an OpenAI-compatible endpoint at /api.
    # For direct Ollama, use http://<host>:11434/v1 instead.
    base_url=os.getenv("LLM_BASE_URL", "http://100.86.159.44:38080/api"),
    api_key=os.getenv("OPENWEBUI_API_KEY", "no_key"),
)

memory = SummarizationMemory(llm=llm)
agent = TinyAgent(llm=llm, memory=memory)

response = agent.run("Hi, my name is Sarah. Why are flamingos pink?") 
print(f"Summarized Conversation: {agent.memory.get_messages()}") 
