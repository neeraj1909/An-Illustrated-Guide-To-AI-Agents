import json
import os
import urllib.request
from dataclasses import asdict 
from pathlib import Path

from dotenv import load_dotenv

from agent import TinyAgent
from llm import LLM
from memory import Memory, TrimmingMemory, SummarizationMemory, RAGMemory
from embeddings import EmbeddingModel 

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


llm = LLM(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
    # Open WebUI exposes an OpenAI-compatible endpoint at /api.
    # For direct Ollama, use http://<host>:11434/v1 instead.
    base_url=os.getenv("LLM_BASE_URL", "http://100.86.159.44:38080/api"),
    api_key=os.getenv("OPENWEBUI_API_KEY", "no_key"),
)

# Initialize EmbeddingGemma
embedding_model = EmbeddingModel(model="embeddinggemma")

# RAGMemory with external documents
documents = [
    "Sarah works as a marine biologist studying coral reefs.",
    "Sarah lives in Lisbon, Portugal.",
    "Sarah's favorite hobby is rock climbing.",
    "Sarah favorite animal is flamingos.",
    "Sarah speaks fluent Spanish and Portuguese.",
    "Ilse is a software engineer at a renewable energy startup.",
    "Ilse lives in Amsterdam, the Netherlands.",
    "Ilse plays the cello in a local string quartet.",
    "Ilse's favorite author is Brandon Sanderson.",
    "Ilse's favorite animal is dolphins.",
]
memory = RAGMemory(documents=documents, embedding_model=embedding_model)

agent = TinyAgent(llm=llm, memory=memory)

response = agent.run("What is Sarah's favorite animal?") 
print(f"Response: {response}\n")
print(f"Summarized Conversation: {agent.memory.get_messages()}") 
