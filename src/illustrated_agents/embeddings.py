import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class EmbeddingModel:
    """Create embeddings using the remote Ollama server."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv("LLM_BASE_URL")
        )
        if not self.base_url:
            raise ValueError("Set OLLAMA_BASE_URL in .env")
        self.base_url = self.base_url.rstrip("/")

    def embed(self, text: str) -> list[float]:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())

        return data["data"][0]["embedding"]


if __name__ == "__main__":
    embedding = EmbeddingModel()
    vector = embedding.embed("Dolphins are amazing!")
    print(f"Received {len(vector)} dimensions from {embedding.model}")
    print(f"{vector}") 
