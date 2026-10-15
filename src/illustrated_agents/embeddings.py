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
    embedding_model = EmbeddingModel()
    
    # Create embeddings
    embedding_a = embedding_model.embed("I love flamingos.")
    embedding_b = embedding_model.embed("Dolphins use echolocation.")
    embedding_c = embedding_model.embed("Flamingos are pink birds.")
    
    # Calculate cosine similarity between A and B
    dot_ab = sum(x * y for x, y in zip(embedding_a, embedding_b))
    norm_a = sum(x * x for x in embedding_a) ** 0.5
    norm_b = sum(x * x for x in embedding_b) ** 0.5
    similarity_ab = dot_ab / (norm_a * norm_b)
    
    # Calculate cosine similarity between A and C
    dot_ac = sum(x * y for x, y in zip(embedding_a, embedding_c))
    norm_c = sum(x * x for x in embedding_c) ** 0.5
    similarity_ac = dot_ac / (norm_a * norm_c)
    print(f"Similarity between A and B: {similarity_ab}")
    print(f"Similarity between A and C: {similarity_ac}")
