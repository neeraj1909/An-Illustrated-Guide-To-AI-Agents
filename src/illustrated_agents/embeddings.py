import json
import os
import resource
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from llm import benchmark_context, request_metrics, system_snapshot


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
        self.request_history: list[dict] = []
        self.last_metrics: dict = {}
        self.run_context = benchmark_context(self.base_url)

    def embed(self, text: str) -> list[float]:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json"},
        )

        system_before = system_snapshot()
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        started_ns = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read())
        finally:
            finished_ns = time.perf_counter_ns()
            usage_after = resource.getrusage(resource.RUSAGE_SELF)
            self.last_metrics = request_metrics(
                started_ns, finished_ns, usage_before, system_before, usage_after
            )
            self.request_history.append(self.last_metrics)

        for key in ("total_duration", "load_duration", "prompt_eval_count"):
            if key in data:
                self.last_metrics[key] = data[key]

        return data["data"][0]["embedding"]

    def metrics_summary(self) -> dict:
        """Return a small summary suitable for a benchmark log."""
        elapsed = [item["client_wall_ms"] for item in self.request_history]
        ordered = sorted(elapsed)
        p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] if ordered else None
        return {
            "model": self.model,
            "requests": len(elapsed),
            "total_client_wall_ms": round(sum(elapsed), 3),
            "mean_client_wall_ms": round(sum(elapsed) / len(elapsed), 3)
            if elapsed
            else None,
            "p95_client_wall_ms": round(p95, 3) if p95 is not None else None,
            "run_context": self.run_context,
            "request_history": self.request_history,
            "last_request": self.last_metrics,
        }


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
