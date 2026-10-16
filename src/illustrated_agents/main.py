import json
import os
from pathlib import Path
from datetime import datetime, timezone
import uuid

from dotenv import load_dotenv

from agent import TinyAgent
from llm import LLM
from memory import RAGMemory
from embeddings import EmbeddingModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _artifact_slug(value: str) -> str:
    """Make a model or host label safe to use in an artifact filename."""
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def write_benchmark_artifact(
    *,
    response: str,
    query: str,
    documents: list[str],
    llm: LLM,
    embedding_model: EmbeddingModel,
    memory: RAGMemory,
) -> Path:
    """Persist one structured benchmark artifact for this application run."""
    recorded_at = datetime.now(timezone.utc)
    location = llm.run_context["benchmark_location"]
    filename = (
        f"run_{recorded_at.strftime('%Y%m%dT%H%M%S.%fZ')}_"
        f"{_artifact_slug(location)}_"
        f"{_artifact_slug(llm.model)}_"
        f"{_artifact_slug(embedding_model.model)}_"
        f"{uuid.uuid4().hex[:8]}.json"
    )
    configured_dir = os.getenv("BENCHMARK_ARTIFACT_DIR")
    artifact_dir = (
        Path(configured_dir)
        if configured_dir
        else Path(__file__).resolve().parents[2] / "results"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / filename
    artifact = {
        "schema_version": 1,
        "status": "completed",
        "recorded_at_utc": recorded_at.isoformat(),
        "workload": {
            "query": query,
            "document_count": len(documents),
        },
        "run_context": {
            "generation": llm.run_context,
            "embedding": embedding_model.run_context,
        },
        "generation": llm.metrics_summary(),
        "embedding": embedding_model.metrics_summary(),
        "rag": {
            "index": memory.index_metrics,
            "last_search": memory.last_search_metrics,
        },
        "response": response,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact_path


llm = LLM(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
    # Open WebUI exposes an OpenAI-compatible endpoint at /api.
    # For direct Ollama, use http://<host>:11434/v1 instead.
    base_url=os.getenv("LLM_BASE_URL", "http://100.86.159.44:38080/api"),
    api_key=os.getenv("OPENWEBUI_API_KEY", "no_key"),
)

# Initialize EmbeddingGemma
embedding_model = EmbeddingModel(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
)

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

query = "What is Sarah's favorite animal?"
response = agent.run(query)
artifact_path = write_benchmark_artifact(
    response=response,
    query=query,
    documents=documents,
    llm=llm,
    embedding_model=embedding_model,
    memory=memory,
)
print(f"Response: {response}\n")
print(f"Summarized Conversation: {agent.memory.get_messages()}")
print(f"Benchmark artifact: {artifact_path}")
print("Generation metrics:")
print(json.dumps(llm.metrics_summary(), indent=2))
print("Embedding metrics:")
print(json.dumps(embedding_model.metrics_summary(), indent=2))
print("RAG timing metrics:")
print(
    json.dumps(
        {
            "index": memory.index_metrics,
            "last_search": memory.last_search_metrics,
        },
        indent=2,
    )
)
