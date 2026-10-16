# An Illustrated Guide to AI Agents

A small educational agent framework that sends a user task to an Ollama model through an OpenAI-compatible chat-completions endpoint. It also demonstrates LLM-backed conversation summarization and remote text embeddings.

The current flow is:

```text
main.py → TinyAgent → SummarizationMemory → LLM → Open WebUI or Ollama → model
embeddings.py → EmbeddingModel → embeddings endpoint
```

## Requirements

- Python 3.12 or newer
- Network access to the machine running Open WebUI or Ollama
- A model installed and available on that server

The project uses Python's standard library for HTTP requests. The only third-party dependency is `python-dotenv`, which loads the local `.env` file.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Ubuntu cannot create a virtual environment, install the missing package:

```bash
sudo apt install python3-venv
```

## Configuration

Copy or edit the repository-level `.env` file:

```env
OPENWEBUI_API_KEY="sk-your-openwebui-api-key"
OLLAMA_MODEL="exact-model-id"
LLM_BASE_URL="http://100.86.159.44:38080/api"

# Optional: used by embeddings.py
OLLAMA_BASE_URL="http://100.86.159.44:11434/v1"
OLLAMA_EMBEDDING_MODEL="embeddinggemma"
```

`main.py` loads this file automatically. The `.env` file is excluded from Git, so do not commit it or share its contents.

`embeddings.py` reads `OLLAMA_EMBEDDING_MODEL` and `OLLAMA_BASE_URL`. The latter should be the base URL before `/embeddings`; if it is not set, the module falls back to `LLM_BASE_URL`.

### Recommended: use Open WebUI

The default configuration calls Open WebUI's OpenAI-compatible endpoint:

```text
http://100.86.159.44:38080/api/chat/completions
```

This requires an Open WebUI API key. Browser sign-in is not enough because Python does not share Chrome's cookies.

Create a key in Open WebUI under **Settings → Account → API Keys**. If that section is missing, an administrator must enable API keys and grant your account permission.

Use the exact model ID shown by Open WebUI. Do not leave the placeholder value in `.env`.

Test the API key before running Python:

```bash
set -a
source .env
set +a

curl -i "$LLM_BASE_URL/models" -H "Authorization: Bearer $OPENWEBUI_API_KEY"
```

### Alternative: use Ollama directly

If Ollama itself is exposed on the network, use its OpenAI-compatible endpoint:

```env
OPENWEBUI_API_KEY="ollama"
OLLAMA_MODEL="exact-ollama-model-name"
LLM_BASE_URL="http://100.86.159.44:11434/v1"
```

The `ollama` value is only a placeholder accepted by the current client; direct Ollama normally does not require an API key.

Test direct Ollama access from the Ubuntu VM:

```bash
curl -i --max-time 5 http://100.86.159.44:11434/api/tags
```

If this times out, Ollama is not exposed to the network or a firewall is blocking port `11434`. Changing `.env` alone will not fix that.

## Run the example

From the repository root:

```bash
source .venv/bin/activate
cd src/illustrated_agents
python main.py
```

The example sends:

```text
What is 2 + 2?
```

and prints the model response together with the agent trajectory.

## Compare models and record timing

The existing clients record timing and host information for every generation and embedding request. `main.py` prints a JSON summary after the RAG example. Select one generation model and one embedding model through environment variables, then repeat the same command for each pair:

```bash
mkdir -p results
for embedding in embeddinggemma-2:270m-q8_0-text embeddinggemma:300m-qat-q4_0; do
  for model in llama3.2:3b-instruct-q4_K_M qwen3:4b-instruct-2507-q4_K_M; do
    for run in 1 2 3 4 5; do
      OLLAMA_MODEL="$model" \
      OLLAMA_EMBEDDING_MODEL="$embedding" \
      LLM_BASE_URL="http://127.0.0.1:11434/v1" \
      OLLAMA_BASE_URL="http://127.0.0.1:11434/v1" \
      PYTHONPATH=src python src/illustrated_agents/main.py \
        > "results/${model//\//_}_${embedding//\//_}_${run}.log"
    done
  done
done
```

Run this command on the Mac Ollama host and again on the CPU host. Running it beside Ollama is necessary for the RAM, swap, CPU load, and process metrics to describe the inference machine. Keep the same model tag, quantization, prompt, documents, temperature, and Ollama version on both systems. Run each pair several times and compare the median and p95 `client_wall_ms`; the summary also includes process CPU time, maximum resident memory, available system memory before and after the request, swap used before and after, token counts, and tokens per second when token counts are returned by the endpoint. The `request_history` fields are available if you want to build a CSV or notebook analysis later.

Each summary also includes `run_context`: `benchmark_location` comes from `BENCHMARK_LOCATION` in `.env`, while `client_hostname`, `client_platform`, `endpoint`, `endpoint_host`, and `endpoint_is_local` identify where the process ran and which Ollama endpoint it used. Set `BENCHMARK_LOCATION` to `mac-m3x-server` or `local-cpu` when switching machines. Older log files do not contain this field; rerun them after enabling the updated instrumentation.

Every successful application run also writes one structured JSON artifact to `results/`. The filename contains the UTC timestamp, benchmark location, model pair, and a unique suffix. Set `BENCHMARK_ARTIFACT_DIR` to choose another directory. The artifact contains the workload, run context, generation metrics, embedding metrics, RAG timings, and response, so later analysis does not need to parse terminal output.

These numbers measure the current application path. `client_wall_ms` includes HTTP overhead. If the request goes through Open WebUI or across the network, it also includes proxy and network time; direct local Ollama URLs make the Mac-versus-CPU comparison meaningful. Ollama's native timing fields are preserved when an OpenAI-compatible endpoint returns them.

The example uses `SummarizationMemory`, which replaces the accumulated conversation with a running system summary after each assistant response. `Memory` keeps the full conversation, while `TrimmingMemory` keeps the system message and the most recent two turns.

Alternatively, run from the repository root with:

```bash
source .venv/bin/activate
PYTHONPATH=src python src/illustrated_agents/main.py
```

## Generate an embedding

Configure `OLLAMA_BASE_URL` and `OLLAMA_EMBEDDING_MODEL` in `.env`, then run:

```bash
source .venv/bin/activate
PYTHONPATH=src python src/illustrated_agents/embeddings.py
```

The example sends `Dolphins are amazing!` to the configured embeddings endpoint and prints the returned vector dimensions and values.

## Project structure

```text
.
├── .env                         # Local configuration; ignored by Git
├── requirements.txt             # Python dependencies
├── src/
│   └── illustrated_agents/
│       ├── main.py              # Example entry point and configuration
│       ├── agent.py             # TinyAgent orchestration
│       ├── memory.py             # Full, trimmed, and summarized conversation memory
│       ├── embeddings.py         # Remote text embedding client
│       └── llm.py               # OpenAI-compatible HTTP client and trajectory types
└── README.md
```

## Troubleshooting

### `HTTP Error 401: Unauthorized`

The request reached Open WebUI, but authentication failed. Check that:

- `LLM_BASE_URL` points to the intended server.
- `OPENWEBUI_API_KEY` contains a real Open WebUI key, usually starting with `sk-`.
- The key was not replaced or revoked.
- The API-key feature is enabled for your account.

Browser login and API-key authentication are separate.

### `HTTP Error 400` or `404`

Check that `OLLAMA_MODEL` exactly matches a model available on the selected server and that the base URL is correct. The base URL should end at `/api` for Open WebUI or `/v1` for direct Ollama; the client appends `/chat/completions`.

### Connection timeout or connection refused

The Ubuntu VM cannot reach the configured host and port. Confirm that:

- The Mac is online and reachable over the network or VPN.
- The service is listening on the network interface, not only `localhost`.
- The relevant firewall port is allowed.

### `externally-managed-environment`

Do not install packages into Ubuntu's system Python. Activate `.venv` first and install with:

```bash
python -m pip install -r requirements.txt
```

## Development notes

The code intentionally uses `urllib.request` instead of an SDK so the request/response flow remains visible for the educational examples. `TinyAgent` currently performs one LLM step and records it in `Trajectory`. Memory strategies and embeddings are implemented as small, inspectable modules; tools and planning remain placeholders for later chapters.
