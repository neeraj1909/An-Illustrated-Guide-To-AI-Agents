# An Illustrated Guide to AI Agents

A small educational agent framework that sends a user task to an Ollama model through an OpenAI-compatible chat-completions endpoint.

The current flow is:

```text
main.py → TinyAgent → LLM → Open WebUI or Ollama → model
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
```

`main.py` loads this file automatically. The `.env` file is excluded from Git, so do not commit it or share its contents.

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

Alternatively, run from the repository root with:

```bash
source .venv/bin/activate
PYTHONPATH=src python src/illustrated_agents/main.py
```

## Project structure

```text
.
├── .env                         # Local configuration; ignored by Git
├── requirements.txt             # Python dependencies
├── src/
│   └── illustrated_agents/
│       ├── main.py              # Example entry point and configuration
│       ├── agent.py             # TinyAgent orchestration
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

The code intentionally uses `urllib.request` instead of an SDK so the request/response flow remains visible for the educational examples. `TinyAgent` currently performs one LLM step and records it in `Trajectory`; memory, tools, and planning are placeholders for later chapters.
