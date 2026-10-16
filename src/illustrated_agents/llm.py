from dataclasses import dataclass
import json
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse


def _sysctl_int(name: str) -> int | None:
    """Read a numeric macOS system value when available."""
    if sys.platform != "darwin":
        return None
    try:
        output = subprocess.check_output(["sysctl", "-n", name], text=True, timeout=2)
        return int(output.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def system_snapshot() -> dict:
    """Return portable host metrics for associating with one model request."""
    snapshot = {
        "system_cpu_count": os.cpu_count(),
        "system_load_1m": None,
        "system_total_memory_bytes": None,
        "system_available_memory_bytes": None,
        "system_swap_used_bytes": None,
        "system_platform": platform.platform(),
    }
    try:
        snapshot["system_load_1m"] = os.getloadavg()[0]
    except OSError:
        pass

    if sys.platform.startswith("linux"):
        values = {}
        try:
            with open("/proc/meminfo", encoding="utf-8") as meminfo:
                for line in meminfo:
                    key, value = line.split(":", 1)
                    values[key] = int(value.strip().split()[0]) * 1024
        except (OSError, ValueError):
            values = {}
        snapshot["system_total_memory_bytes"] = values.get("MemTotal")
        snapshot["system_available_memory_bytes"] = values.get("MemAvailable")
        if values.get("SwapTotal") is not None and values.get("SwapFree") is not None:
            snapshot["system_swap_used_bytes"] = (
                values["SwapTotal"] - values["SwapFree"]
            )
    elif sys.platform == "darwin":
        snapshot["system_total_memory_bytes"] = _sysctl_int("hw.memsize")
        try:
            vm_stat = subprocess.check_output(["vm_stat"], text=True, timeout=2)
            page_size = _sysctl_int("vm.pagesize") or 4096
            pages = {}
            for line in vm_stat.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                pages[key.strip()] = int(value.strip().rstrip("."))
            available_pages = sum(
                pages.get(name, 0)
                for name in ("Pages free", "Pages inactive", "Pages speculative")
            )
            snapshot["system_available_memory_bytes"] = available_pages * page_size
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        try:
            swap = subprocess.check_output(
                ["sysctl", "-n", "vm.swapusage"], text=True, timeout=2
            )
            match = re.search(r"used = ([0-9.]+)([KMG]?)", swap)
            if match:
                multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[
                    match.group(2)
                ]
                snapshot["system_swap_used_bytes"] = int(
                    float(match.group(1)) * multiplier
                )
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    return snapshot


def benchmark_context(base_url: str) -> dict:
    """Identify where the benchmark process ran and where it sent requests."""
    endpoint_host = urlparse(base_url).hostname
    is_local_endpoint = endpoint_host in {"localhost", "127.0.0.1", "::1"}
    return {
        "benchmark_location": os.getenv(
            "BENCHMARK_LOCATION", "local" if is_local_endpoint else "remote"
        ),
        "client_hostname": socket.gethostname(),
        "client_platform": platform.platform(),
        "endpoint": base_url,
        "endpoint_host": endpoint_host,
        "endpoint_is_local": is_local_endpoint,
    }


def request_metrics(
    start_ns: int,
    end_ns: int,
    usage_before: resource.struct_rusage,
    system_before: dict,
    usage_after: resource.struct_rusage,
) -> dict:
    """Calculate client time, process CPU/RSS, and host resource deltas."""
    rss = usage_after.ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    system_after = system_snapshot()
    return {
        "client_wall_ms": round((end_ns - start_ns) / 1_000_000, 3),
        "process_user_cpu_ms": round(
            (usage_after.ru_utime - usage_before.ru_utime) * 1000, 3
        ),
        "process_system_cpu_ms": round(
            (usage_after.ru_stime - usage_before.ru_stime) * 1000, 3
        ),
        "process_max_rss_bytes": rss,
        "system_cpu_count": system_after["system_cpu_count"],
        "system_load_1m_before": system_before["system_load_1m"],
        "system_load_1m_after": system_after["system_load_1m"],
        "system_total_memory_bytes": system_after["system_total_memory_bytes"],
        "system_available_memory_before_bytes": system_before[
            "system_available_memory_bytes"
        ],
        "system_available_memory_after_bytes": system_after[
            "system_available_memory_bytes"
        ],
        "system_swap_used_before_bytes": system_before["system_swap_used_bytes"],
        "system_swap_used_after_bytes": system_after["system_swap_used_bytes"],
        "system_platform": system_after["system_platform"],
    }


@dataclass
class Response:
    """Structured reponse from LLM calls."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


@dataclass
class Step:
    """A single step in an agent's trajectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class Trajectory:
    """Records agent execution as a sequence of runs."""

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run with the given query."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """Record a step from a Response, optionally with an observation."""
        # Add THOUGHT
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # Add ACTION/OBSERVATION or ANSWER
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "no_key",
        think: bool = False,
    ):
        """Initialize the LLM with the given model."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think
        self.request_history: list[dict] = []
        self.last_metrics: dict = {}
        self.run_context = benchmark_context(self.base_url)

    def generate(self, messages: list[dict], tools: list | None = None) -> Response:
        """Generate a response from the LLM given a list of messages."""
        # Build the request body
        body = {
            "model": self.model,
            "messages": messages,
        }

        # Tools and Reasoning
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"

        # POST to the OpenAI-compatible /chat/completions endpoint
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
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

        # Extract message, tool_call, and metadata
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        usage = data.get("usage", {})
        metadata = {
            "model": data.get("model", self.model),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "timing": self.last_metrics,
        }
        if metadata["completion_tokens"] and self.last_metrics["client_wall_ms"]:
            metadata["client_tokens_per_second"] = round(
                metadata["completion_tokens"]
                * 1000
                / self.last_metrics["client_wall_ms"],
                3,
            )
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ):
            if key in data:
                metadata[key] = data[key]

        # Format as Response dataclass
        return Response(
            content=message.get("content"),
            reasoning=message.get("reasoning"),
            tool_call=tool_call,
            metadata=metadata,
        )

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
