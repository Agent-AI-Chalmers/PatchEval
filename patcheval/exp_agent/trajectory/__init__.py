"""Export independent CLI recordings as agent-transcript/v1 messages and metrics.

Source-specific readers own capture recovery and usage observations. This entry
point selects the reader, adds deployment provenance, and writes case artifacts.
"""

import json
from pathlib import Path

from . import codex, opencode, trae


def build_trajectory(work_dir, agent_kind=None, model_provider=None, model_name=None):
    """Return (records, metrics) for one case work directory.

    model_provider/model_name are the deployment's provenance, not the CLI's
    internal provider key: the runner knows them at launch, but neither CLI puts
    the upstream vendor in its event stream.
    """
    if agent_kind is not None and agent_kind not in {"codex", "traecli", "opencode"}:
        raise ValueError(f"Unsupported trajectory agent: {agent_kind!r}")
    work_dir = Path(work_dir)
    prompt_path = work_dir / "prompt.txt"
    prompt = (
        prompt_path.read_text(encoding="utf-8", errors="replace")
        if prompt_path.exists()
        else ""
    )
    kind = (
        agent_kind
        if agent_kind is not None
        else (
            "opencode"
            if any(
                (work_dir / name).exists()
                for name in ("opencode-session.json", "opencode.db")
            )
            else "traecli" if (work_dir / "trae-sessions").exists() else "codex"
        )
    )
    if kind == "opencode":
        records, metrics = opencode.read_trajectory(work_dir, prompt)
    elif kind == "traecli":
        records, metrics = trae.read_trajectory(work_dir, prompt)
    else:
        records, metrics = codex.read_trajectory(work_dir, prompt, kind)
    messages = [record["message"] for record in records]
    metrics["tool_calls"] = sum(
        len(message.get("tool_calls", [])) for message in messages
    )
    if model_provider or model_name:
        for message in messages:
            if message["role"] != "assistant":
                continue
            metadata = message.setdefault("response_metadata", {})
            if model_provider:
                metadata["model_provider"] = model_provider
            if model_name:
                metadata["model_name"] = model_name
    metrics["agent"] = kind
    metrics["model_provider"] = model_provider
    metrics["model_name"] = model_name
    metrics["messages"] = len(messages)
    return records, metrics


def write_trajectory(
    work_dir, output_dir, cve, agent_kind=None, model_provider=None, model_name=None
):
    """Write <cve>.trajectory.jsonl and <cve>.metrics.json; return their paths."""
    records, metrics = build_trajectory(
        work_dir, agent_kind, model_provider, model_name
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / f"{cve}.trajectory.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics_path = output_dir / f"{cve}.metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return trajectory_path, metrics_path
