"""Normalize external CLI agent output into the repository trajectory schema.

The project records runs as JSONL lines of {"seq", "event", "message"}. External
agents emit their own event streams, so this module maps the two verified
formats (Codex item events, OpenCode session export) onto that schema and writes
one trajectory plus one metrics file per case.

Token accounting follows scripts/token_usage: input_tokens is the inclusive
prompt total, cache read/creation and reasoning are subsets, and uncached input
is derived. Both CLIs are converted to that one vocabulary so a field means the
same thing in every trajectory.
"""

import json
from pathlib import Path

SHELL_TOOL = "shell"


def build_trajectory(work_dir, agent_kind=None, model_provider=None, model_name=None):
    """Return (records, metrics) for one case work directory.

    model_provider/model_name are the deployment's provenance, not the CLI's
    internal provider key: the runner knows them at launch, but neither CLI puts
    the upstream vendor in its event stream.
    """
    work_dir = Path(work_dir)
    prompt_path = work_dir / "prompt.txt"
    prompt = (
        prompt_path.read_text(encoding="utf-8", errors="replace")
        if prompt_path.exists()
        else ""
    )
    export_path = work_dir / "opencode-session.json"
    kind = agent_kind or ("opencode" if export_path.exists() else "codex")
    if kind == "opencode":
        try:
            session = json.loads(export_path.read_text(encoding="utf-8"))
            notice = None
        except (FileNotFoundError, json.JSONDecodeError):
            session = _opencode_event_session(
                _read_events(work_dir / "agent_events.jsonl")
            )
            notice = (
                "OpenCode session export is missing or incomplete; recovered persisted "
                "events only. Trajectory and usage may be incomplete."
            )
        messages, metrics = opencode_messages(session, prompt)
        if notice:
            metrics["notices"].append(notice)
    else:
        messages, metrics = codex_messages(
            _read_events(work_dir / "agent_stdout.txt"), prompt
        )
    records = [
        {"seq": seq, "event": "message", "message": message}
        for seq, message in enumerate(messages, 1)
    ]
    if model_provider or model_name:
        for message in messages:
            if message["type"] != "ai":
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


def _read_events(path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _opencode_event_session(events):
    """Recover the session shape from completed parts persisted before interruption."""
    messages = {}
    totals = {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}}
    cost = None
    has_usage = False
    for event in events:
        part = event.get("part") or {}
        message_id = part.get("messageID")
        if not message_id:
            continue
        entry = messages.setdefault(
            message_id,
            {"info": {"role": "assistant", "id": message_id}, "parts": []},
        )
        if part.get("type") in {"text", "reasoning", "tool"}:
            entry["parts"].append(part)
        elif part.get("type") == "step-finish":
            raw = part.get("tokens") or {}
            if raw:
                entry["info"]["tokens"] = raw
                has_usage = True
                for key in ("input", "output", "reasoning"):
                    totals[key] += raw.get(key) or 0
                for key in ("read", "write"):
                    totals["cache"][key] += (raw.get("cache") or {}).get(key) or 0
            if part.get("cost") is not None:
                cost = (cost or 0) + part["cost"]
    return {
        "info": {"tokens": totals if has_usage else {}, "cost": cost},
        "messages": list(messages.values()),
    }


def _canonical_usage(
    input_tokens, cache_read, cache_creation, output, reasoning, total
):
    """Build the one usage shape every trajectory uses.

    input_tokens must be inclusive: cache read/creation are subsets, matching
    LangChain usage_metadata and scripts/token_usage. OpenCode reports uncached
    input separately, so its caller adds the cache values back in.
    """
    input_tokens = input_tokens or 0
    output = output or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output,
        "total_tokens": total if total is not None else input_tokens + output,
        "input_token_details": {
            "cache_read": cache_read or 0,
            "cache_creation": cache_creation or 0,
        },
        "output_token_details": {"reasoning": reasoning or 0},
    }


def _canonical_totals(usage):
    """Collapse a usage_metadata dict into the report vocabulary."""
    input_tokens = usage.get("input_tokens") or 0
    details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}
    cache_read = details.get("cache_read") or 0
    cache_creation = details.get("cache_creation") or 0
    return {
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "uncached_input_tokens": max(input_tokens - cache_read - cache_creation, 0),
        "output_tokens": usage.get("output_tokens") or 0,
        "reasoning_tokens": output_details.get("reasoning") or 0,
        "total_tokens": usage.get("total_tokens") or 0,
    }


def _call_message(call_id, call_args, name=SHELL_TOOL):
    return {
        "type": "ai",
        "id": f"{call_id}-call",
        "content": [
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": call_args,
            }
        ],
        # Viewers read tool_calls, not content blocks, to render a call and
        # link it to its result.
        "tool_calls": [
            {
                "name": name,
                "args": call_args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
        "invalid_tool_calls": [],
        "additional_kwargs": {},
    }


def codex_messages(events, prompt):
    """Map Codex item events onto trajectory messages."""
    messages = [{"type": "human", "id": "prompt", "content": prompt}] if prompt else []
    metrics = {"tool_calls": 0, "tokens": {}, "notices": []}
    usage = None
    for event in events:
        kind = event.get("type")
        item = event.get("item") or {}
        if kind == "item.completed" and item.get("type") == "command_execution":
            call_id = item.get("id") or f"call-{metrics['tool_calls']}"
            messages.append(_call_message(call_id, {"command": item.get("command")}))
            messages.append(
                {
                    "type": "tool",
                    "tool_call_id": call_id,
                    "name": SHELL_TOOL,
                    "content": item.get("aggregated_output") or "",
                    "status": "success" if item.get("exit_code") == 0 else "error",
                    "additional_kwargs": {
                        "exit_code": item.get("exit_code"),
                        "traceback": "",
                    },
                }
            )
            metrics["tool_calls"] += 1
        elif kind == "item.completed" and item.get("type") == "file_change":
            call_id = item.get("id") or f"call-{metrics['tool_calls']}"
            changes = item.get("changes") or []
            messages.append(_call_message(call_id, {"changes": changes}, "apply_patch"))
            messages.append(
                {
                    "type": "tool",
                    "tool_call_id": call_id,
                    "name": "apply_patch",
                    "content": json.dumps(changes, ensure_ascii=False),
                    "status": (
                        "success" if item.get("status") == "completed" else "error"
                    ),
                    "additional_kwargs": {
                        "status": item.get("status"),
                        "traceback": "",
                    },
                }
            )
            metrics["tool_calls"] += 1
        elif kind == "item.completed" and item.get("type") == "agent_message":
            messages.append(
                {
                    "type": "ai",
                    "id": item.get("id"),
                    "content": [{"type": "text", "text": item.get("text") or ""}],
                }
            )
        elif kind == "item.completed" and item.get("type") == "error":
            metrics["notices"].append(item.get("message") or "")
        elif kind == "turn.completed":
            raw = event.get("usage") or {}
            # Codex input_tokens already includes the cached subset.
            usage = _canonical_usage(
                raw.get("input_tokens"),
                raw.get("cached_input_tokens"),
                0,
                raw.get("output_tokens"),
                raw.get("reasoning_output_tokens"),
                None,
            )
    if usage:
        metrics["tokens"] = _canonical_totals(usage)
        _attach_aggregate_usage(messages, usage)
    elif events:
        metrics["notices"].append(
            "No turn.completed usage received; token usage is unavailable."
        )
    return messages, metrics


def opencode_messages(session, prompt):
    """Map an OpenCode session export onto trajectory messages.

    The export already contains the user turn, so the prompt file is only used
    when the session is empty.
    """
    messages = []
    metrics = {"tool_calls": 0, "tokens": {}, "cost": None, "notices": []}
    info = session.get("info") or {}
    raw = info.get("tokens") or {}
    if raw:
        cache = raw.get("cache") or {}
        cache_read = cache.get("read") or 0
        cache_creation = cache.get("write") or 0
        # OpenCode's input is uncached only; add cache back to get the
        # inclusive total the trajectory vocabulary expects. Reasoning is also
        # a separate OpenCode bucket and belongs inside inclusive output.
        usage = _canonical_usage(
            (raw.get("input") or 0) + cache_read + cache_creation,
            cache_read,
            cache_creation,
            (raw.get("output") or 0) + (raw.get("reasoning") or 0),
            raw.get("reasoning"),
            None,
        )
        metrics["tokens"] = _canonical_totals(usage)
    metrics["cost"] = info.get("cost")
    for entry in session.get("messages") or []:
        meta = entry.get("info") or {}
        parts = entry.get("parts") or []
        role = meta.get("role")
        if role == "user":
            text = "".join(
                part.get("text") or "" for part in parts if part.get("type") == "text"
            )
            messages.append({"type": "human", "id": meta.get("id"), "content": text})
            continue
        if role != "assistant":
            continue
        blocks = []
        tool_calls = []
        pending_tools = []
        for part in parts:
            kind = part.get("type")
            if kind == "text":
                blocks.append({"type": "text", "text": part.get("text") or ""})
            elif kind == "reasoning":
                blocks.append({"type": "thinking", "thinking": part.get("text") or ""})
            elif kind == "tool":
                state = part.get("state") or {}
                call_id = (
                    part.get("callID")
                    or part.get("id")
                    or f"call-{metrics['tool_calls']}"
                )
                name = part.get("tool") or part.get("name") or SHELL_TOOL
                call_args = state.get("input") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": call_args,
                    }
                )
                tool_calls.append(
                    {
                        "name": name,
                        "args": call_args,
                        "id": call_id,
                        "type": "tool_call",
                    }
                )
                metadata = state.get("metadata") or {}
                additional = {
                    "status": state.get("status"),
                    "traceback": "",
                }
                # OpenCode reports the real unified diff for edits and patches.
                # Carrying it is what lets the viewer show a diff instead of a
                # before/after pair or a raw argument dump.
                diff = metadata.get("diff")
                if isinstance(diff, str) and diff:
                    additional["diff"] = diff
                    additional["diff_truncated"] = bool(metadata.get("truncated"))
                pending_tools.append(
                    {
                        "type": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": (
                            state.get("error")
                            if state.get("status") == "error"
                            else state.get("output")
                        ) or "",
                        "status": (
                            "success"
                            if state.get("status") in (None, "completed")
                            else "error"
                        ),
                        "additional_kwargs": additional,
                    }
                )
                metrics["tool_calls"] += 1
        message = {
            "type": "ai",
            "id": meta.get("id"),
            "content": blocks,
            "tool_calls": tool_calls,
            "invalid_tool_calls": [],
            "additional_kwargs": {},
        }
        raw_usage = meta.get("tokens") or {}
        if raw_usage:
            cache = raw_usage.get("cache") or {}
            cache_read = cache.get("read") or 0
            cache_creation = cache.get("write") or 0
            message["usage_metadata"] = _canonical_usage(
                (raw_usage.get("input") or 0) + cache_read + cache_creation,
                cache_read,
                cache_creation,
                (raw_usage.get("output") or 0) + (raw_usage.get("reasoning") or 0),
                raw_usage.get("reasoning"),
                raw_usage.get("total"),
            )
        messages.append(message)
        messages.extend(pending_tools)
    if prompt and not any(message["type"] == "human" for message in messages):
        messages.insert(0, {"type": "human", "id": "prompt", "content": prompt})
    return messages, metrics


def _attach_aggregate_usage(messages, usage):
    for message in reversed(messages):
        if message["type"] == "ai":
            message["usage_metadata"] = usage
            message.setdefault("response_metadata", {})["usage_scope"] = "run_aggregate"
            return
