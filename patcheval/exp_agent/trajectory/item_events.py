"""Project Codex/Trae JSON stdout; turn totals never become response usage."""

import json

from .records import (
    aggregate_usage,
    identify_messages,
    normalize_usage,
    usage_totals,
)

SHELL_TOOL = "shell"


def read_trajectory(
    work_dir, prompt, agent, *, cache_creation_field="cache_write_input_tokens"
):
    messages, metrics = messages_from_events(
        _read_events(work_dir / "agent_stdout.txt"),
        prompt,
        cache_creation_field=cache_creation_field,
    )
    records = [
        {"seq": seq, "event": "message", "agent": agent, "message": message}
        for seq, message in enumerate(identify_messages(messages), 1)
    ]
    return records, metrics


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


def _call_message(call_id, call_args, name=SHELL_TOOL):
    return {
        "role": "assistant",
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


def messages_from_events(
    events, prompt, *, cache_creation_field="cache_write_input_tokens"
):
    """Map shared item events while retaining each CLI's original usage vocabulary.

    Codex uses cache_write_input_tokens; Trae uses cache_creation_input_tokens.
    Both totals already include their cache and reasoning subsets. Turn totals
    remain in metrics because item events do not identify upstream responses.
    """
    messages = [{"role": "user", "id": "prompt", "content": prompt}] if prompt else []
    metrics = {"tool_calls": 0, "tokens": {}, "notices": []}
    usages = []
    source_usage = []
    pending_turn = False
    for event in events:
        kind = event.get("type")
        item = event.get("item") or {}
        if kind == "turn.started":
            if pending_turn:
                usages.append(None)
            pending_turn = True
        elif kind == "turn.failed":
            usages.append(None)
            pending_turn = False
            metrics["notices"].append(
                (event.get("error") or {}).get("message") or "Turn failed"
            )
        elif kind == "item.completed" and item.get("type") == "command_execution":
            call_id = item.get("id") or f"call-{metrics['tool_calls']}"
            messages.append(_call_message(call_id, {"command": item.get("command")}))
            messages.append(
                {
                    "role": "tool",
                    "id": f"{call_id}-result",
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
            if item.get("exit_code") is None:
                messages[-1].pop("status")
            metrics["tool_calls"] += 1
        elif kind == "item.completed" and item.get("type") == "file_change":
            call_id = item.get("id") or f"call-{metrics['tool_calls']}"
            changes = item.get("changes") or []
            messages.append(_call_message(call_id, {"changes": changes}, "apply_patch"))
            messages.append(
                {
                    "role": "tool",
                    "id": f"{call_id}-result",
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
            if item.get("status") not in {"completed", "failed"}:
                messages[-1].pop("status")
            metrics["tool_calls"] += 1
        elif kind == "item.completed" and item.get("type") in {
            "agent_message",
            "reasoning",
        }:
            messages.append(
                {
                    "role": "assistant",
                    "id": item.get("id"),
                    "content": [
                        {
                            "type": (
                                "reasoning" if item["type"] == "reasoning" else "text"
                            ),
                            "text": item.get("text") or "",
                        }
                    ],
                }
            )
        elif kind == "item.completed" and item.get("type") == "error":
            metrics["notices"].append(item.get("message") or "")
        elif kind == "turn.completed":
            pending_turn = False
            raw = event.get("usage") or {}
            # Codex input_tokens already includes the cached subset.
            usages.append(
                normalize_usage(
                    raw.get("input_tokens"),
                    raw.get("cached_input_tokens"),
                    raw.get(cache_creation_field),
                    raw.get("output_tokens"),
                    raw.get("reasoning_output_tokens"),
                    raw.get("total_tokens"),
                )
            )
            source_usage.append(raw)
    if pending_turn:
        usages.append(None)
        metrics["notices"].append("A recorded turn has no completion usage.")
    metrics["tokens"] = usage_totals(aggregate_usage(usages))
    metrics["source_usage"] = source_usage
    metrics["usage_scope"] = "recorded-turns"
    metrics["usage_complete"] = False
    if not any(usage is not None for usage in usages) and events:
        metrics["notices"].append(
            "No turn.completed usage received; token usage is unavailable."
        )
    return messages, metrics
