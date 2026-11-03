"""Recover OpenCode's owned session messages and normalize its exclusive buckets."""

import json
import sqlite3
from contextlib import closing

from .records import (
    aggregate_usage,
    identify_messages,
    normalize_usage,
    read_jsonl_capture,
    usage_totals,
    validate_token_count,
)

SHELL_TOOL = "shell"


def read_trajectory(work_dir, prompt):
    """Prefer the durable database, then the root export, then interrupted events."""
    database = work_dir / "opencode.db"
    if database.exists():
        return _opencode_records(_opencode_database(database), prompt)
    try:
        session = json.loads(
            (work_dir / "opencode-session.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        event_path = work_dir / "agent_events.jsonl"
        events, partial = (
            read_jsonl_capture(event_path, "OpenCode event stream")
            if event_path.exists()
            else ([], False)
        )
        records, metrics = _opencode_records([_opencode_event_session(events)], prompt)
        metrics["notices"].append(
            "OpenCode session export is missing or incomplete; recovered persisted "
            "events only. Trajectory and usage may be incomplete."
        )
        if partial:
            metrics["tokens"] = usage_totals(None)
            metrics["notices"].append(
                "OpenCode event capture ends with an incomplete record."
            )
        return records, metrics
    return _opencode_records([session], prompt)


def _opencode_event_session(events):
    """Recover the session shape from completed parts persisted before interruption."""
    messages = {}
    cost = None
    observed_parts = {}
    for event in events:
        part = event.get("part") or {}
        message_id = part.get("messageID")
        if not message_id:
            continue
        # Persisted updates with the same part identity describe one observation.
        part_id = part.get("id")
        if part_id:
            observed_parts[(message_id, part_id)] = part
        else:
            observed_parts[(message_id, len(observed_parts))] = part
    for part in observed_parts.values():
        message_id = part["messageID"]
        entry = messages.setdefault(
            message_id,
            {"info": {"role": "assistant", "id": message_id}, "parts": []},
        )
        entry["parts"].append(part)
        if part.get("type") == "step-finish":
            raw = part.get("tokens") or {}
            if raw:
                entry["info"]["tokens"] = raw
            entry["info"]["finish"] = part.get("reason")
            if part.get("cost") is not None:
                cost = (cost or 0) + part["cost"]
    return {
        "info": {"cost": cost},
        "messages": list(messages.values()),
    }


def _opencode_database(path):
    """Read this case's durable SQLite snapshot, including child sessions.

    OpenCode's JSON stdout and root-session export omit child sessions. Its
    database is written during execution, so it also survives a killed CLI.
    Read message ownership from the storage rows rather than copying forked
    context into another session's accounting. Auth/config stay outside this DB.
    """
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        sessions = {}
        for row in db.execute("SELECT * FROM session ORDER BY time_created, id"):
            row = dict(row)
            info = {
                "id": row["id"],
                "parentID": row.get("parent_id"),
                "cost": row.get("cost"),
                "tokens": {
                    "input": row.get("tokens_input"),
                    "output": row.get("tokens_output"),
                    "reasoning": row.get("tokens_reasoning"),
                    "cache": {
                        "read": row.get("tokens_cache_read"),
                        "write": row.get("tokens_cache_write"),
                    },
                },
                "database_row": row,
            }
            sessions[row["id"]] = {"info": info, "messages": []}
        messages = {}
        for row in db.execute("SELECT * FROM message ORDER BY time_created, id"):
            info = {
                **json.loads(row["data"]),
                "id": row["id"],
                "sessionID": row["session_id"],
            }
            entry = {"info": info, "parts": []}
            if row["session_id"] not in sessions:
                raise ValueError("OpenCode message has no owning session")
            sessions[row["session_id"]]["messages"].append(entry)
            messages[row["id"]] = entry
        for row in db.execute("SELECT * FROM part ORDER BY time_created, id"):
            entry = messages.get(row["message_id"])
            if entry is None or entry["info"]["sessionID"] != row["session_id"]:
                raise ValueError("OpenCode part has no matching owning message")
            entry["parts"].append(
                {
                    **json.loads(row["data"]),
                    "id": row["id"],
                    "messageID": row["message_id"],
                    "sessionID": row["session_id"],
                }
            )
        return list(sessions.values())


def _opencode_records(sessions, prompt):
    """Project owned messages once, retaining session ancestry and source evidence."""
    if not sessions:
        sessions = [{"info": {}, "messages": []}]
    records = []
    usages = []
    sources = []
    costs = []
    notices = []
    parents = {
        session["info"]["id"]: session["info"]["parentID"]
        for session in sessions
        if session.get("info", {}).get("parentID")
    }
    for session in sessions:
        info = session.get("info") or {}
        identity = info.get("id")
        owned = [
            entry
            for entry in session.get("messages") or []
            if entry.get("info", {}).get("sessionID") in {None, identity}
        ]
        messages, metrics = opencode_messages(
            {**session, "messages": owned}, prompt if not info.get("parentID") else ""
        )
        ancestry = []
        parent = identity
        while parent in parents:
            parent = parents[parent]
            if parent == identity or parent in ancestry:
                raise ValueError("Cyclic OpenCode session ancestry")
            ancestry.insert(0, parent)
        for message in messages:
            record = {
                "seq": len(records) + 1,
                "event": "message",
                "agent": "opencode",
                "message": message,
            }
            if identity:
                record["run_id"] = identity
            if ancestry:
                record["parent_ids"] = ancestry
            if message["role"] == "assistant":
                record["call_id"] = message["id"]
                usages.append(message.get("usage_metadata"))
                if (
                    message.get("response_metadata", {})
                    .get("opencode_message", {})
                    .get("summary")
                ):
                    record["source"] = "summarization"
            records.append(record)
        sources.append({"session_id": identity, "usage": metrics["source_usage"]})
        costs.append(metrics["cost"])
        notices.extend(metrics["notices"])
    return records, {
        "tokens": usage_totals(aggregate_usage(usages)),
        "source_usage": sources,
        "cost": (
            sum(costs) if costs and all(cost is not None for cost in costs) else None
        ),
        "usage_scope": "recorded-responses",
        "usage_complete": False,
        "model_responses_with_usage_count": sum(
            r["message"].get("usage_metadata") is not None
            for r in records
            if r["message"]["role"] == "assistant"
        ),
        "notices": notices
        + [
            "OpenCode recordings omit upstream request starts and may omit auxiliary requests such as title generation."
        ],
    }


def _opencode_usage(raw):
    """OpenCode reports ordinary input/output separately from cache/reasoning.

    Absent optional buckets add nothing to an adapter-specific inclusive sum,
    but remain unknown as measurements. An absent ordinary bucket leaves the
    corresponding inclusive total unknown.
    """
    cache = raw.get("cache") or {}
    ordinary_input = validate_token_count(raw.get("input"))
    ordinary_output = validate_token_count(raw.get("output"))
    read = validate_token_count(cache.get("read"))
    creation = validate_token_count(cache.get("write"))
    reasoning = validate_token_count(raw.get("reasoning"))
    return normalize_usage(
        input_tokens=(
            None
            if ordinary_input is None
            else ordinary_input + (read or 0) + (creation or 0)
        ),
        cache_read=read,
        cache_creation=creation,
        output=None if ordinary_output is None else ordinary_output + (reasoning or 0),
        reasoning=reasoning,
        total=raw.get("total"),
    )


def opencode_messages(session, prompt):
    """Map an OpenCode session export onto trajectory messages.

    The export already contains the user turn, so the prompt file is only used
    when the session is empty. Source message/part metadata is retained rather
    than discarded by the display projection. A newly created assistant message
    has zero-filled counters before any response completes; those defaults do
    not establish measured zero usage, even if the session has aggregate totals.
    """
    messages = []
    metrics = {"tool_calls": 0, "tokens": {}, "cost": None, "notices": []}
    info = session.get("info") or {}
    raw = info.get("tokens") or {}
    metrics["source_usage"] = raw
    metrics["usage_complete"] = False
    metrics["cost"] = info.get("cost")
    for entry in session.get("messages") or []:
        meta = entry.get("info") or {}
        parts = entry.get("parts") or []
        role = meta.get("role")
        if role == "user":
            content = (
                "".join(part.get("text") or "" for part in parts)
                if all(part.get("type") == "text" for part in parts)
                else parts
            )
            messages.append(
                {
                    "role": "user",
                    "id": meta.get("id"),
                    "content": content,
                    "response_metadata": {
                        "opencode_message": meta,
                        "opencode_parts": parts,
                    },
                }
            )
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
                metrics["tool_calls"] += 1
                if state.get("status") not in {"completed", "error"}:
                    continue
                pending_tools.append(
                    {
                        "role": "tool",
                        "id": f"{call_id}-result",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": (
                            state.get("error")
                            if state.get("status") == "error"
                            else state.get("output")
                        )
                        or "",
                        "status": (
                            "success" if state.get("status") == "completed" else "error"
                        ),
                        "additional_kwargs": additional,
                        "response_metadata": {"opencode_part": part},
                    }
                )
            elif kind not in {"step-start", "step-finish"}:
                blocks.append(dict(part))
        message = {
            "role": "assistant",
            "id": meta.get("id"),
            "content": blocks,
            "tool_calls": tool_calls,
            "invalid_tool_calls": [],
            "additional_kwargs": {},
            "response_metadata": {"opencode_message": meta, "opencode_parts": parts},
        }
        raw_usage = meta.get("tokens") or {}
        completed_steps = [part for part in parts if part.get("type") == "step-finish"]
        completed_step = bool(completed_steps)
        if completed_step:
            raw_usage = completed_steps[-1].get("tokens") or {}
        unfinished = not completed_step and (
            meta.get("error") is not None
            or ("time" in meta and (meta.get("time") or {}).get("completed") is None)
        )
        if raw_usage and not unfinished:
            message["usage_metadata"] = _opencode_usage(raw_usage)
        if raw_usage:
            message["response_metadata"]["token_usage"] = raw_usage
        messages.append(message)
        messages.extend(pending_tools)
    if prompt and not any(message["role"] == "user" for message in messages):
        messages.insert(0, {"role": "user", "id": "prompt", "content": prompt})
    messages = identify_messages(messages)
    if any(
        message["role"] == "assistant" and message.get("usage_metadata") is None
        for message in messages
    ):
        metrics["notices"].append(
            "A recorded assistant response has no completed usage measurement."
        )
    metrics["usage_scope"] = "recorded-responses"
    metrics["tokens"] = usage_totals(
        aggregate_usage(
            [
                message.get("usage_metadata")
                for message in messages
                if message["role"] == "assistant"
            ]
        )
    )
    return messages, metrics
