"""Normalize external CLI agent output into the repository trajectory schema.

The project records runs as JSONL lines of {"seq", "event", "message"}. External
agents emit their own event streams, so this module maps the two verified
formats (Codex item events, OpenCode session export) onto that schema and writes
one trajectory plus one metrics file per case.

Token meanings and message records follow the agent-transcript contract.
This adapter is independent of the agents runtime and experiment reporting code.
CLI observations may lack model-call boundaries; message-only transcripts and
separately scoped aggregate metrics preserve that limitation.
"""

import json
import warnings
from pathlib import Path

SHELL_TOOL = "shell"
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
)
MAX_TOKEN_COUNT = 9007199254740991


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
    rollout_paths = (
        sorted((work_dir / "codex-sessions").rglob("rollout-*.jsonl"))
        if kind == "codex"
        else []
    )
    if rollout_paths:
        records, metrics = _codex_rollouts(rollout_paths, kind)
        messages = [record["message"] for record in records]
    elif kind == "opencode":
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
    messages = _identify_messages(messages)
    metrics["tool_calls"] = sum(
        len(message.get("tool_calls", [])) for message in messages
    )
    if not rollout_paths:
        records = [
            {"seq": seq, "event": "message", "agent": kind, "message": message}
            for seq, message in enumerate(messages, 1)
        ]
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
        if part.get("type") in {"text", "reasoning", "tool"}:
            entry["parts"].append(part)
        elif part.get("type") == "step-finish":
            raw = part.get("tokens") or {}
            if raw:
                entry["info"]["tokens"] = raw
            if part.get("cost") is not None:
                cost = (cost or 0) + part["cost"]
    return {
        "info": {"cost": cost},
        "messages": list(messages.values()),
    }


def _count(value):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_TOKEN_COUNT:
        raise ValueError("Token counts must be nonnegative safe integers or null")
    return value


def _canonical_usage(
    input_tokens, cache_read, cache_creation, output, reasoning, total
):
    """Normalize inclusive totals and subsets without inventing measurements.

    The caller owns source-specific addition of exclusive buckets. A reported
    total is evidence only: canonical total is derived from known components.
    """
    input_tokens, cache_read, cache_creation, output, reasoning = (
        _count(value)
        for value in (input_tokens, cache_read, cache_creation, output, reasoning)
    )
    derived = (
        _count(input_tokens + output)
        if input_tokens is not None and output is not None
        else None
    )
    reported = _count(total)
    if reported is not None and derived is not None and reported != derived:
        warnings.warn("transcript_source_total_mismatch", RuntimeWarning, stacklevel=2)
    if (
        input_tokens is not None
        and (cache_read or 0) + (cache_creation or 0) > input_tokens
    ):
        raise ValueError("Cache subsets exceed inclusive input")
    if output is not None and reasoning is not None and reasoning > output:
        raise ValueError("Reasoning subset exceeds inclusive output")
    usage = dict(
        zip(
            TOKEN_FIELDS,
            (
                input_tokens,
                output,
                derived,
                cache_read,
                cache_creation,
                reasoning,
            ),
        )
    )
    return usage if any(value is not None for value in usage.values()) else None


def _opencode_usage(raw):
    """OpenCode reports ordinary input/output separately from cache/reasoning.

    Absent optional buckets add nothing to an adapter-specific inclusive sum,
    but remain unknown as measurements. An absent ordinary bucket leaves the
    corresponding inclusive total unknown.
    """
    cache = raw.get("cache") or {}
    ordinary_input = _count(raw.get("input"))
    ordinary_output = _count(raw.get("output"))
    read = _count(cache.get("read"))
    creation = _count(cache.get("write"))
    reasoning = _count(raw.get("reasoning"))
    return _canonical_usage(
        (
            None
            if ordinary_input is None
            else ordinary_input + (read or 0) + (creation or 0)
        ),
        read,
        creation,
        None if ordinary_output is None else ordinary_output + (reasoning or 0),
        reasoning,
        raw.get("total"),
    )


def _aggregate_usage(usages):
    """Sum a field only if every covered observation reports it."""
    return {
        field: (
            sum(usage[field] for usage in usages)
            if usages
            and all(usage is not None and usage[field] is not None for usage in usages)
            else None
        )
        for field in TOKEN_FIELDS
    }


def _canonical_totals(usage):
    result = dict(usage) if usage else dict.fromkeys(TOKEN_FIELDS)
    operands = [
        result[field]
        for field in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    ]
    result["uncached_input_tokens"] = (
        operands[0] - operands[1] - operands[2]
        if all(value is not None for value in operands)
        else None
    )
    return result


def _identify_messages(messages):
    """Retain source identities, assigning local IDs only when unavailable.

    Repeated identical observations are stored once; conflicting observations
    of the same identity are rejected rather than silently renamed.
    """
    reserved = {message.get("id") for message in messages if message.get("id")}
    seen = {}
    for index, message in enumerate(messages, 1):
        identity = message.get("id")
        if identity is None or identity == "":
            identity = f"patcheval-message-{index}"
            while identity in reserved:
                identity += "-local"
            message["id"] = identity
            reserved.add(identity)
        if not isinstance(identity, str):
            raise TypeError("Message identity must be a string")
        if not identity.strip():
            raise ValueError("Message identity must not be blank")
        if identity in seen and seen[identity] != message:
            raise ValueError("Conflicting observations of the same message")
        seen[identity] = message
    return list(seen.values())


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


def _read_rollout(path):
    """Keep valid records from an interrupted tail; reject corrupt complete lines."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    events = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                return events, True
            raise ValueError(f"Malformed Codex rollout: {path.name}:{index + 1}")
        if not isinstance(event, dict):
            raise TypeError("Codex rollout records must be objects")
        events.append(event)
    return events, False


def _codex_rollout_records(events, agent, filename):
    """Project one sequential rollout, using response completions as delimiters.

    A completion's response_id identifies one upstream response; its preceding
    assistant output items form that response, including reasoning and tool
    requests. Tool results and new inputs cannot belong to that output. Flush
    unfinished output without usage before crossing such a boundary. Do not use
    token_count snapshots: their last/total values repeat completion usage.
    """
    # Forked rollouts can copy parent history after their own session meta.
    # The filename identifies the owning session; thread-settings markers switch
    # back from inherited history to this session's execution.
    owners = [
        event.get("payload", {}).get("id") or event.get("payload", {}).get("session_id")
        for event in events
        if event.get("type") == "session_meta"
    ]
    owner = next(
        (identity for identity in owners if identity and filename.endswith(identity)),
        owners[-1] if owners else filename,
    )
    active_thread = owner
    records = []
    pending = []
    usages = []
    source_usage = []
    thread = owner
    turn = None
    parent = None
    session_source = None
    pending_turns = set()
    seen_responses = {}

    def append(message, **extra):
        envelope = {"event": "message", "agent": agent, "run_id": thread}
        envelope["metadata"] = {"codex_source": session_source, "turn_id": turn}
        if parent:
            envelope["parent_ids"] = [parent]
        envelope.update(extra)
        message = dict(message)
        original_id = message.get("id")
        message["id"] = f"{thread}:{original_id or 'item-' + str(len(records) + 1)}"
        records.append({**envelope, "message": message})

    def response(completion=None):
        nonlocal pending
        if not pending and completion is None:
            return
        blocks = []
        calls = []
        for item in pending:
            kind = item.get("type")
            if kind == "message":
                blocks.extend(item.get("content") or [])
            else:
                blocks.append(dict(item))
            if kind in {"function_call", "custom_tool_call"}:
                arguments = item.get("arguments")
                if kind == "function_call" and isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"arguments": arguments}
                elif kind == "custom_tool_call":
                    arguments = {"input": item.get("input")}
                calls.append(
                    {
                        "id": item.get("call_id"),
                        "name": item.get("name"),
                        "args": arguments,
                        "type": "tool_call",
                    }
                )
        message = {
            "role": "assistant",
            "content": blocks,
            "tool_calls": calls,
            "response_metadata": {"response_items": pending},
        }
        extra = {}
        if any(
            item.get("type") in {"compaction", "context_compaction"} for item in pending
        ):
            extra["source"] = "summarization"
        if completion is not None:
            identity = completion["response_id"]
            raw = completion.get("usage") or {}
            usage = _canonical_usage(
                raw.get("input_tokens"),
                raw.get("cached_input_tokens"),
                raw.get("cache_write_input_tokens"),
                raw.get("output_tokens"),
                raw.get("reasoning_output_tokens"),
                raw.get("total_tokens"),
            )
            message.update(id=identity, usage_metadata=usage)
            message["response_metadata"]["token_usage_record"] = completion
            extra["call_id"] = f"{thread}:{identity}"
            usages.append(usage)
            source_usage.append(completion)
        else:
            usages.append(None)
        append(message, **extra)
        pending = []

    for event in events:
        payload = event.get("payload") or {}
        kind = event.get("type")
        if kind == "session_meta":
            identity = payload.get("id") or payload.get("session_id")
            active_thread = identity or owner
            if active_thread != owner:
                continue
            source = payload.get("source") or {}
            session_source = source
            if isinstance(source, dict):
                spawn = source.get("subagent", {})
                if isinstance(spawn, dict):
                    parent = (spawn.get("thread_spawn") or {}).get("parent_thread_id")
            instructions = payload.get("base_instructions") or {}
            text = (
                instructions.get("text")
                if isinstance(instructions, dict)
                else instructions
            )
            if text and not records:
                append({"id": "instructions", "role": "system", "content": text})
        elif kind == "event_msg" and payload.get("type") == "thread_settings_applied":
            active_thread = payload.get("thread_id") or active_thread
        elif active_thread != owner:
            continue
        elif kind == "event_msg":
            if payload.get("type") == "task_started":
                turn = payload.get("turn_id")
                pending_turns.add(turn)
            elif payload.get("type") in {
                "task_complete",
                "task_completed",
                "turn_aborted",
            }:
                response()
                pending_turns.discard(payload.get("turn_id") or turn)
                if payload.get("type") == "turn_aborted":
                    usages.append(None)
        elif kind == "turn_context":
            turn = payload.get("turn_id") or turn
        elif kind == "response_item":
            item_kind = payload.get("type")
            role = payload.get("role")
            if role in {"system", "developer", "user"} or item_kind == "agent_message":
                response()
                append(
                    {
                        "id": payload.get("id"),
                        "role": (
                            "user"
                            if item_kind == "agent_message"
                            else "system" if role == "developer" else role
                        ),
                        "content": payload.get("content")
                        or ([payload] if item_kind != "message" else []),
                        "response_metadata": {"response_item": payload},
                    }
                )
            elif item_kind in {"function_call_output", "custom_tool_call_output"}:
                response()
                output = payload.get("output", "")
                append(
                    {
                        "id": payload.get("id"),
                        "role": "tool",
                        "tool_call_id": payload.get("call_id"),
                        "content": (
                            output
                            if isinstance(output, (str, list))
                            else json.dumps(output)
                        ),
                        "response_metadata": {"response_item": payload},
                    }
                )
            else:
                pending.append(payload)
        elif kind == "token_usage_record":
            identity = payload.get("response_id")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Codex response usage requires response_id")
            if payload.get("thread_id") not in {None, thread}:
                raise ValueError("Codex response usage has a different thread identity")
            if identity in seen_responses:
                if seen_responses[identity] != payload:
                    raise ValueError("Conflicting Codex response usage observations")
                continue
            seen_responses[identity] = payload
            response(payload)
        elif kind == "compacted":
            response()
            usages.append(None)
            append(
                {
                    "role": "assistant",
                    "content": payload.get("message") or "",
                    "response_metadata": {"compacted": payload},
                },
                source="summarization",
            )
    response()
    if pending_turns:
        usages.append(None)
    return records, usages, source_usage


def _codex_rollouts(paths, agent):
    records = []
    usages = []
    sources = []
    truncated = False
    threads = set()
    for path in paths:
        events, partial = _read_rollout(path)
        projected, usage, raw = _codex_rollout_records(events, agent, path.stem)
        identities = {record["run_id"] for record in projected}
        if threads & identities:
            raise ValueError("Multiple Codex rollouts for the same thread")
        threads.update(identities)
        records.extend(projected)
        usages.extend(usage)
        sources.extend(raw)
        truncated |= partial
    if truncated:
        usages.append(None)
    seen = {}
    for record in records:
        identity = record["message"]["id"]
        if identity in seen and seen[identity] != record:
            raise ValueError("Conflicting Codex rollout message observations")
        seen[identity] = record
    records = list(seen.values())
    parents = {
        record["run_id"]: record["parent_ids"][-1]
        for record in records
        if record.get("parent_ids")
    }
    for record in records:
        ancestry = []
        identity = record["run_id"]
        while identity in parents:
            identity = parents[identity]
            if identity == record["run_id"] or identity in ancestry:
                raise ValueError("Cyclic Codex thread ancestry")
            ancestry.insert(0, identity)
        if ancestry:
            record["parent_ids"] = ancestry
    for seq, record in enumerate(records, 1):
        record["seq"] = seq
    return records, {
        "tokens": _canonical_totals(_aggregate_usage(usages)),
        "source_usage": sources,
        "usage_scope": "recorded-responses",
        "usage_complete": False,
        "model_responses_with_usage_count": sum(usage is not None for usage in usages),
        "notices": [
            "Raw rollouts record response completions, not all request starts; call coverage is unproven."
        ]
        + (
            ["An interrupted rollout tail was retained as incomplete."]
            if truncated
            else []
        ),
    }


def codex_messages(events, prompt):
    """Map Codex item events onto trajectory messages."""
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
                _canonical_usage(
                    raw.get("input_tokens"),
                    raw.get("cached_input_tokens"),
                    raw.get("cache_write_input_tokens"),
                    raw.get("output_tokens"),
                    raw.get("reasoning_output_tokens"),
                    raw.get("total_tokens"),
                )
            )
            source_usage.append(raw)
    if pending_turn:
        usages.append(None)
        metrics["notices"].append("A recorded turn has no completion usage.")
    metrics["tokens"] = _canonical_totals(_aggregate_usage(usages))
    metrics["source_usage"] = source_usage
    metrics["usage_scope"] = "recorded-turns"
    metrics["usage_complete"] = False
    if not any(usage is not None for usage in usages) and events:
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
    session_usage = _opencode_usage(raw) if raw else None
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
            messages.append({"role": "user", "id": meta.get("id"), "content": content})
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
                    }
                )
            else:
                blocks.append(dict(part))
        message = {
            "role": "assistant",
            "id": meta.get("id"),
            "content": blocks,
            "tool_calls": tool_calls,
            "invalid_tool_calls": [],
            "additional_kwargs": {},
        }
        raw_usage = meta.get("tokens") or {}
        if raw_usage:
            message["usage_metadata"] = _opencode_usage(raw_usage)
            message["response_metadata"] = {"token_usage": raw_usage}
        messages.append(message)
        messages.extend(pending_tools)
    if prompt and not any(message["role"] == "user" for message in messages):
        messages.insert(0, {"role": "user", "id": "prompt", "content": prompt})
    messages = _identify_messages(messages)
    metrics["usage_scope"] = (
        "session" if session_usage is not None else "recorded-messages"
    )
    metrics["tokens"] = _canonical_totals(
        session_usage
        if session_usage is not None
        else _aggregate_usage(
            [
                message.get("usage_metadata")
                for message in messages
                if message["role"] == "assistant"
            ]
        )
    )
    return messages, metrics
