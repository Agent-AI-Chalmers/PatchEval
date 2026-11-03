"""Project Codex rollouts and Codex/Trae stdout without inventing call coverage."""

from . import item_events
from .records import (
    aggregate_usage,
    finalize_records,
    normalize_usage,
    read_jsonl_capture,
    usage_totals,
)
from .responses import context_message, response_message


def read_trajectory(work_dir, prompt, agent):
    """Prefer per-response rollouts; stdout supplies separately scoped turn totals."""
    paths = (
        sorted((work_dir / "codex-sessions").rglob("rollout-*.jsonl"))
        if agent == "codex"
        else []
    )
    if paths:
        return _codex_rollouts(paths, agent)
    return item_events.read_trajectory(work_dir, prompt, agent)


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
        message = response_message(pending)
        extra = {}
        if any(
            item.get("type") in {"compaction", "context_compaction"} for item in pending
        ):
            extra["source"] = "summarization"
        if completion is not None:
            identity = completion["response_id"]
            raw = completion.get("usage") or {}
            usage = normalize_usage(
                input_tokens=raw.get("input_tokens"),
                cache_read=raw.get("cached_input_tokens"),
                cache_creation=raw.get("cache_write_input_tokens"),
                output=raw.get("output_tokens"),
                reasoning=raw.get("reasoning_output_tokens"),
                total=raw.get("total_tokens"),
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
            context = context_message(payload)
            if context is not None:
                response()
                append(context)
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
        events, partial = read_jsonl_capture(path, "Codex rollout")
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
    records = finalize_records(records)
    return records, {
        "tokens": usage_totals(aggregate_usage(usages)),
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
