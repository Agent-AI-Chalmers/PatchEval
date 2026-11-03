"""Project Trae's completion-first history journal without replaying snapshots."""

from . import item_events
from .records import (
    aggregate_usage,
    finalize_records,
    normalize_usage,
    read_jsonl_capture,
    usage_totals,
)
from .responses import context_message, response_message


def read_trajectory(work_dir, prompt):
    paths = sorted((work_dir / "trae-sessions").rglob("rollout-*.jsonl"))
    if not paths:
        return item_events.read_trajectory(
            work_dir,
            prompt,
            "traecli",
            cache_creation_field="cache_creation_input_tokens",
        )
    records = []
    usages = []
    sources = []
    mutations = []
    threads = set()
    partial = False
    for path in paths:
        events, truncated = read_jsonl_capture(path, "Trae rollout")
        projected, observed, raw, unprojected, owner = _rollout_records(
            events, path.stem
        )
        if owner in threads:
            raise ValueError("Multiple Trae rollouts for the same thread")
        threads.add(owner)
        records.extend(projected)
        usages.extend(observed)
        sources.extend(raw)
        mutations.extend(unprojected)
        partial |= truncated
    if partial:
        usages.append(None)
    records = finalize_records(records)
    notices = [
        "Trae rollouts record completions, not complete upstream request coverage."
    ]
    if partial:
        notices.append("An interrupted Trae rollout tail was retained as incomplete.")
    if mutations:
        notices.append(
            "Non-append history mutations are retained as source evidence; their usage is unknown."
        )
    return records, {
        "tokens": usage_totals(aggregate_usage(usages)),
        "source_usage": sources,
        "source_history_mutations": mutations,
        "usage_scope": "recorded-responses",
        "usage_complete": False,
        "model_responses_with_usage_count": sum(
            record["message"].get("usage_metadata") is not None for record in records
        ),
        "notices": notices,
    }


def _rollout_records(events, filename):
    """Match each completion to the following appended assistant items.

    Trae 0.213.1 persists token_usage_record before history_mutation appends its
    output. Flush that response before a new completion, input, tool result, or
    turn end. Display completions and token_count are repeated views, not new
    responses. Replacement snapshots must not be charged as new execution.
    """
    owners = [
        e.get("payload", {}).get("session_id") or e.get("payload", {}).get("id")
        for e in events
        if e.get("type") == "session_meta"
    ]
    owner = next(
        (identity for identity in owners if identity and filename.endswith(identity)),
        owners[-1] if owners else filename,
    )
    active_thread = owner
    turn = None
    parent = None
    session_source = None
    records = []
    items = []
    completion = None
    usages = []
    sources = []
    unprojected = []
    pending_turns = set()
    seen_responses = {}
    seen_commits = {}
    seen_items = {}

    def append(message, **extra):
        message = dict(message)
        message["id"] = (
            f"{owner}:{message.get('id') or 'item-' + str(len(records) + 1)}"
        )
        record = {
            "event": "message",
            "agent": "traecli",
            "run_id": owner,
            "message": message,
            "metadata": {"trae_source": session_source, "turn_id": turn},
            **extra,
        }
        if parent:
            record["parent_ids"] = [parent]
        records.append(record)

    def flush_response():
        nonlocal items, completion
        if not items and completion is None:
            return
        message = response_message(items)
        extra = {}
        if any(
            item.get("type") in {"compaction", "context_compaction"} for item in items
        ):
            extra["source"] = "summarization"
        if completion is None:
            usages.append(None)
        else:
            raw = completion.get("usage") or {}
            usage = normalize_usage(
                raw.get("input_tokens"),
                raw.get("cached_input_tokens"),
                raw.get("cache_creation_input_tokens"),
                raw.get("output_tokens"),
                raw.get("reasoning_output_tokens"),
                raw.get("total_tokens"),
            )
            message.update(id=completion["response_id"], usage_metadata=usage)
            message["response_metadata"]["token_usage_record"] = completion
            extra["call_id"] = f"{owner}:{completion['response_id']}"
            usages.append(usage)
            sources.append(completion)
        append(message, **extra)
        items = []
        completion = None

    for event in events:
        payload = event.get("payload") or {}
        kind = event.get("type")
        if kind == "session_meta":
            active_thread = payload.get("session_id") or payload.get("id") or owner
            if active_thread != owner:
                continue
            session_source = payload.get("source")
            if isinstance(session_source, dict):
                spawn = session_source.get("subagent") or {}
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
        elif kind == "turn_context":
            turn = payload.get("turn_id") or turn
        elif kind == "token_usage_record":
            identity = payload.get("response_id")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Trae response usage requires response_id")
            if payload.get("thread_id") not in {None, owner}:
                raise ValueError("Trae response usage has a different thread identity")
            if identity in seen_responses:
                if seen_responses[identity] != payload:
                    raise ValueError("Conflicting Trae response usage observations")
                continue
            flush_response()
            seen_responses[identity] = payload
            completion = payload
        elif kind == "history_mutation":
            if payload.get("version") != 1:
                raise ValueError("Unsupported Trae history mutation version")
            identity = payload.get("commit_id")
            if identity in seen_commits:
                if seen_commits[identity] != payload:
                    raise ValueError("Conflicting Trae history commit observations")
                continue
            if not isinstance(identity, str) or not identity:
                raise ValueError("Trae history mutation requires commit_id")
            seen_commits[identity] = payload
            if payload.get("operation") != "append":
                flush_response()
                unprojected.append(payload)
                usages.append(None)
                continue
            for item in payload.get("items") or []:
                identity = item.get("id")
                if identity and identity in seen_items:
                    if seen_items[identity] != item:
                        raise ValueError("Conflicting Trae history item observations")
                    continue
                if identity:
                    seen_items[identity] = item
                context = context_message(item)
                if context is None:
                    items.append(item)
                else:
                    flush_response()
                    append(context)
        elif kind == "event_msg":
            event_type = payload.get("type")
            if event_type == "task_started":
                turn = payload.get("turn_id")
                pending_turns.add(turn)
            elif event_type in {"task_complete", "task_completed", "turn_aborted"}:
                flush_response()
                pending_turns.discard(payload.get("turn_id") or turn)
                if event_type == "turn_aborted":
                    usages.append(None)
        elif kind == "compacted":
            flush_response()
            usages.append(None)
            append(
                {
                    "role": "assistant",
                    "content": payload.get("message") or "",
                    "response_metadata": {"compacted": payload},
                },
                source="summarization",
            )
    flush_response()
    if pending_turns:
        usages.append(None)
    return records, usages, sources, unprojected, owner
