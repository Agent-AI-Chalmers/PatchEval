"""Shared record identity, capture parsing, and canonical token accounting rules."""

import json
import warnings

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
)
MAX_TOKEN_COUNT = 9007199254740991


def validate_token_count(value):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_TOKEN_COUNT:
        raise ValueError("Token counts must be nonnegative safe integers or null")
    return value


def normalize_usage(
    *, input_tokens, cache_read, cache_creation, output, reasoning, total
):
    """Normalize inclusive totals and subsets without inventing measurements.

    The caller owns source-specific addition of exclusive buckets. A reported
    total is evidence only: canonical total is derived from known components.
    """
    input_tokens, cache_read, cache_creation, output, reasoning = (
        validate_token_count(value)
        for value in (input_tokens, cache_read, cache_creation, output, reasoning)
    )
    derived = (
        validate_token_count(input_tokens + output)
        if input_tokens is not None and output is not None
        else None
    )
    reported = validate_token_count(total)
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


def aggregate_usage(usages):
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


def usage_totals(usage):
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


def identify_messages(messages):
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


def read_jsonl_capture(path, source):
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
            raise ValueError(f"Malformed {source}: {path.name}:{index + 1}")
        if not isinstance(event, dict):
            raise TypeError(f"{source} records must be objects")
        events.append(event)
    return events, False


def finalize_records(records):
    """Reject conflicting message identities and complete observed branch ancestry."""
    seen = {}
    for record in records:
        identity = record["message"]["id"]
        if identity in seen and seen[identity] != record:
            raise ValueError("Conflicting rollout message observations")
        seen[identity] = record
    records = list(seen.values())
    parents = {
        record["run_id"]: record["parent_ids"][-1]
        for record in records
        if record.get("parent_ids")
    }
    for seq, record in enumerate(records, 1):
        ancestry = []
        identity = record["run_id"]
        while identity in parents:
            identity = parents[identity]
            if identity == record["run_id"] or identity in ancestry:
                raise ValueError("Cyclic rollout thread ancestry")
            ancestry.insert(0, identity)
        if ancestry:
            record["parent_ids"] = ancestry
        record["seq"] = seq
    return records
