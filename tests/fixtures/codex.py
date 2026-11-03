"""Independent codex samples and capture builders for PatchEval tests."""

import json

CODEX_EVENTS = [
    {"type": "thread.started", "thread_id": "thread-1"},
    {
        "type": "item.completed",
        "item": {"id": "item_0", "type": "agent_message", "text": "Looking around."},
    },
    {
        "type": "item.completed",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "ls -la",
            "aggregated_output": "a.txt",
            "exit_code": 0,
        },
    },
    {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 80,
            "output_tokens": 10,
            "reasoning_output_tokens": 5,
        },
    },
]


def write_codex_rollout(work, events, name="root"):
    path = work / "codex-sessions" / f"rollout-{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return path


def codex_rollout_sample():
    return [
        {
            "type": "session_meta",
            "payload": {
                "id": "thread-one",
                "source": "exec",
                "base_instructions": {"text": "repair safely"},
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": "turn-one"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "fix bug"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "id": "reason-one",
                "summary": [{"type": "summary_text", "text": "inspect"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": "fc-one",
                "call_id": "tool-one",
                "name": "shell",
                "arguments": '{"command":"pwd"}',
            },
        },
        {
            "type": "token_usage_record",
            "payload": {
                "thread_id": "thread-one",
                "turn_id": "turn-one",
                "response_id": "response-one",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 2,
                    "cache_write_input_tokens": 1,
                    "output_tokens": 3,
                    "reasoning_output_tokens": 1,
                    "total_tokens": 13,
                },
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "tool-one",
                "output": "/src",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "fixed"}],
            },
        },
        {
            "type": "token_usage_record",
            "payload": {
                "thread_id": "thread-one",
                "turn_id": "turn-one",
                "response_id": "response-two",
                "usage": {
                    "input_tokens": 20,
                    "cached_input_tokens": 2,
                    "cache_write_input_tokens": 1,
                    "output_tokens": 3,
                    "reasoning_output_tokens": 1,
                    "total_tokens": 23,
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": "turn-one"},
        },
    ]
