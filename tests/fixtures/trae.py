"""Independent samples of Trae 0.213.1's completion-first history journal."""

import json


def write_trae_rollout(work, events, name="trae-thread"):
    path = work / "trae-sessions" / f"rollout-{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return path


def trae_rollout_sample():
    def mutation(commit, items):
        return {
            "type": "history_mutation",
            "payload": {
                "version": 1,
                "commit_id": commit,
                "turn_id": "turn-one",
                "operation": "append",
                "items": items,
            },
        }

    def completion(identity, input_tokens):
        return {
            "type": "token_usage_record",
            "payload": {
                "thread_id": "trae-thread",
                "session_id": "trae-thread",
                "turn_id": "turn-one",
                "response_id": identity,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 3,
                    "total_tokens": input_tokens + 3,
                    "cached_input_tokens": 2,
                    "cache_creation_input_tokens": 1,
                    "reasoning_output_tokens": 1,
                },
            },
        }

    return [
        {
            "type": "session_meta",
            "payload": {
                "id": "trae-thread",
                "session_id": "trae-thread",
                "source": "exec",
                "history_format": "canonical_v1",
                "base_instructions": {"text": "repair safely"},
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": "turn-one"},
        },
        mutation(
            "user-commit",
            [
                {
                    "type": "message",
                    "id": "user-one",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "fix bug"}],
                }
            ],
        ),
        completion("response-one", 10),
        mutation(
            "request-commit",
            [
                {
                    "type": "function_call",
                    "id": "function-one",
                    "call_id": "tool-one",
                    "name": "probe_nonexistent_tool",
                    "arguments": "{}",
                }
            ],
        ),
        mutation(
            "result-commit",
            [
                {
                    "type": "function_call_output",
                    "id": "result-one",
                    "call_id": "tool-one",
                    "output": "unsupported call",
                }
            ],
        ),
        completion("response-two", 20),
        mutation(
            "answer-commit",
            [
                {
                    "type": "message",
                    "id": "message-two",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "fixed"}],
                }
            ],
        ),
        {
            "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": "turn-one"},
        },
    ]
