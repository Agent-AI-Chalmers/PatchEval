import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "patcheval/exp_agent/trajectory.py"
    spec = importlib.util.spec_from_file_location("trajectory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

OPENCODE_SESSION = {
    "info": {
        "id": "ses_1",
        "cost": 0.01,
        "tokens": {
            "input": 10,
            "output": 5,
            "reasoning": 1,
            "cache": {"read": 2, "write": 0},
        },
    },
    "messages": [
        {
            "info": {"role": "user", "id": "msg_1"},
            "parts": [{"type": "text", "text": "hi"}],
        },
        {
            "info": {
                "role": "assistant",
                "id": "msg_2",
                "tokens": {"input": 10, "output": 5},
            },
            "parts": [
                {"type": "reasoning", "text": "thinking"},
                {"type": "text", "text": "done"},
                {
                    "type": "tool",
                    "tool": "bash",
                    "callID": "call_1",
                    "state": {
                        "status": "completed",
                        "input": {"command": "ls"},
                        "output": "a.txt",
                        "metadata": {
                            "diff": "Index: a.txt\n--- a.txt\n+++ b.txt\n@@ -1 +1 @@\n-a\n+b\n",
                            "truncated": False,
                        },
                    },
                },
            ],
        },
    ],
}


def test_codex_events_map_to_trajectory_messages():
    module = load_module()
    messages, metrics = module.codex_messages(CODEX_EVENTS, "fix the bug")

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "assistant",
        "tool",
    ]
    assert messages[0]["content"] == "fix the bug"
    assert messages[1]["content"] == [{"type": "text", "text": "Looking around."}]
    call = messages[2]["content"][0]
    assert call["type"] == "tool_use"
    assert call["input"] == {"command": "ls -la"}
    assert messages[3]["content"] == "a.txt"
    assert messages[3]["tool_call_id"] == call["id"]
    assert messages[3]["additional_kwargs"]["exit_code"] == 0
    assert messages[3]["status"] == "success"
    assert messages[2]["tool_calls"] == [
        {
            "name": "shell",
            "args": {"command": "ls -la"},
            "id": call["id"],
            "type": "tool_call",
        }
    ]
    assert messages[2]["invalid_tool_calls"] == []
    assert metrics["tool_calls"] == 1
    # Canonical shape: inclusive input, cache and reasoning as subsets.
    assert metrics["tokens"] == {
        "input_tokens": 100,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": None,
        "uncached_input_tokens": None,
        "output_tokens": 10,
        "reasoning_tokens": 5,
        "total_tokens": 110,
    }
    # Run totals cannot be attributed to an individual response.
    assert all("usage_metadata" not in message for message in messages)
    assert all(
        "usage_scope" not in message.get("response_metadata", {})
        for message in messages
    )


def test_opencode_session_export_maps_to_trajectory_messages():
    module = load_module()
    messages, metrics = module.opencode_messages(OPENCODE_SESSION, "fix the bug")

    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    kinds = [block["type"] for block in messages[1]["content"]]
    assert kinds == ["thinking", "text", "tool_use"]
    assert messages[1]["content"][2]["input"] == {"command": "ls"}
    assert messages[2]["content"] == "a.txt"
    assert messages[2]["additional_kwargs"]["status"] == "completed"
    assert messages[2]["additional_kwargs"]["diff"].startswith("Index: a.txt")
    assert messages[2]["additional_kwargs"]["diff_truncated"] is False
    assert messages[2]["status"] == "success"
    assert messages[1]["tool_calls"] == [
        {"name": "bash", "args": {"command": "ls"}, "id": "call_1", "type": "tool_call"}
    ]
    assert metrics["tool_calls"] == 1
    # OpenCode's input is uncached; the canonical value adds cache back.
    assert metrics["tokens"]["input_tokens"] == 12
    assert metrics["tokens"]["cache_read_input_tokens"] == 2
    assert metrics["tokens"]["uncached_input_tokens"] == 10
    assert metrics["tokens"]["total_tokens"] == 18
    assert metrics["cost"] == 0.01


def test_both_clis_share_one_token_vocabulary():
    module = load_module()
    _, codex = module.codex_messages(CODEX_EVENTS, "")
    _, opencode = module.opencode_messages(OPENCODE_SESSION, "")

    assert set(codex["tokens"]) == set(opencode["tokens"])
    assert codex["tokens"]["uncached_input_tokens"] is None
    assert opencode["tokens"]["uncached_input_tokens"] == 10


def test_model_provenance_is_recorded_on_ai_messages(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "prompt.txt").write_text("fix the bug")
    (work / "agent_stdout.txt").write_text(
        "\n".join(json.dumps(event) for event in CODEX_EVENTS)
    )

    module = load_module()
    records, metrics = module.build_trajectory(
        work, "codex", "deepseek", "deepseek-v4-pro"
    )

    for record in records:
        message = record["message"]
        if message["role"] != "assistant":
            assert "response_metadata" not in message
            continue
        assert message["response_metadata"]["model_provider"] == "deepseek"
        assert message["response_metadata"]["model_name"] == "deepseek-v4-pro"
    assert metrics["model_provider"] == "deepseek"
    assert metrics["model_name"] == "deepseek-v4-pro"


def test_write_trajectory_records_sequence_and_metrics(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "prompt.txt").write_text("fix the bug")
    (work / "agent_stdout.txt").write_text(
        "\n".join(json.dumps(event) for event in CODEX_EVENTS)
    )

    module = load_module()
    trajectory, metrics = module.write_trajectory(
        work, tmp_path / "trajectories", "CVE-test"
    )

    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    assert [record["seq"] for record in records] == [1, 2, 3, 4]
    assert all(record["event"] == "message" for record in records)
    payload = json.loads(metrics.read_text())
    assert payload["agent"] == "codex"
    assert payload["tool_calls"] == 1
    assert payload["messages"] == 4
    assert payload["tokens"]["total_tokens"] == 110
    assert all("usage_metadata" not in record["message"] for record in records)


def test_write_trajectory_prefers_opencode_export(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "prompt.txt").write_text("fix the bug")
    (work / "opencode-session.json").write_text(json.dumps(OPENCODE_SESSION))

    (work / "agent_events.jsonl").write_text(
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "messageID": "ignored",
                    "text": "partial stream",
                },
            }
        )
    )
    module = load_module()
    trajectory, metrics = module.write_trajectory(
        work, tmp_path / "trajectories", "CVE-test"
    )

    payload = json.loads(metrics.read_text())
    assert payload["agent"] == "opencode"
    assert payload["cost"] == 0.01
    assert payload["tool_calls"] == 1
    assert payload["notices"] == []
    assert "partial stream" not in trajectory.read_text()


def test_opencode_reasoning_is_inclusive_in_message_and_run_totals():
    module = load_module()
    tokens = {
        "input": 2257,
        "output": 57,
        "reasoning": 101,
        "total": 10223,
        "cache": {"read": 7808, "write": 0},
    }
    messages, metrics = module.opencode_messages(
        {
            "info": {"tokens": tokens},
            "messages": [
                {"info": {"role": "assistant", "tokens": tokens}, "parts": []}
            ],
        },
        "",
    )
    usage = messages[0]["usage_metadata"]
    assert usage["input_tokens"] == 10065
    assert usage["output_tokens"] == 158
    assert usage["reasoning_tokens"] == 101
    assert usage["total_tokens"] == 10223
    assert metrics["tokens"]["output_tokens"] == 158
    assert metrics["tokens"]["total_tokens"] == usage["total_tokens"]


@pytest.mark.parametrize("agent", ["traecli", "opencode"])
def test_explicit_agent_identity_survives_missing_session_export(tmp_path, agent):
    (tmp_path / "agent_stdout.txt").write_text(
        "\n".join(json.dumps(event) for event in CODEX_EVENTS)
    )
    records, metrics = load_module().build_trajectory(tmp_path, agent_kind=agent)
    assert metrics["agent"] == agent
    if agent == "opencode":
        assert records == []
        assert metrics["notices"]
    else:
        assert metrics["tool_calls"] == 1


@pytest.mark.parametrize(
    "status,expected", [("completed", "success"), ("failed", "error")]
)
def test_codex_file_changes_survive_trajectory_export(tmp_path, status, expected):
    """Native apply_patch emits file_change rather than command_execution."""
    changes = [{"path": "src/main.js", "kind": "update"}]
    event = {
        "type": "item.completed",
        "item": {
            "id": "edit-1",
            "type": "file_change",
            "changes": changes,
            "status": status,
        },
    }
    (tmp_path / "agent_stdout.txt").write_text(json.dumps(event) + "\n")
    trajectory, metrics = load_module().write_trajectory(
        tmp_path, tmp_path / "out", "CVE-test"
    )
    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    call, result = [record["message"] for record in records]
    assert call["tool_calls"] == [
        {
            "name": "apply_patch",
            "args": {"changes": changes},
            "id": "edit-1",
            "type": "tool_call",
        }
    ]
    assert result["tool_call_id"] == "edit-1"
    assert json.loads(result["content"]) == changes
    assert result["status"] == expected
    assert json.loads(metrics.read_text())["tool_calls"] == 1


def test_opencode_tool_error_survives_trajectory_export(tmp_path):
    """OpenCode stores failed tool diagnostics in error rather than output."""
    session = {
        "messages": [
            {
                "info": {"role": "assistant", "id": "message-1"},
                "parts": [
                    {
                        "type": "tool",
                        "callID": "call-1",
                        "tool": "glob",
                        "state": {
                            "status": "error",
                            "input": {"pattern": "**/*.py"},
                            "error": "ripgrep execution failed",
                        },
                    }
                ],
            }
        ]
    }
    (tmp_path / "opencode-session.json").write_text(json.dumps(session))
    trajectory, _ = load_module().write_trajectory(
        tmp_path, tmp_path / "out", "CVE-test", agent_kind="opencode"
    )
    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    result = records[-1]["message"]
    assert result["tool_call_id"] == "call-1"
    assert result["status"] == "error"
    assert result["content"] == "ripgrep execution failed"


@pytest.mark.parametrize("export", [None, "", '{"info":'])
def test_opencode_interrupted_run_recovers_persisted_events(tmp_path, export):
    """Timeouts can remove the container before session export completes."""
    (tmp_path / "prompt.txt").write_text("fix the bug")
    if export is not None:
        (tmp_path / "opencode-session.json").write_text(export)
    events = [
        {
            "type": "tool_use",
            "part": {
                "type": "tool",
                "messageID": "msg-1",
                "tool": "bash",
                "callID": "call-1",
                "state": {
                    "status": "completed",
                    "input": {"command": "ls"},
                    "output": "a.txt",
                },
            },
        },
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "messageID": "msg-1",
                "cost": 0.1,
                "tokens": {
                    "input": 10,
                    "output": 2,
                    "reasoning": 1,
                    "cache": {"read": 5, "write": 0},
                },
            },
        },
        {
            "type": "text",
            "part": {
                "type": "text",
                "messageID": "msg-2",
                "text": "checking",
            },
        },
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "messageID": "msg-2",
                "cost": 0.2,
                "tokens": {"input": 20, "output": 3, "reasoning": 0},
            },
        },
    ]
    (tmp_path / "agent_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + '\n{"type":'
    )
    records, metrics = load_module().build_trajectory(tmp_path, "opencode")
    messages = [record["message"] for record in records]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2]["content"] == "a.txt"
    assert messages[3]["content"] == [{"type": "text", "text": "checking"}]
    assert metrics["tool_calls"] == 1
    assert metrics["tokens"]["input_tokens"] == 35
    assert metrics["tokens"]["output_tokens"] == 6
    assert metrics["cost"] == pytest.approx(0.3)
    assert metrics["notices"]


def test_export_uses_contract_roles_and_unique_message_ids(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "prompt.txt").write_text("fix")
    (work / "agent_stdout.txt").write_text(
        "\n".join(json.dumps(event) for event in CODEX_EVENTS)
    )
    records, _ = load_module().build_trajectory(work, "codex")
    messages = [record["message"] for record in records]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "assistant",
        "tool",
    ]
    ids = [message["id"] for message in messages]
    assert all(isinstance(value, str) and value for value in ids)
    assert len(set(ids)) == len(ids)
    assert all("type" not in message for message in messages)


def test_codex_missing_measurements_remain_unknown():
    _, metrics = load_module().codex_messages(
        [{"type": "turn.completed", "usage": {"input_tokens": 10}}], ""
    )
    assert metrics["tokens"]["input_tokens"] == 10
    assert metrics["tokens"]["output_tokens"] is None
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["tokens"]["cache_creation_input_tokens"] is None
    assert metrics["tokens"]["reasoning_tokens"] is None


def test_codex_counts_all_recorded_turns_without_assigning_response_usage():
    messages, metrics = load_module().codex_messages(CODEX_EVENTS * 2, "")
    assert metrics["tokens"]["total_tokens"] == 220
    assert all("usage_metadata" not in message for message in messages)
    assert metrics["usage_scope"] == "recorded-turns"
    assert metrics["usage_complete"] is False


@pytest.mark.parametrize("agent", ["codex", "opencode"])
def test_export_records_obey_message_and_usage_contract(tmp_path, agent):
    """Use this adapter's own CLI samples, without the main package or its corpus."""
    if agent == "codex":
        (tmp_path / "agent_stdout.txt").write_text(
            "\n".join(json.dumps(event) for event in CODEX_EVENTS)
        )
    else:
        (tmp_path / "opencode-session.json").write_text(json.dumps(OPENCODE_SESSION))
    records, metrics = load_module().build_trajectory(tmp_path, agent)
    identities = []
    fields = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
    }
    for seq, record in enumerate(records, 1):
        assert record["seq"] == seq
        assert record["event"] == "message"
        assert record["agent"] == agent
        message = record["message"]
        assert message["role"] in {"system", "user", "assistant", "tool"}
        assert "type" not in message
        assert isinstance(message["content"], (str, list))
        assert isinstance(message["id"], str) and message["id"]
        identities.append(message["id"])
        usage = message.get("usage_metadata")
        if usage is not None:
            assert set(usage) == fields
            assert all(
                value is None or (type(value) is int and 0 <= value <= 9007199254740991)
                for value in usage.values()
            )
    assert len(identities) == len(set(identities))
    assert metrics["usage_complete"] is False


@pytest.mark.parametrize("count", [None, 0])
def test_opencode_optional_buckets_preserve_unknown_and_zero(count):
    raw = {"input": 10, "output": 2}
    if count is not None:
        raw.update(reasoning=count, cache={"read": count, "write": count})
    messages, metrics = load_module().opencode_messages(
        {"messages": [{"info": {"role": "assistant", "tokens": raw}, "parts": []}]}, ""
    )
    usage = messages[0]["usage_metadata"]
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == 12
    assert usage["cache_read_input_tokens"] == count
    assert usage["cache_creation_input_tokens"] == count
    assert usage["reasoning_tokens"] == count
    assert metrics["tokens"]["uncached_input_tokens"] == (None if count is None else 10)


def test_opencode_total_mismatch_keeps_raw_evidence():
    raw = {"input": 10, "output": 2, "total": 999}
    with pytest.warns(RuntimeWarning, match="transcript_source_total_mismatch"):
        messages, _ = load_module().opencode_messages(
            {"messages": [{"info": {"role": "assistant", "tokens": raw}}]}, ""
        )
    assert messages[0]["usage_metadata"]["total_tokens"] == 12
    assert messages[0]["response_metadata"]["token_usage"] == raw


def test_opencode_session_totals_are_not_added_to_message_usage():
    raw = {"input": 10, "output": 2}
    _, metrics = load_module().opencode_messages(
        {
            "info": {"tokens": raw},
            "messages": [
                {"info": {"role": "assistant", "tokens": raw}},
                {"info": {"role": "assistant", "tokens": raw}},
            ],
        },
        "",
    )
    assert metrics["tokens"]["total_tokens"] == 12
    assert metrics["usage_scope"] == "session"
    assert metrics["source_usage"] == raw


def test_opencode_missing_response_usage_keeps_aggregate_unknown():
    _, metrics = load_module().opencode_messages(
        {
            "messages": [
                {"info": {"role": "assistant", "tokens": {"input": 10, "output": 2}}},
                {"info": {"role": "assistant"}},
            ]
        },
        "",
    )
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["usage_scope"] == "recorded-messages"


@pytest.mark.parametrize("status", ["pending", "running", None])
def test_opencode_unfinished_tool_has_no_fabricated_result(status):
    messages, metrics = load_module().opencode_messages(
        {
            "messages": [
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {
                            "type": "tool",
                            "callID": "pending",
                            "tool": "bash",
                            "state": {"status": status, "input": {"command": "pwd"}},
                        }
                    ],
                }
            ]
        },
        "",
    )
    assert len(messages) == 1
    assert messages[0]["tool_calls"][0]["id"] == "pending"
    assert metrics["tool_calls"] == 1


@pytest.mark.parametrize("value", [True, -1, 1.5, "10", 9007199254740992])
def test_invalid_cli_token_counts_are_rejected(value):
    with pytest.raises(ValueError):
        load_module().codex_messages(
            [{"type": "turn.completed", "usage": {"input_tokens": value}}], ""
        )


def test_source_total_alone_cannot_establish_canonical_total():
    _, metrics = load_module().codex_messages(
        [{"type": "turn.completed", "usage": {"total_tokens": 100}}], ""
    )
    assert all(value is None for value in metrics["tokens"].values())
    assert metrics["source_usage"] == [{"total_tokens": 100}]


def test_codex_missing_turn_measurement_propagates_per_field():
    _, metrics = load_module().codex_messages(
        [
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
            {"type": "turn.completed", "usage": {"input_tokens": 5}},
        ],
        "",
    )
    assert metrics["tokens"]["input_tokens"] == 15
    assert metrics["tokens"]["output_tokens"] is None
    assert metrics["tokens"]["total_tokens"] is None


def test_recovered_opencode_updates_do_not_repeat_tool_calls(tmp_path):
    parts = [
        {
            "id": "tool",
            "messageID": "m1",
            "type": "tool",
            "callID": "c1",
            "state": {"status": "running", "input": {"command": "pwd"}},
        },
        {
            "id": "tool",
            "messageID": "m1",
            "type": "tool",
            "callID": "c1",
            "state": {
                "status": "completed",
                "input": {"command": "pwd"},
                "output": "/src",
            },
        },
    ]
    (tmp_path / "agent_events.jsonl").write_text(
        "\n".join(json.dumps({"part": part}) for part in parts)
    )
    records, metrics = load_module().build_trajectory(tmp_path, "opencode")
    assert [record["message"]["role"] for record in records] == ["assistant", "tool"]
    assert records[-1]["message"]["content"] == "/src"
    assert metrics["tool_calls"] == 1


def test_repeated_opencode_messages_do_not_double_count_usage():
    entry = {
        "info": {"id": "m1", "role": "assistant", "tokens": {"input": 10, "output": 2}}
    }
    messages, metrics = load_module().opencode_messages(
        {"messages": [entry, entry]}, ""
    )
    assert len(messages) == 1
    assert metrics["tokens"]["total_tokens"] == 12


def test_codex_reasoning_is_retained_as_assistant_content():
    messages, _ = load_module().codex_messages(
        [
            {
                "type": "item.completed",
                "item": {"id": "r1", "type": "reasoning", "text": "inspect caller"},
            }
        ],
        "",
    )
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == [{"type": "reasoning", "text": "inspect caller"}]


def test_opencode_user_attachments_remain_in_original_content():
    parts = [
        {"type": "text", "text": "inspect"},
        {"type": "file", "mime": "image/png", "url": "data:image/png;base64,AA=="},
    ]
    messages, _ = load_module().opencode_messages(
        {"messages": [{"info": {"id": "u1", "role": "user"}, "parts": parts}]}, ""
    )
    assert messages[0]["content"] == parts


@pytest.mark.parametrize(
    "field,value",
    [
        ("cached_input_tokens", 11),
        ("reasoning_output_tokens", 3),
    ],
)
def test_codex_impossible_subsets_are_rejected(field, value):
    raw = {"input_tokens": 10, "output_tokens": 2, field: value}
    with pytest.raises(ValueError, match="subset"):
        load_module().codex_messages([{"type": "turn.completed", "usage": raw}], "")


def test_conflicting_message_identity_fails_export(tmp_path):
    (tmp_path / "agent_stdout.txt").write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {
                    "type": "item.completed",
                    "item": {"id": "a", "type": "agent_message", "text": "one"},
                },
                {
                    "type": "item.completed",
                    "item": {"id": "a", "type": "agent_message", "text": "two"},
                },
            ]
        )
    )
    with pytest.raises(ValueError, match="Conflicting observations"):
        load_module().build_trajectory(tmp_path, "codex")


@pytest.mark.parametrize(
    "ending",
    [
        [{"type": "turn.started"}],
        [
            {"type": "turn.started"},
            {"type": "turn.failed", "error": {"message": "interrupted"}},
        ],
    ],
)
def test_codex_unfinished_turn_prevents_complete_aggregate(ending):
    _, metrics = load_module().codex_messages(CODEX_EVENTS + ending, "")
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["notices"]
    assert metrics["usage_complete"] is False


def test_codex_reports_cache_writes_using_native_field():
    _, metrics = load_module().codex_messages(
        [
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 2,
                    "cache_write_input_tokens": 1,
                    "output_tokens": 3,
                },
            }
        ],
        "",
    )
    assert metrics["tokens"]["cache_creation_input_tokens"] == 1
    assert metrics["tokens"]["uncached_input_tokens"] == 7


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


def test_codex_rollout_preserves_per_response_usage_and_content(tmp_path):
    write_codex_rollout(tmp_path, codex_rollout_sample())
    # Stdout is a second view of the same calls; it must not be added again.
    (tmp_path / "agent_stdout.txt").write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 30, "output_tokens": 6},
            }
        )
    )
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    responses = [record for record in records if record.get("call_id")]
    assert [
        record["message"]["usage_metadata"]["total_tokens"] for record in responses
    ] == [13, 23]
    assert responses[0]["message"]["tool_calls"][0]["id"] == "tool-one"
    assert responses[0]["message"]["content"][0]["type"] == "reasoning"
    assert responses[1]["message"]["content"] == [
        {"type": "output_text", "text": "fixed"}
    ]
    assert metrics["tokens"]["total_tokens"] == 36
    assert metrics["usage_scope"] == "recorded-responses"
    assert any(record["message"]["role"] == "system" for record in records)
    assert any(
        record["message"].get("tool_call_id") == "tool-one" for record in records
    )


def test_codex_rollout_token_snapshots_are_not_counted_twice(tmp_path):
    events = codex_rollout_sample()
    events.insert(
        6,
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"input_tokens": 10, "output_tokens": 3},
                    "total_token_usage": {"input_tokens": 10, "output_tokens": 3},
                },
            },
        },
    )
    events.insert(7, events[5])
    write_codex_rollout(tmp_path, events)
    _, metrics = load_module().build_trajectory(tmp_path, "codex")
    assert metrics["tokens"]["total_tokens"] == 36
    assert metrics["model_responses_with_usage_count"] == 2


def test_codex_rollout_incomplete_response_is_retained_without_guessing(tmp_path):
    events = codex_rollout_sample()[:-2]
    write_codex_rollout(tmp_path, events)
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    assert records[-1]["message"]["content"] == [
        {"type": "output_text", "text": "fixed"}
    ]
    assert "usage_metadata" not in records[-1]["message"]
    assert metrics["tokens"]["total_tokens"] is None


def test_codex_rollout_truncated_tail_is_partial_but_interior_corruption_fails(
    tmp_path,
):
    path = write_codex_rollout(tmp_path, codex_rollout_sample())
    path.write_text(path.read_text() + '{"type":')
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    assert records
    assert metrics["tokens"]["total_tokens"] is None
    path.write_text(path.read_text() + '\n{"type":"event_msg","payload":{}}\n')
    with pytest.raises(ValueError, match="Malformed Codex rollout"):
        load_module().build_trajectory(tmp_path, "codex")


def test_codex_rollouts_keep_branch_ids_and_ancestry(tmp_path):
    root = codex_rollout_sample()
    child = json.loads(json.dumps(root))
    child[0]["payload"].update(
        id="thread-child",
        source={"subagent": {"thread_spawn": {"parent_thread_id": "thread-one"}}},
    )
    for event in child:
        if event["type"] == "token_usage_record":
            event["payload"]["thread_id"] = "thread-child"
    write_codex_rollout(tmp_path, root)
    write_codex_rollout(tmp_path, child, "child")
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    ids = [record["message"]["id"] for record in records]
    assert len(ids) == len(set(ids))
    assert all(
        record["parent_ids"] == ["thread-one"]
        for record in records
        if record["run_id"] == "thread-child"
    )
    assert metrics["tokens"]["total_tokens"] == 72


def test_codex_compaction_preserves_summary_and_unknown_usage(tmp_path):
    events = codex_rollout_sample()
    events.insert(
        7,
        {
            "type": "compacted",
            "payload": {"message": "context summary", "replacement_history": []},
        },
    )
    write_codex_rollout(tmp_path, events)
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    summary = next(
        record for record in records if record.get("source") == "summarization"
    )
    assert summary["message"]["content"] == "context summary"
    assert metrics["tokens"]["total_tokens"] is None


def test_codex_rollout_usage_does_not_cross_tool_result_boundary(tmp_path):
    events = codex_rollout_sample()
    events[5], events[6] = events[6], events[5]
    write_codex_rollout(tmp_path, events)
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    request = next(record for record in records if record["message"].get("tool_calls"))
    assert "usage_metadata" not in request["message"]
    response = next(
        record
        for record in records
        if record.get("call_id") == "thread-one:response-one"
    )
    assert response["message"]["content"] == []
    assert metrics["tokens"]["total_tokens"] is None


def test_codex_forked_rollout_does_not_recount_inherited_parent_execution(tmp_path):
    root = codex_rollout_sample()
    child = json.loads(json.dumps(root))
    child[0]["payload"].update(
        id="child",
        source={"subagent": {"thread_spawn": {"parent_thread_id": "thread-one"}}},
    )
    for event in child:
        if event["type"] == "token_usage_record":
            event["payload"]["thread_id"] = "child"
    write_codex_rollout(tmp_path, root)
    write_codex_rollout(tmp_path, root + child, "fork")
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    assert metrics["tokens"]["total_tokens"] == 72
    assert metrics["model_responses_with_usage_count"] == 4
    assert {record["run_id"] for record in records} == {"thread-one", "child"}


def test_codex_tool_definitions_are_context_not_an_unmeasured_response(tmp_path):
    events = codex_rollout_sample()
    events.insert(
        2,
        {
            "type": "response_item",
            "payload": {"type": "additional_tools", "role": "developer", "tools": []},
        },
    )
    write_codex_rollout(tmp_path, events)
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    assert metrics["tokens"]["total_tokens"] == 36
    tools = next(
        record
        for record in records
        if record["message"]["content"]
        == [{"type": "additional_tools", "role": "developer", "tools": []}]
    )
    assert tools["message"]["role"] == "system"


def test_codex_child_meta_precedes_copied_history_with_explicit_thread_switch(tmp_path):
    parent = codex_rollout_sample()
    child = json.loads(json.dumps(parent))
    child[0]["payload"].update(
        id="child",
        source={"subagent": {"thread_spawn": {"parent_thread_id": "thread-one"}}},
    )
    for event in child:
        if event["type"] == "token_usage_record":
            event["payload"]["thread_id"] = "child"
    write_codex_rollout(tmp_path, parent)
    events = (
        [child[0]]
        + parent
        + [
            {
                "type": "event_msg",
                "payload": {"type": "thread_settings_applied", "thread_id": "child"},
            }
        ]
        + child[1:]
    )
    write_codex_rollout(tmp_path, events, "child")
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    assert metrics["tokens"]["total_tokens"] == 72
    assert metrics["model_responses_with_usage_count"] == 4
    child_records = [record for record in records if record["run_id"] == "child"]
    assert child_records
    assert all(record["parent_ids"] == ["thread-one"] for record in child_records)


def test_codex_inter_agent_input_does_not_become_assistant_output(tmp_path):
    """Incoming agent messages are context, not output charged to the next response."""
    events = codex_rollout_sample()
    incoming = {
        "type": "agent_message",
        "id": "delivery-one",
        "author": "/root/helper",
        "recipient": "/root",
        "content": [{"type": "input_text", "text": "found the bug"}],
    }
    events.insert(3, {"type": "response_item", "payload": incoming})
    write_codex_rollout(tmp_path, events)
    records, metrics = load_module().build_trajectory(tmp_path, "codex")
    delivery = next(
        r["message"] for r in records if r["message"]["id"].endswith(":delivery-one")
    )
    assert delivery["role"] == "user"
    assert delivery["content"] == incoming["content"]
    assert delivery["response_metadata"]["response_item"] == incoming
    assert "usage_metadata" not in delivery
    assert metrics["tokens"]["total_tokens"] == 36
    assert all(
        incoming
        not in r["message"].get("response_metadata", {}).get("response_items", [])
        for r in records
    )
