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

    assert [message["type"] for message in messages] == ["human", "ai", "ai", "tool"]
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
        "cache_read_tokens": 80,
        "cache_creation_tokens": 0,
        "uncached_input_tokens": 20,
        "output_tokens": 10,
        "reasoning_tokens": 5,
        "total_tokens": 110,
    }
    usage = messages[2]["usage_metadata"]
    assert usage["input_tokens"] == 100
    assert usage["input_token_details"]["cache_read"] == 80
    assert usage["output_token_details"]["reasoning"] == 5
    assert usage["total_tokens"] == 110
    assert messages[2]["response_metadata"]["usage_scope"] == "run_aggregate"


def test_opencode_session_export_maps_to_trajectory_messages():
    module = load_module()
    messages, metrics = module.opencode_messages(OPENCODE_SESSION, "fix the bug")

    assert [message["type"] for message in messages] == ["human", "ai", "tool"]
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
    assert metrics["tokens"]["cache_read_tokens"] == 2
    assert metrics["tokens"]["uncached_input_tokens"] == 10
    assert metrics["tokens"]["total_tokens"] == 18
    assert metrics["cost"] == 0.01


def test_both_clis_share_one_token_vocabulary():
    module = load_module()
    _, codex = module.codex_messages(CODEX_EVENTS, "")
    _, opencode = module.opencode_messages(OPENCODE_SESSION, "")

    assert set(codex["tokens"]) == set(opencode["tokens"])
    for metrics in (codex, opencode):
        tokens = metrics["tokens"]
        assert tokens["uncached_input_tokens"] == (
            tokens["input_tokens"]
            - tokens["cache_read_tokens"]
            - tokens["cache_creation_tokens"]
        )


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
        if message["type"] != "ai":
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
    assert usage["output_token_details"]["reasoning"] == 101
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
    assert [message["type"] for message in messages] == ["human", "ai", "tool", "ai"]
    assert messages[2]["content"] == "a.txt"
    assert messages[3]["content"] == [{"type": "text", "text": "checking"}]
    assert metrics["tool_calls"] == 1
    assert metrics["tokens"]["input_tokens"] == 35
    assert metrics["tokens"]["output_tokens"] == 6
    assert metrics["cost"] == pytest.approx(0.3)
    assert metrics["notices"]
