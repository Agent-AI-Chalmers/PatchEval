"""OpenCode session ownership, interrupted capture, and usage regressions."""

import json

import pytest
from trajectory import build_trajectory, write_trajectory

from .fixtures.opencode import (
    OPENCODE_SESSION,
    export_opencode_session,
    opencode_database_session,
    write_opencode_database,
)


def test_opencode_session_export_maps_to_trajectory_messages(tmp_path):
    messages, metrics = export_opencode_session(
        tmp_path, OPENCODE_SESSION, "fix the bug"
    )

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


def test_opencode_reasoning_is_inclusive_in_message_and_run_totals(tmp_path):
    tokens = {
        "input": 2257,
        "output": 57,
        "reasoning": 101,
        "total": 10223,
        "cache": {"read": 7808, "write": 0},
    }
    messages, metrics = export_opencode_session(
        tmp_path,
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
    trajectory, _ = write_trajectory(
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
    records, metrics = build_trajectory(tmp_path, "opencode")
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
    assert metrics["tokens"]["input_tokens"] is None
    assert metrics["tokens"]["output_tokens"] is None
    assert [
        m["usage_metadata"]["total_tokens"] for m in messages if m.get("usage_metadata")
    ] == [18, 23]
    assert metrics["cost"] == pytest.approx(0.3)
    assert metrics["notices"]


@pytest.mark.parametrize("count", [None, 0])
def test_opencode_optional_buckets_preserve_unknown_and_zero(count, tmp_path):
    raw = {"input": 10, "output": 2}
    if count is not None:
        raw.update(reasoning=count, cache={"read": count, "write": count})
    messages, metrics = export_opencode_session(
        tmp_path,
        {"messages": [{"info": {"role": "assistant", "tokens": raw}, "parts": []}]},
        "",
    )
    usage = messages[0]["usage_metadata"]
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == 12
    assert usage["cache_read_input_tokens"] == count
    assert usage["cache_creation_input_tokens"] == count
    assert usage["reasoning_tokens"] == count
    assert metrics["tokens"]["uncached_input_tokens"] == (None if count is None else 10)


def test_opencode_total_mismatch_keeps_raw_evidence(tmp_path):
    raw = {"input": 10, "output": 2, "total": 999}
    with pytest.warns(RuntimeWarning, match="transcript_source_total_mismatch"):
        messages, _ = export_opencode_session(
            tmp_path, {"messages": [{"info": {"role": "assistant", "tokens": raw}}]}, ""
        )
    assert messages[0]["usage_metadata"]["total_tokens"] == 12
    assert messages[0]["response_metadata"]["token_usage"] == raw


def test_opencode_session_totals_are_not_added_to_message_usage(tmp_path):
    raw = {"input": 10, "output": 2}
    _, metrics = export_opencode_session(
        tmp_path,
        {
            "info": {"tokens": raw},
            "messages": [
                {"info": {"role": "assistant", "tokens": raw}},
                {"info": {"role": "assistant", "tokens": raw}},
            ],
        },
        "",
    )
    assert metrics["tokens"]["total_tokens"] == 24
    assert metrics["usage_scope"] == "recorded-responses"
    assert metrics["source_usage"] == [{"session_id": None, "usage": raw}]


def test_opencode_missing_response_usage_keeps_aggregate_unknown(tmp_path):
    _, metrics = export_opencode_session(
        tmp_path,
        {
            "messages": [
                {"info": {"role": "assistant", "tokens": {"input": 10, "output": 2}}},
                {"info": {"role": "assistant"}},
            ]
        },
        "",
    )
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["usage_scope"] == "recorded-responses"


@pytest.mark.parametrize("status", ["pending", "running", None])
def test_opencode_unfinished_tool_has_no_fabricated_result(status, tmp_path):
    messages, metrics = export_opencode_session(
        tmp_path,
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
    records, metrics = build_trajectory(tmp_path, "opencode")
    assert [record["message"]["role"] for record in records] == ["assistant", "tool"]
    assert records[-1]["message"]["content"] == "/src"
    assert metrics["tool_calls"] == 1


def test_repeated_opencode_messages_do_not_double_count_usage(tmp_path):
    entry = {
        "info": {"id": "m1", "role": "assistant", "tokens": {"input": 10, "output": 2}}
    }
    messages, metrics = export_opencode_session(
        tmp_path, {"messages": [entry, entry]}, ""
    )
    assert len(messages) == 1
    assert metrics["tokens"]["total_tokens"] == 12


def test_opencode_user_attachments_remain_in_original_content(tmp_path):
    parts = [
        {"type": "text", "text": "inspect"},
        {"type": "file", "mime": "image/png", "url": "data:image/png;base64,AA=="},
    ]
    messages, _ = export_opencode_session(
        tmp_path,
        {"messages": [{"info": {"id": "u1", "role": "user"}, "parts": parts}]},
        "",
    )
    assert messages[0]["content"] == parts


def test_opencode_database_keeps_child_sessions_and_reads_uncheckpointed_wal(tmp_path):
    """The root export omits child execution and may never run after a timeout."""
    sessions = [
        opencode_database_session("root"),
        opencode_database_session("child", "root"),
    ]
    db = write_opencode_database(tmp_path, sessions)
    try:
        (tmp_path / "opencode-session.json").write_text(json.dumps(OPENCODE_SESSION))
        records, metrics = build_trajectory(tmp_path, "opencode")
    finally:
        db.close()
    assert [r["run_id"] for r in records] == ["root", "child"]
    assert records[1]["parent_ids"] == ["root"]
    assert [r["call_id"] for r in records] == ["root-response", "child-response"]
    assert metrics["tokens"]["total_tokens"] == 50
    assert metrics["tokens"]["input_tokens"] == 40
    assert metrics["tokens"]["output_tokens"] == 10
    assert metrics["tokens"]["reasoning_tokens"] == 2
    assert metrics["usage_scope"] == "recorded-responses"
    assert metrics["model_responses_with_usage_count"] == 2
    assert metrics["usage_complete"] is False
    source = records[1]["message"]["response_metadata"]
    assert source["opencode_message"]["id"] == "child-response"
    assert source["opencode_parts"][0]["text"] == "answer from child"
    assert metrics["source_usage"][0]["usage"]["input"] == 999


@pytest.mark.parametrize("completed", [False, True])
def test_opencode_zero_defaults_are_unknown_until_response_completes(
    tmp_path, completed
):
    session = opencode_database_session("root")
    meta = session["messages"][0]["info"]
    meta["tokens"] = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache": {"read": 0, "write": 0},
    }
    if not completed:
        del meta["time"]["completed"]
    db = write_opencode_database(tmp_path, [session])
    db.close()
    records, metrics = build_trajectory(tmp_path, "opencode")
    message = records[0]["message"]
    assert metrics["tokens"]["total_tokens"] == (0 if completed else None)
    assert (message.get("usage_metadata") is not None) is completed
    assert message["response_metadata"]["token_usage"] == meta["tokens"]


def test_opencode_database_preserves_compaction_source(tmp_path):
    session = opencode_database_session("root")
    session["messages"][0]["info"]["summary"] = True
    db = write_opencode_database(tmp_path, [session])
    db.close()
    records, _ = build_trajectory(tmp_path, "opencode")
    assert records[0]["source"] == "summarization"


def test_opencode_database_rejects_cyclic_session_ancestry(tmp_path):
    db = write_opencode_database(
        tmp_path,
        [
            opencode_database_session("root", "child"),
            opencode_database_session("child", "root"),
        ],
    )
    db.close()
    with pytest.raises(ValueError, match="Cyclic"):
        build_trajectory(tmp_path, "opencode")


def test_opencode_export_does_not_recount_inherited_foreign_messages(tmp_path):
    session = opencode_database_session("child", "root")
    inherited = opencode_database_session("root")["messages"][0]
    inherited["info"]["sessionID"] = "root"
    session["messages"].insert(0, inherited)
    (tmp_path / "opencode-session.json").write_text(json.dumps(session))
    records, metrics = build_trajectory(tmp_path, "opencode")
    assert len(records) == 1
    assert records[0]["call_id"] == "child-response"
    assert metrics["tokens"]["total_tokens"] == 25


def test_opencode_event_capture_rejects_corrupt_complete_lines(tmp_path):
    (tmp_path / "agent_events.jsonl").write_text('{"type":broken}\n')
    with pytest.raises(ValueError, match="Malformed OpenCode"):
        build_trajectory(tmp_path, "opencode")


def test_opencode_step_usage_survives_interruption_before_message_update(tmp_path):
    """OpenCode persists step-finish before updating the assistant's default counters."""
    session = opencode_database_session("root")
    entry = session["messages"][0]
    measured = entry["info"].pop("tokens")
    entry["info"]["tokens"] = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache": {"read": 0, "write": 0},
    }
    del entry["info"]["time"]["completed"]
    entry["parts"].append({"type": "step-finish", "tokens": measured, "reason": "stop"})
    db = write_opencode_database(tmp_path, [session])
    db.close()
    records, metrics = build_trajectory(tmp_path, "opencode")
    assert records[0]["message"]["usage_metadata"]["total_tokens"] == 25
    assert metrics["tokens"]["total_tokens"] == 25
    assert (
        records[0]["message"]["response_metadata"]["opencode_message"]["tokens"][
            "input"
        ]
        == 0
    )
