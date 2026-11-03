"""Codex response capture, inherited history, and turn fallback regressions."""

import json

import pytest
from trajectory import build_trajectory, write_trajectory
from trajectory.item_events import messages_from_events

from .fixtures.codex import CODEX_EVENTS, codex_rollout_sample, write_codex_rollout


def test_codex_events_map_to_trajectory_messages():
    messages, metrics = messages_from_events(CODEX_EVENTS, "fix the bug")

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
    trajectory, metrics = write_trajectory(tmp_path, tmp_path / "out", "CVE-test")
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


def test_codex_missing_measurements_remain_unknown():
    _, metrics = messages_from_events(
        [{"type": "turn.completed", "usage": {"input_tokens": 10}}], ""
    )
    assert metrics["tokens"]["input_tokens"] == 10
    assert metrics["tokens"]["output_tokens"] is None
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["tokens"]["cache_creation_input_tokens"] is None
    assert metrics["tokens"]["reasoning_tokens"] is None


def test_codex_counts_all_recorded_turns_without_assigning_response_usage():
    messages, metrics = messages_from_events(CODEX_EVENTS * 2, "")
    assert metrics["tokens"]["total_tokens"] == 220
    assert all("usage_metadata" not in message for message in messages)
    assert metrics["usage_scope"] == "recorded-turns"
    assert metrics["usage_complete"] is False


def test_codex_missing_turn_measurement_propagates_per_field():
    _, metrics = messages_from_events(
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


def test_codex_reasoning_is_retained_as_assistant_content():
    messages, _ = messages_from_events(
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
        messages_from_events([{"type": "turn.completed", "usage": raw}], "")


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
    _, metrics = messages_from_events(CODEX_EVENTS + ending, "")
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["notices"]
    assert metrics["usage_complete"] is False


def test_codex_reports_cache_writes_using_native_field():
    _, metrics = messages_from_events(
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
    records, metrics = build_trajectory(tmp_path, "codex")
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
    _, metrics = build_trajectory(tmp_path, "codex")
    assert metrics["tokens"]["total_tokens"] == 36
    assert metrics["model_responses_with_usage_count"] == 2


def test_codex_rollout_incomplete_response_is_retained_without_guessing(tmp_path):
    events = codex_rollout_sample()[:-2]
    write_codex_rollout(tmp_path, events)
    records, metrics = build_trajectory(tmp_path, "codex")
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
    records, metrics = build_trajectory(tmp_path, "codex")
    assert records
    assert metrics["tokens"]["total_tokens"] is None
    path.write_text(path.read_text() + '\n{"type":"event_msg","payload":{}}\n')
    with pytest.raises(ValueError, match="Malformed Codex rollout"):
        build_trajectory(tmp_path, "codex")


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
    records, metrics = build_trajectory(tmp_path, "codex")
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
    records, metrics = build_trajectory(tmp_path, "codex")
    summary = next(
        record for record in records if record.get("source") == "summarization"
    )
    assert summary["message"]["content"] == "context summary"
    assert metrics["tokens"]["total_tokens"] is None


def test_codex_rollout_usage_does_not_cross_tool_result_boundary(tmp_path):
    events = codex_rollout_sample()
    events[5], events[6] = events[6], events[5]
    write_codex_rollout(tmp_path, events)
    records, metrics = build_trajectory(tmp_path, "codex")
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
    records, metrics = build_trajectory(tmp_path, "codex")
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
    records, metrics = build_trajectory(tmp_path, "codex")
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
    records, metrics = build_trajectory(tmp_path, "codex")
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
    records, metrics = build_trajectory(tmp_path, "codex")
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
