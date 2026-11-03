"""Trae's history ordering, response usage, and interrupted capture regressions."""

import json

import pytest
from trajectory import build_trajectory

from .fixtures.trae import trae_rollout_sample, write_trae_rollout


def test_trae_completion_precedes_its_output_without_stealing_tool_results(tmp_path):
    write_trae_rollout(tmp_path, trae_rollout_sample())
    records, metrics = build_trajectory(tmp_path, "traecli")
    assert [r["message"]["role"] for r in records] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    responses = [r for r in records if r.get("call_id")]
    assert [r["message"]["usage_metadata"]["total_tokens"] for r in responses] == [
        13,
        23,
    ]
    assert responses[0]["message"]["tool_calls"][0]["id"] == "tool-one"
    assert responses[1]["message"]["content"] == [
        {"type": "output_text", "text": "fixed"}
    ]
    assert responses[1]["call_id"] == "trae-thread:response-two"
    assert records[3]["message"]["tool_call_id"] == "tool-one"
    assert "status" not in records[3]["message"]
    assert all(
        r["agent"] == "traecli" and r["run_id"] == "trae-thread" for r in records
    )
    assert metrics["tokens"]["total_tokens"] == 36
    assert metrics["tokens"]["cache_creation_input_tokens"] == 2
    assert metrics["model_responses_with_usage_count"] == 2
    assert metrics["usage_scope"] == "recorded-responses"
    assert metrics["usage_complete"] is False


def test_trae_repeated_views_and_commits_do_not_add_response_usage(tmp_path):
    events = trae_rollout_sample()
    events.insert(5, events[4])
    events.insert(4, events[3])
    events.insert(
        8,
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"total_token_usage": {"total_tokens": 999}},
            },
        },
    )
    events[-2]["payload"]["display_completions"] = [
        {
            "item": {
                "type": "AgentMessage",
                "id": "message-two",
                "content": [{"type": "Text", "text": "fixed"}],
            }
        }
    ]
    write_trae_rollout(tmp_path, events)
    records, metrics = build_trajectory(tmp_path, "traecli")
    assert len(records) == 5
    assert metrics["tokens"]["total_tokens"] == 36
    assert metrics["model_responses_with_usage_count"] == 2


def test_trae_stdout_uses_its_native_cache_creation_field(tmp_path):
    event = {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 30,
            "cached_input_tokens": 4,
            "cache_creation_input_tokens": 2,
            "output_tokens": 6,
            "reasoning_output_tokens": 2,
        },
    }
    (tmp_path / "agent_stdout.txt").write_text(json.dumps(event) + "\n")
    records, metrics = build_trajectory(tmp_path, "traecli")
    assert records == []
    assert metrics["tokens"]["total_tokens"] == 36
    assert metrics["tokens"]["cache_creation_input_tokens"] == 2
    assert metrics["usage_scope"] == "recorded-turns"
    assert metrics["source_usage"] == [event["usage"]]


@pytest.mark.parametrize(
    "ending", ["missing-completion", "unfinished-turn", "truncated-tail"]
)
def test_trae_partial_capture_preserves_content_and_unknown_totals(tmp_path, ending):
    events = trae_rollout_sample()
    if ending == "missing-completion":
        del events[6]
    elif ending == "unfinished-turn":
        events.pop()
    path = write_trae_rollout(tmp_path, events)
    if ending == "truncated-tail":
        with path.open("a") as stream:
            stream.write('{"type":')
    records, metrics = build_trajectory(tmp_path, "traecli")
    assert records[-1]["message"]["content"] == [
        {"type": "output_text", "text": "fixed"}
    ]
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["usage_complete"] is False
    if ending == "missing-completion":
        assert records[-1]["message"].get("usage_metadata") is None


def test_trae_replacement_history_is_evidence_not_new_model_execution(tmp_path):
    events = trae_rollout_sample()
    replacement = {
        "version": 1,
        "commit_id": "replacement",
        "operation": "replace",
        "items": events[7]["payload"]["items"],
    }
    events.insert(-1, {"type": "history_mutation", "payload": replacement})
    write_trae_rollout(tmp_path, events)
    records, metrics = build_trajectory(tmp_path, "traecli")
    assert len([r for r in records if r.get("call_id")]) == 2
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["source_history_mutations"] == [replacement]


def test_trae_child_capture_does_not_recount_copied_parent_history(tmp_path):
    parent = trae_rollout_sample()
    child = json.loads(json.dumps(parent))
    child[0]["payload"].update(
        id="child",
        session_id="child",
        source={"subagent": {"thread_spawn": {"parent_thread_id": "trae-thread"}}},
    )
    for event in child:
        if event["type"] == "token_usage_record":
            event["payload"].update(thread_id="child", session_id="child")
    write_trae_rollout(tmp_path, parent)
    switch = {
        "type": "event_msg",
        "payload": {"type": "thread_settings_applied", "thread_id": "child"},
    }
    write_trae_rollout(tmp_path, [child[0], *parent, switch, *child[1:]], name="child")
    records, metrics = build_trajectory(tmp_path, "traecli")
    assert {r["run_id"] for r in records} == {"trae-thread", "child"}
    assert all(
        r["parent_ids"] == ["trae-thread"] for r in records if r["run_id"] == "child"
    )
    assert metrics["tokens"]["total_tokens"] == 72
    assert metrics["model_responses_with_usage_count"] == 4


def test_trae_corrupt_complete_line_fails_capture(tmp_path):
    path = write_trae_rollout(tmp_path, trae_rollout_sample())
    with path.open("a") as stream:
        stream.write('{"type":broken}\n')
    with pytest.raises(ValueError, match="Malformed Trae"):
        build_trajectory(tmp_path, "traecli")


@pytest.mark.parametrize("value", [True, -1, 1.5, "10", 9007199254740992])
def test_trae_invalid_usage_cannot_enter_contract(tmp_path, value):
    events = trae_rollout_sample()
    events[3]["payload"]["usage"]["input_tokens"] = value
    write_trae_rollout(tmp_path, events)
    with pytest.raises(ValueError, match="nonnegative safe integers"):
        build_trajectory(tmp_path, "traecli")


def test_trae_developer_and_structured_user_context_remain_source_evidence(tmp_path):
    events = trae_rollout_sample()
    item = {
        "id": "developer-one",
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "runtime context"}],
    }
    events[2]["payload"]["items"].insert(0, item)
    attachment = {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
    events[2]["payload"]["items"][1]["content"].append(attachment)
    write_trae_rollout(tmp_path, events)
    records, _ = build_trajectory(tmp_path, "traecli")
    developer = next(
        r["message"] for r in records if r["message"]["id"].endswith(":developer-one")
    )
    assert developer["role"] == "system"
    assert developer["response_metadata"]["response_item"] == item
    user = next(r["message"] for r in records if r["message"]["role"] == "user")
    assert attachment in user["content"]


@pytest.mark.parametrize("count", [None, 0])
def test_trae_optional_subsets_preserve_unknown_and_explicit_zero(tmp_path, count):
    events = trae_rollout_sample()
    for event in events:
        if event["type"] != "token_usage_record":
            continue
        usage = event["payload"]["usage"]
        for field in ("cache_creation_input_tokens", "reasoning_output_tokens"):
            if count is None:
                del usage[field]
            else:
                usage[field] = count
    write_trae_rollout(tmp_path, events)
    records, metrics = build_trajectory(tmp_path, "traecli")
    responses = [r["message"] for r in records if r.get("call_id")]
    assert metrics["tokens"]["total_tokens"] == 36
    assert metrics["tokens"]["cache_creation_input_tokens"] == count
    assert metrics["tokens"]["reasoning_tokens"] == count
    assert all(
        m["usage_metadata"]["cache_creation_input_tokens"] == count for m in responses
    )


def test_trae_source_total_mismatch_keeps_original_measurement(tmp_path):
    events = trae_rollout_sample()
    events[3]["payload"]["usage"]["total_tokens"] = 999
    write_trae_rollout(tmp_path, events)
    with pytest.warns(RuntimeWarning, match="transcript_source_total_mismatch"):
        records, metrics = build_trajectory(tmp_path, "traecli")
    response = next(r["message"] for r in records if r.get("call_id"))
    assert response["usage_metadata"]["total_tokens"] == 13
    assert (
        response["response_metadata"]["token_usage_record"]["usage"]["total_tokens"]
        == 999
    )
    assert metrics["tokens"]["total_tokens"] == 36


def test_trae_conflicting_completion_cannot_reuse_response_identity(tmp_path):
    events = trae_rollout_sample()
    conflict = json.loads(json.dumps(events[3]))
    conflict["payload"]["usage"]["input_tokens"] = 11
    events.insert(4, conflict)
    write_trae_rollout(tmp_path, events)
    with pytest.raises(ValueError, match="Conflicting Trae response usage"):
        build_trajectory(tmp_path, "traecli")


def test_trae_compaction_is_preserved_without_claiming_known_usage(tmp_path):
    events = trae_rollout_sample()
    events.insert(-1, {"type": "compacted", "payload": {"message": "summary so far"}})
    write_trae_rollout(tmp_path, events)
    records, metrics = build_trajectory(tmp_path, "traecli")
    assert records[-1]["source"] == "summarization"
    assert records[-1]["message"]["content"] == "summary so far"
    assert metrics["tokens"]["total_tokens"] is None


def test_trae_native_capture_can_identify_agent_without_stdout(tmp_path):
    write_trae_rollout(tmp_path, trae_rollout_sample())
    records, metrics = build_trajectory(tmp_path)
    assert all(r["agent"] == "traecli" for r in records)
    assert metrics["agent"] == "traecli"
