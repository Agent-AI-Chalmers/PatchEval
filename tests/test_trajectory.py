"""Artifact writing and shared transcript contract regressions."""

import json

import pytest
from trajectory import build_trajectory, write_trajectory
from trajectory.item_events import messages_from_events
from trajectory.opencode import opencode_messages

from .fixtures.codex import CODEX_EVENTS
from .fixtures.opencode import OPENCODE_SESSION


def test_both_clis_share_one_token_vocabulary():
    _, codex = messages_from_events(CODEX_EVENTS, "")
    _, opencode = opencode_messages(OPENCODE_SESSION, "")

    assert set(codex["tokens"]) == set(opencode["tokens"])
    assert codex["tokens"]["uncached_input_tokens"] is None
    assert opencode["tokens"]["uncached_input_tokens"] == 10


def test_model_provenance_is_recorded_on_assistant_messages(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "prompt.txt").write_text("fix the bug")
    (work / "agent_stdout.txt").write_text(
        "\n".join(json.dumps(event) for event in CODEX_EVENTS)
    )

    records, metrics = build_trajectory(work, "codex", "deepseek", "deepseek-v4-pro")

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

    trajectory, metrics = write_trajectory(work, tmp_path / "trajectories", "CVE-test")

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
    trajectory, metrics = write_trajectory(work, tmp_path / "trajectories", "CVE-test")

    payload = json.loads(metrics.read_text())
    assert payload["agent"] == "opencode"
    assert payload["cost"] == 0.01
    assert payload["tool_calls"] == 1
    assert payload["usage_complete"] is False
    assert payload["notices"]
    assert "partial stream" not in trajectory.read_text()


@pytest.mark.parametrize("agent", ["traecli", "opencode"])
def test_explicit_agent_identity_survives_missing_session_export(tmp_path, agent):
    (tmp_path / "agent_stdout.txt").write_text(
        "\n".join(json.dumps(event) for event in CODEX_EVENTS)
    )
    records, metrics = build_trajectory(tmp_path, agent_kind=agent)
    assert metrics["agent"] == agent
    if agent == "opencode":
        assert records == []
        assert metrics["notices"]
    else:
        assert metrics["tool_calls"] == 1


def test_export_uses_contract_roles_and_unique_message_ids(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "prompt.txt").write_text("fix")
    (work / "agent_stdout.txt").write_text(
        "\n".join(json.dumps(event) for event in CODEX_EVENTS)
    )
    records, _ = build_trajectory(work, "codex")
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


@pytest.mark.parametrize("agent", ["codex", "opencode", "traecli"])
def test_export_records_obey_message_and_usage_contract(tmp_path, agent):
    """Use this adapter's own CLI samples, without the main package or its corpus."""
    if agent in {"codex", "traecli"}:
        (tmp_path / "agent_stdout.txt").write_text(
            "\n".join(json.dumps(event) for event in CODEX_EVENTS)
        )
    else:
        (tmp_path / "opencode-session.json").write_text(json.dumps(OPENCODE_SESSION))
    records, metrics = build_trajectory(tmp_path, agent)
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


@pytest.mark.parametrize("value", [True, -1, 1.5, "10", 9007199254740992])
def test_invalid_cli_token_counts_are_rejected(value):
    with pytest.raises(ValueError):
        messages_from_events(
            [{"type": "turn.completed", "usage": {"input_tokens": value}}], ""
        )


def test_source_total_alone_cannot_establish_canonical_total():
    _, metrics = messages_from_events(
        [{"type": "turn.completed", "usage": {"total_tokens": 100}}], ""
    )
    assert all(value is None for value in metrics["tokens"].values())
    assert metrics["source_usage"] == [{"total_tokens": 100}]


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
        build_trajectory(tmp_path, "codex")
