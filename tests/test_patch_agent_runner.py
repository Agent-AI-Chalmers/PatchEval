"""Verify case selection, seed isolation, and durable runner output without Docker."""

import asyncio
import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner():
    return importlib.import_module("patch_agent_runner")


def test_prompt_does_not_leak_answer_fields(runner):
    sample = {
        "cve_id": "CVE-0000-0000",
        "cve_description": "described bug",
        "vul_func": [{"snippet": "ANSWER_SNIPPET", "commit": "ANSWER_COMMIT"}],
        "fix_func": "ANSWER_FIX",
        "patch_url": "https://example.org/ANSWER_COMMIT",
    }

    prompt = runner._prompt(sample, "/workspace/project")

    assert "described bug" in prompt
    assert "ANSWER" not in prompt


def test_generation_uses_cleaned_dataset_and_no_network(runner, tmp_path):
    args = runner.build_parser().parse_args(
        [
            "--agent-command",
            "command",
            "--model-api-upstream",
            "https://example.org/v1",
            "--expect-model",
            "test-model",
            "--model-provider",
            "deepseek",
        ]
    )
    assert args.expect_model == "test-model"
    assert args.model_provider == "deepseek"
    assert args.input.endswith("patcheval_verified_cleaned.json")
    command = runner._docker_run_args("container", "image", tmp_path, [])
    assert command[command.index("--network") + 1] == "none"


@pytest.mark.parametrize(
    "target,config",
    [
        ("/opt/codex-home-src", "probe.config.toml"),
        ("/opt/trae-home-src", "probe.traecli.toml"),
        ("/opt/opencode-config-src", "opencode/opencode.json"),
        ("/opt/opencode-data-src", "opencode/auth.json"),
    ],
)
def test_agent_seeds_exclude_personal_state(runner, tmp_path, target, config):
    source = tmp_path / "source"
    source.mkdir()
    resource = source / config
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text("configured input")
    for name in [
        "AGENTS.md",
        "history.jsonl",
        "memories/memory_summary.md",
        "sessions/previous.json",
        "opencode/opencode.db",
        "opencode/storage/session/old.json",
    ]:
        item = source / name
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("private sentinel")
    destination = tmp_path / "seeds"
    destination.mkdir()
    mounts = runner.isolated_agent_mounts([(str(source), target, "ro")], destination)
    exposed = Path(mounts[0][0])
    assert mounts[0][1:] == (target, "ro")
    assert (exposed / config).read_text() == "configured input"
    assert [
        p.relative_to(exposed).as_posix() for p in exposed.rglob("*") if p.is_file()
    ] == [config]


@pytest.mark.parametrize(
    "cve,expected", [(None, ["CVE-first"]), ("CVE-second", ["CVE-second"])]
)
def test_case_selection_is_explicit_and_recorded(
    runner, tmp_path, monkeypatch, cve, expected
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {"cve_id": name, "image_url": "test-image"}
                for name in ["CVE-first", "CVE-second"]
            ]
        )
    )
    args = runner.build_parser().parse_args(
        [
            "--input",
            str(dataset),
            "--output-dir",
            str(tmp_path / "output"),
            "--agent-command",
            "test",
            "--model-api-upstream",
            "http://localhost/v1",
            "--expect-model",
            "test-model",
        ]
    )
    args.cve = cve
    args.seed_directory = str(tmp_path)
    snapshot = {"reasoning_effort": {"requested": "low", "configured": "low"}}
    args.launch_config = tmp_path / "launch.json"
    args.launch_config.write_text(json.dumps(snapshot))

    async def run_one(sample, index, args, sem, mounts):
        return runner.GenerationResult(
            index,
            sample["cve_id"],
            "test",
            "test-image",
            "/workspace",
            "test",
            "failed",
            False,
            None,
            False,
            0,
            "",
            "fixture",
        )

    monkeypatch.setattr(runner, "_run_one", run_one)
    assert asyncio.run(runner._generate(args)) == 1
    record = json.loads((Path(args.run_root) / "run_metadata.json").read_text())
    assert record["selected_cves"] == expected
    assert record["launch_config"] == snapshot
    assert record["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert [
        json.loads(line)["cve"]
        for line in (Path(args.run_root) / "results.jsonl").read_text().splitlines()
    ] == expected


@pytest.mark.parametrize("agent", ["codex", "opencode", "traecli"])
def test_runner_exports_selected_agent_identity(runner, tmp_path, monkeypatch, agent):
    args = runner.build_parser().parse_args(
        [
            "--agent",
            agent,
            "--agent-command",
            "test",
            "--model-api-upstream",
            "http://localhost/v1",
            "--expect-model",
            "test",
        ]
    )
    args.run_root = str(tmp_path)

    async def failed_start(*args, **kwargs):
        return runner.CommandResult([], 1, "", "fixture failure", 0)

    async def remove(*args):
        pass

    export = Mock()
    monkeypatch.setattr(runner, "_run", failed_start)
    monkeypatch.setattr(runner, "_remove_container", remove)
    monkeypatch.setattr(runner, "write_trajectory", export)
    asyncio.run(
        runner._run_one(
            {"cve_id": "CVE-test", "image_url": "fixture"},
            0,
            args,
            asyncio.Semaphore(1),
            [],
        )
    )
    assert export.call_args.kwargs["agent_kind"] == agent


@pytest.mark.parametrize("agent", ["codex", "traecli", "opencode"])
@pytest.mark.parametrize("probe_failed", [False, True, "spawn_error"])
def test_generated_case_records_actual_runtime(
    runner, tmp_path, monkeypatch, agent, probe_failed
):
    """Probe the case container, and retain explicit unknowns if probes fail."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "cve_id": "CVE-test",
                    "image_url": "mutable:latest",
                    "workdir": "/workspace/project",
                }
            ]
        )
    )
    args = runner.build_parser().parse_args(
        [
            "--input",
            str(dataset),
            "--output-dir",
            str(tmp_path / "output"),
            "--agent",
            agent,
            "--agent-command",
            "fixture",
            "--model-api-upstream",
            "http://localhost/v1",
            "--expect-model",
            "test-model",
        ]
    )
    args.seed_directory = str(tmp_path)
    image_id = "sha256:" + "a" * 64
    version_command = (
        f"/usr/local/bin/{'traex' if agent == 'traecli' else agent} --version"
    )
    probes = []

    async def run(command, **kwargs):
        if "inspect" in command:
            probes.append(command)
            if probe_failed == "spawn_error":
                raise OSError("synthetic spawn failure")
            return runner.CommandResult(
                command, 1 if probe_failed else 0, image_id, "", 0
            )
        return runner.CommandResult(command, 0, "", "", 0)

    async def exec_container(container, command, **kwargs):
        if command == version_command:
            probes.append(command)
            if probe_failed == "spawn_error":
                raise OSError("synthetic spawn failure")
            return runner.CommandResult(
                [], 1 if probe_failed else 0, "CLI 1.2.3\n", "", 0
            )
        if "git diff HEAD" in command:
            work = next((Path(args.run_root) / ".work").iterdir())
            (work / "llm.patch").write_text("fixture patch")
        return runner.CommandResult([], 0, "", "", 0)

    async def agent_run(*args, **kwargs):
        return runner.CommandResult([], 0, "", "", 0)

    monkeypatch.setattr(runner, "_run", run)
    monkeypatch.setattr(runner, "_docker_exec", exec_container)
    monkeypatch.setattr(runner, "_run_agent", agent_run)
    assert asyncio.run(runner._generate(args)) == 0
    result = json.loads((Path(args.run_root) / "results.jsonl").read_text())
    assert result["status"] == "generated"
    assert result["image"] == "mutable:latest"
    assert result["runtime"] == {
        "image_id": None if probe_failed else image_id,
        "cli_version": None if probe_failed else "1.2.3",
        "capture_errors": (
            ["image_id_unavailable", "cli_version_unavailable"] if probe_failed else []
        ),
    }
    assert len(probes) == 2
    assert probes[0][-1] == result["container_name"]
    assert probes[1] == version_command
    metadata = json.loads((Path(args.run_root) / "run_metadata.json").read_text())
    assert metadata["launch_config"] is None


@pytest.mark.parametrize("agent", ["codex", "traecli", "opencode"])
@pytest.mark.parametrize("timeout", [False, True])
def test_agent_output_survives_timeout(runner, tmp_path, monkeypatch, agent, timeout):
    """Real subprocess output must survive cancellation at the inference deadline."""
    spawn = asyncio.create_subprocess_exec
    marker = tmp_path / "ready"
    event = {
        "type": "item.completed",
        "item": {
            "id": "item-1",
            "type": "command_execution",
            "command": "ls",
            "aggregated_output": "a.txt",
            "exit_code": 0,
        },
    }
    output = json.dumps(event) + "\n"
    program = Path(__file__).parent / "fixtures/streaming_cli.py"

    async def spawn_agent(*args, **kwargs):
        process = await spawn(
            sys.executable,
            "-u",
            str(program),
            output,
            str(marker),
            "30" if timeout else "0",
            **kwargs,
        )

        async def wait_until_ready():
            while not marker.exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_until_ready(), timeout=5)
        return process

    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", spawn_agent)
    result = asyncio.run(
        runner._run_agent(
            "fixture",
            "/workspace",
            agent,
            tmp_path,
            {},
            0.1 if timeout else 5,
        )
    )
    assert result.timed_out is timeout
    assert result.stdout == output
    assert result.stderr == "diagnostic\n"
    assert (tmp_path / "agent_stdout.txt").read_text() == output
    assert (tmp_path / "agent_stderr.txt").read_text() == "diagnostic\n"
    assert (result.exit_code != 0) is timeout
    if agent != "opencode":
        from trajectory import build_trajectory

        records, metrics = build_trajectory(
            tmp_path,
            agent_kind=agent,
        )
        assert metrics["tool_calls"] == 1
        assert all(value is None for value in metrics["tokens"].values())
        assert metrics["notices"]
        assert records[-1]["message"]["content"] == "a.txt"
