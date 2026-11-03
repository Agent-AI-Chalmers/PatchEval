"""Execute host launchers and container entrypoints using recording test CLIs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("agent", ["codex", "opencode", "traecli"])
@pytest.mark.parametrize("runtime_present", [True, False])
def test_shell_entrypoint_uses_cleaned_input_and_proxy(
    tmp_path, agent, runtime_present
):
    if agent == "opencode" and not runtime_present:
        pytest.skip("OpenCode has no separate Code Mode runtime")
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "runtime"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "python"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    config = tmp_path / "profile.config.toml"
    if agent == "opencode":
        config = tmp_path / "config/opencode/opencode.json"
        config.parent.mkdir(parents=True)
        (tmp_path / "data").mkdir()
    if agent == "codex":
        helper = tmp_path / "codex-code-mode-host"
        if runtime_present:
            helper.write_bytes(binary.read_bytes())
            helper.chmod(0o755)
    if agent == "traecli":
        config = tmp_path / "profile.traecli.toml"
        helper = tmp_path / "traex-code-mode-host"
        if runtime_present:
            helper.write_bytes(binary.read_bytes())
            helper.chmod(0o755)
    config.write_text("{}")
    env = dict(
        os.environ,
        PATH=str(stub_dir) + os.pathsep + os.environ["PATH"],
        MODEL_API_UPSTREAM="https://provider.example/v1",
        MODEL_API_MODEL="test-model",
        MODEL_API_PROVIDER="deepseek",
    )
    if agent == "codex":
        env.update(
            CODEX_BIN=str(binary),
            CODEX_CONFIG=str(config),
            CODEX_MODEL_PROVIDER="provider",
        )
    elif agent == "traecli":
        env.update(
            TRAE_BIN=str(binary),
            TRAE_CONFIG=str(config),
            TRAE_MODEL_PROVIDER="provider",
        )
    else:
        env.update(
            OPENCODE_BIN=str(binary),
            OPENCODE_CONFIG=str(config),
            OPENCODE_PROVIDER="provider",
        )
    result = subprocess.run(
        ["bash", str(root / "patcheval/exp_agent/run_infer.sh"), agent, "test"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if not runtime_present:
        assert result.returncode != 0
        assert "Missing" in result.stderr and "Code Mode runtime" in result.stderr
        assert "--agent-command" not in result.stdout
        return
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index("--input") + 1].endswith(
        "patcheval_verified_cleaned.json"
    )
    assert (
        arguments[arguments.index("--model-api-upstream") + 1]
        == env["MODEL_API_UPSTREAM"]
    )
    assert arguments[arguments.index("--expect-model") + 1] == env["MODEL_API_MODEL"]
    assert (
        arguments[arguments.index("--model-provider") + 1] == env["MODEL_API_PROVIDER"]
    )
    assert arguments[arguments.index("--agent") + 1] == agent
    command = arguments[arguments.index("--agent-command") + 1]
    mounts = [
        arguments[i + 1] for i, value in enumerate(arguments) if value == "--mount"
    ]
    script_mount = next(m for m in mounts if m.endswith(":/opt/agent-run.sh:ro"))
    assert Path(script_mount.split(":")[0]).is_file()

    if agent in {"codex", "traecli"}:
        assert "bash /opt/agent-run.sh" in command
        assert "PATCHAGENT_PROFILE=profile" in command
        assert "PATCHAGENT_PROVIDER=provider" in command
    else:
        assert command == "bash /opt/agent-run.sh"
        assert arguments[arguments.index("--opencode-provider") + 1] == "provider"


@pytest.fixture(scope="module")
def bubblewrap():
    binary = shutil.which("bwrap")
    if binary is None:
        pytest.skip("bubblewrap is required to run absolute container paths safely")
    probe = subprocess.run(
        [binary, "--ro-bind", "/", "/", "--", "/bin/true"],
        capture_output=True,
        check=False,
    )
    if probe.returncode:
        pytest.skip("bubblewrap user namespaces are unavailable")
    return binary


@pytest.fixture
def run_container_entrypoint(tmp_path, bubblewrap):
    """Mount the real script in a private filesystem and substitute only its CLI."""
    root = Path(__file__).resolve().parents[1] / "patcheval/exp_agent"
    opt = tmp_path / "opt"
    opt.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    (results / "prompt.txt").write_text("repair sentinel")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    recorder = Path(__file__).parent / "fixtures/recording_cli.py"
    for name in ("codex", "traex", "opencode"):
        binary = binaries / name
        shutil.copyfile(recorder, binary)
        binary.chmod(0o755)
    shutil.copyfile(root / "model_api_bridge.py", opt / "model-api-bridge.py")
    for seed in (
        "codex-home-src",
        "trae-home-src",
        "opencode-config-src",
        "opencode-data-src",
    ):
        (opt / seed).mkdir()
    config = opt / "opencode-config-src/opencode/opencode.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "model": "selected/model",
                "small_model": "selected/model",
                "provider": {
                    "selected": {"options": {"baseURL": "https://original.invalid/v1"}}
                },
            }
        )
    )
    env = dict(
        os.environ,
        PATH="/usr/local/bin:/usr/bin:/bin",
        PATCHAGENT_PROFILE="profile",
        PATCHAGENT_PROVIDER="selected",
        PATCHAGENT_OPENCODE_PROVIDER="selected",
        PATCHAGENT_MODEL_API_BASE_URL="http://127.0.0.1:18080/v1",
        PATCHAGENT_WORKDIR="/workspace",
        PATCHAGENT_PROMPT_FILE="/results/prompt.txt",
    )
    filesystem = [
        bubblewrap,
        "--tmpfs",
        "/",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--dev",
        "/dev",
        "--dir",
        "/tmp",
        "--bind",
        str(opt),
        "/opt",
        "--bind",
        str(results),
        "/results",
        "--ro-bind",
        str(binaries),
        "/usr/local/bin",
        "--dir",
        "/workspace",
        "--chdir",
        "/workspace",
    ]

    def run(agent):
        script = {
            "codex": "codex-run.sh",
            "opencode": "opencode-run.sh",
            "traecli": "trae-run.sh",
        }[agent]
        shutil.copyfile(root / "agents/scripts" / script, opt / "agent-run.sh")
        result = subprocess.run(
            [*filesystem, "--", "bash", "/opt/agent-run.sh"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        calls = [
            json.loads(line)
            for line in (results / "calls.jsonl").read_text().splitlines()
        ]
        return calls, results

    return run


@pytest.mark.parametrize(
    "agent,home,private_home",
    [
        ("codex", "CODEX_HOME", "/tmp/codex-home"),
        ("traecli", "TRAE_HOME", "/tmp/trae-home"),
    ],
)
def test_codex_and_trae_entrypoints_isolate_configuration(
    run_container_entrypoint, agent, home, private_home
):
    calls, _ = run_container_entrypoint(agent)
    first = calls[0]
    assert first["stdin"] == "repair sentinel"
    argv = first["argv"]
    assert argv[argv.index("--profile") + 1] == "profile"
    assert "--ephemeral" not in argv
    assert "--ignore-rules" in argv
    disabled = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--disable"]
    assert set(disabled) == {"memories", "external_agent_memory_import", "hooks"}
    overrides = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"]
    assert 'web_search="disabled"' in overrides
    assert 'model_providers.selected.base_url="http://127.0.0.1:18080/v1"' in overrides
    assert first["env"][home] == private_home


@pytest.mark.parametrize(
    "agent,directory", [("codex", "codex-sessions"), ("traecli", "trae-sessions")]
)
def test_entrypoint_writes_rollouts_to_results(
    run_container_entrypoint, agent, directory
):
    _, results = run_container_entrypoint(agent)
    rollout = results / directory / "rollout-sentinel.jsonl"
    assert json.loads(rollout.read_text())["payload"]["usage"] == {"input_tokens": 7}
    assert not (results / "auth.json").exists()


def test_opencode_entrypoint_persists_database_and_exports_session(
    run_container_entrypoint,
):
    calls, results = run_container_entrypoint("opencode")
    first = calls[0]
    assert first["stdin"] == "repair sentinel"
    assert first["argv"] == ["run", "--pure", "--format", "json", "--auto"]
    assert (
        first["config"]["provider"]["selected"]["options"]["baseURL"]
        == "http://127.0.0.1:18080/v1"
    )
    for name in (
        "AUTOUPDATE",
        "PROJECT_CONFIG",
        "CLAUDE_CODE",
        "EXTERNAL_SKILLS",
        "DEFAULT_PLUGINS",
    ):
        assert first["env"]["OPENCODE_DISABLE_" + name] == "true"
    assert first["env"]["XDG_CONFIG_HOME"] == "/tmp/opencode-config"
    assert first["env"]["XDG_DATA_HOME"] == "/tmp/opencode-data"
    assert first["env"]["OPENCODE_DB"] == "/results/opencode.db"
    assert (results / "opencode.db").read_text() == "database-sentinel"
    assert calls[1]["argv"] == ["export", "session-sentinel"]
    assert (
        json.loads((results / "opencode-session.json").read_text())["info"]["id"]
        == "session-sentinel"
    )
