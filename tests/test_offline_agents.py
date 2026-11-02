import http.client
import importlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def modules(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "patcheval" / "exp_agent"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("model_api_proxy"), importlib.import_module(
        "patch_agent_runner"
    )


@pytest.fixture
def upstream():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, body, self.headers.get("Authorization")))
            if json.loads(body).get("redirect"):
                self.send_response(307)
                self.send_header("Location", "https://example.org/unrestricted")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"text":"ok"}\n\ndata: [DONE]\n\n')
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


MODEL = "test"


def request(directory, path, body, method="POST", *, model=MODEL):
    body = {**body, **({"model": model} if model is not None else {})}
    connection = http.client.HTTPConnection("localhost")
    connection.sock = socket.socket(socket.AF_UNIX)
    connection.sock.connect(str(directory / "api.sock"))
    connection.request(
        method,
        path,
        json.dumps(body) if method == "POST" else None,
        {"Authorization": "Bearer test", "Content-Type": "application/json"},
    )
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


def test_inference_only_proxy_preserves_stream_and_auth(modules, upstream):
    proxy, _ = modules
    url, requests = upstream
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, body = request(
            directory, "/v1/responses", {"model": "test", "stream": True}
        )
        assert status == 200
        assert body == b'data: {"text":"ok"}\n\ndata: [DONE]\n\n'
        assert requests == [
            ("/v1/responses", b'{"model": "test", "stream": true}', "Bearer test")
        ]
    assert not directory.exists()


@pytest.mark.parametrize(
    "path,body,method",
    [
        ("https://example.org/v1/responses", {}, "POST"),
        ("/v1/files", {}, "POST"),
        ("/v1/responses?target=example.org", {}, "POST"),
        ("/v1/responses", {"tools": [{"type": "web_search"}]}, "POST"),
        ("/v1/messages", {"tools": [{"type": "web_fetch_20250910"}]}, "POST"),
        ("/v1/messages", {"mcp_servers": [{"url": "https://example.org"}]}, "POST"),
        ("/v1/responses", {}, "CONNECT"),
    ],
)
def test_proxy_blocks_other_network_capabilities(modules, upstream, path, body, method):
    proxy, _ = modules
    url, requests = upstream
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, path, body, method)
    assert status in {403, 405}
    assert requests == []


def test_prompt_does_not_leak_answer_fields(modules):
    _, runner = modules
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


def test_generation_uses_cleaned_dataset_and_no_network(modules, tmp_path):
    _, runner = modules
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


def test_evaluation_container_has_no_network(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "patcheval" / "evaluation"
    monkeypatch.syspath_prepend(str(directory))
    evaluation = importlib.import_module("run_evaluation")
    client = SimpleNamespace(containers=Mock())
    monkeypatch.setattr(evaluation.docker, "from_env", lambda: client)
    manager = evaluation.DockerManager(Mock(), "CVE-test")
    assert manager.start_container("image", "container", "patch") == "container"
    assert client.containers.run.call_args.kwargs["network_mode"] == "none"


@pytest.mark.parametrize("model", ["other-deployment", None])
def test_proxy_rejects_other_models(modules, upstream, model):
    proxy, _ = modules
    url, requests = upstream
    body = {} if model is None else {"model": model}
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, "/v1/responses", body, model=None)
    assert status == 403
    assert requests == []


@pytest.mark.parametrize("agent", ["codex", "opencode", "traecli"])
@pytest.mark.parametrize("runtime_present", [True, False])
def test_shell_entrypoint_uses_cleaned_input_and_proxy(
    tmp_path, agent, runtime_present
):
    import os
    import subprocess

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


def test_proxy_does_not_follow_upstream_redirect(modules, upstream):
    proxy, _ = modules
    url, requests = upstream
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, body = request(directory, "/v1/responses", {"redirect": True})
    assert status == 502
    assert len(requests) == 1
    assert b"example.org" not in body


@pytest.mark.parametrize(
    "path,tool",
    [
        ("/v1/messages", {"type": "code_execution_20250522", "name": "code_execution"}),
        ("/v1/messages", {"type": "future_server_tool", "name": "remote"}),
        ("/v1/responses", {"type": "shell", "environment": {"type": "container_auto"}}),
        ("/v1/responses", {"type": "future_server_tool"}),
        ("/v1/chat/completions", {"type": "custom", "name": "unsupported"}),
        ("/v1/responses", {"name": "missing_type"}),
    ],
)
def test_proxy_rejects_unsupported_tool_types(modules, upstream, path, tool):
    proxy, _ = modules
    url, requests = upstream
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, path, {"tools": [tool]})
    assert status == 403
    assert requests == []


@pytest.mark.parametrize(
    "path,tool",
    [
        ("/v1/messages", {"name": "read_file", "input_schema": {"type": "object"}}),
        (
            "/v1/messages/count_tokens",
            {"type": "custom", "name": "read_file", "input_schema": {"type": "object"}},
        ),
        (
            "/v1/chat/completions",
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            },
        ),
        (
            "/v1/responses",
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
        ),
        (
            "/v1/responses",
            {"type": "custom", "name": "apply_patch", "format": {"type": "text"}},
        ),
        ("/v1/responses", {"type": "local_shell"}),
    ],
)
def test_proxy_preserves_supported_client_tools(modules, upstream, path, tool):
    proxy, _ = modules
    url, requests = upstream
    payload = {"tools": [tool]}
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, path, payload)
    assert status == 200
    assert len(requests) == 1
    assert json.loads(requests[0][1]) == {**payload, "model": MODEL}


@pytest.mark.parametrize(
    "path,block",
    [
        (
            "/v1/responses",
            {"type": "input_image", "image_url": "https://example.org/image.png"},
        ),
        (
            "/v1/responses/compact",
            {"type": "input_file", "file_url": "https://example.org/file.pdf"},
        ),
        ("/v1/responses", {"type": "input_file", "file_id": "file-remote"}),
        (
            "/v1/chat/completions",
            {
                "type": "image_url",
                "image_url": {"url": "https://example.org/image.png"},
            },
        ),
        ("/v1/chat/completions", {"type": "file", "file": {"file_id": "file-remote"}}),
        (
            "/v1/messages",
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.org/image.png"},
            },
        ),
        (
            "/v1/messages/count_tokens",
            {"type": "document", "source": {"type": "file", "file_id": "file-remote"}},
        ),
    ],
)
def test_proxy_rejects_external_media_inputs(modules, upstream, path, block):
    proxy, _ = modules
    url, requests = upstream
    field = "input" if "/responses" in path else "messages"
    payload = {field: [{"role": "user", "content": [block]}]}
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, path, payload)
    assert status == 403
    assert requests == []


@pytest.mark.parametrize(
    "path,block",
    [
        (
            "/v1/responses",
            {
                "type": "input_text",
                "text": "Read https://example.org; image_url is a field name.",
            },
        ),
        (
            "/v1/responses",
            {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
        ),
        (
            "/v1/responses",
            {
                "type": "input_file",
                "filename": "test.pdf",
                "file_data": "data:application/pdf;base64,AA==",
            },
        ),
        (
            "/v1/chat/completions",
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ),
        (
            "/v1/messages",
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
            },
        ),
        (
            "/v1/messages",
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "https://example.org",
                },
            },
        ),
    ],
)
def test_proxy_preserves_inline_media_and_text(modules, upstream, path, block):
    proxy, _ = modules
    url, requests = upstream
    field = "input" if "/responses" in path else "messages"
    payload = {field: [{"role": "user", "content": [block]}]}
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, path, payload)
    assert status == 200
    assert json.loads(requests[0][1]) == {**payload, "model": MODEL}


@pytest.mark.parametrize(
    "nested,expected",
    [
        ({"type": "function", "name": "read_file"}, 200),
        ({"type": "web_search"}, 403),
        (
            {
                "type": "namespace",
                "name": "nested",
                "tools": [{"type": "code_interpreter"}],
            },
            403,
        ),
    ],
)
def test_namespace_tools_preserve_offline_boundary(modules, upstream, nested, expected):
    proxy, _ = modules
    url, requests = upstream
    payload = {"tools": [{"type": "namespace", "name": "tools", "tools": [nested]}]}
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, "/v1/responses", payload)
    assert status == expected
    assert len(requests) == (1 if expected == 200 else 0)


@pytest.mark.parametrize(
    "target,config",
    [
        ("/opt/codex-home-src", "probe.config.toml"),
        ("/opt/trae-home-src", "probe.traecli.toml"),
        ("/opt/opencode-config-src", "opencode/opencode.json"),
        ("/opt/opencode-data-src", "opencode/auth.json"),
    ],
)
def test_agent_seeds_exclude_personal_state(modules, tmp_path, target, config):
    _, runner = modules
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
    modules, tmp_path, monkeypatch, cve, expected
):
    import asyncio

    _, runner = modules
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
    assert [
        json.loads(line)["cve"]
        for line in (Path(args.run_root) / "results.jsonl").read_text().splitlines()
    ] == expected


@pytest.mark.parametrize("agent", ["codex", "opencode", "traecli"])
def test_runner_exports_selected_agent_identity(modules, tmp_path, monkeypatch, agent):
    import asyncio

    _, runner = modules
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


@pytest.mark.parametrize("agent", ["codex", "opencode", "traecli"])
def test_container_entrypoint_executes_isolation_flags_and_exports(tmp_path, agent):
    """Run the unchanged entrypoint in a private filesystem with a recording CLI."""
    import os
    import shutil
    import subprocess

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is required to run absolute container paths safely")
    probe = subprocess.run(
        [bwrap, "--ro-bind", "/", "/", "--", "/bin/true"], capture_output=True
    )
    if probe.returncode:
        pytest.skip("bubblewrap user namespaces are unavailable")
    root = Path(__file__).resolve().parents[1] / "patcheval/exp_agent"
    opt = tmp_path / "opt"
    opt.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    (results / "prompt.txt").write_text("repair sentinel")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    recorder = """#!/usr/bin/python3
import json, os, sys
from pathlib import Path
config = Path('/tmp/opencode-config/opencode/opencode.json')
configured = json.loads(config.read_text()) if config.exists() else None
with Path('/results/calls.jsonl').open('a') as handle:
    handle.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ), 'stdin': sys.stdin.read(), 'config': configured}) + '\\n')
if sys.argv[1] == 'run':
    print('{"sessionID":"session-sentinel"}')
elif sys.argv[1] == 'export':
    print('{"info":{"id":"session-sentinel"},"messages":[]}')
"""
    for name in ("codex", "traex", "opencode"):
        binary = binaries / name
        binary.write_text(recorder)
        binary.chmod(0o755)
    script = {
        "codex": "codex-run.sh",
        "opencode": "opencode-run.sh",
        "traecli": "trae-run.sh",
    }[agent]
    shutil.copyfile(root / "agents/scripts" / script, opt / "agent-run.sh")
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
    result = subprocess.run(
        [
            bwrap,
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
            "--",
            "bash",
            "/opt/agent-run.sh",
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line) for line in (results / "calls.jsonl").read_text().splitlines()
    ]
    first = calls[0]
    assert first["stdin"] == "repair sentinel"
    if agent == "opencode":
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
        assert calls[1]["argv"] == ["export", "session-sentinel"]
        assert (
            json.loads((results / "opencode-session.json").read_text())["info"]["id"]
            == "session-sentinel"
        )
    else:
        argv = first["argv"]
        assert argv[argv.index("--profile") + 1] == "profile"
        assert "--ephemeral" in argv and "--ignore-rules" in argv
        disabled = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--disable"]
        assert set(disabled) == {"memories", "external_agent_memory_import", "hooks"}
        overrides = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"]
        assert 'web_search="disabled"' in overrides
        assert (
            'model_providers.selected.base_url="http://127.0.0.1:18080/v1"' in overrides
        )
        home = "CODEX_HOME" if agent == "codex" else "TRAE_HOME"
        assert first["env"][home] == (
            "/tmp/codex-home" if agent == "codex" else "/tmp/trae-home"
        )


@pytest.mark.parametrize("agent", ["codex", "traecli", "opencode"])
@pytest.mark.parametrize("timeout", [False, True])
def test_agent_output_survives_timeout(modules, tmp_path, monkeypatch, agent, timeout):
    """Real subprocess output must survive cancellation at the inference deadline."""
    import asyncio
    import sys

    _, runner = modules
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
    # Signal that both streams have been written before allowing the timeout.
    program = (
        "import sys,time,pathlib; "
        f"sys.stdout.write({output!r}); sys.stdout.flush(); "
        "sys.stderr.write('diagnostic\\n'); sys.stderr.flush(); "
        f"pathlib.Path({str(marker)!r}).touch(); "
        f"time.sleep({30 if timeout else 0})"
    )

    async def spawn_agent(*args, **kwargs):
        process = await spawn(sys.executable, "-u", "-c", program, **kwargs)
        while not marker.exists():
            await asyncio.sleep(0.01)
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
        assert metrics["tokens"] == {}
        assert metrics["notices"]
        assert records[-1]["message"]["content"] == "a.txt"
