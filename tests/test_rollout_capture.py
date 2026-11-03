"""Exercise Codex and Trae captures against a local Responses endpoint, without credentials."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from trajectory import build_trajectory


@pytest.mark.parametrize(
    "agent,binary,home_variable,sessions_directory",
    [
        ("codex", "codex", "CODEX_HOME", "codex-sessions"),
        ("traecli", "traex", "TRAE_HOME", "trae-sessions"),
    ],
)
def test_real_cli_rollout_preserves_two_response_measurements(
    tmp_path, agent, binary, home_variable, sessions_directory
):
    """Stdout has one turn total; rollouts must retain both upstream responses."""
    executable = shutil.which(binary)
    if executable is None:
        pytest.skip(f"{binary} CLI is not installed")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body["model"])
            index = len(requests)
            if index == 1:
                # A nonexistent tool produces a local error and another model call,
                # without allowing the probe to execute a shell command.
                item = {
                    "id": "fc-probe",
                    "type": "function_call",
                    "call_id": "call-probe",
                    "name": "probe_nonexistent_tool",
                    "arguments": "{}",
                }
            else:
                item = {
                    "id": "msg-probe",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "probe done", "annotations": []}
                    ],
                }
            response = {
                "id": f"response-probe-{index}",
                "object": "response",
                "status": "completed",
                "output": [item],
                "usage": {
                    "input_tokens": 10 * index,
                    "output_tokens": 3,
                    "total_tokens": 10 * index + 3,
                    "input_tokens_details": {
                        "cached_tokens": 2,
                        "cache_write_tokens": 1,
                    },
                    "output_tokens_details": {"reasoning_tokens": 1},
                },
            }
            events = [
                {
                    "type": "response.created",
                    "response": {
                        "id": response["id"],
                        "status": "in_progress",
                        "output": [],
                    },
                },
                {"type": "response.output_item.added", "output_index": 0, "item": item},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": response},
            ]
            data = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    home = tmp_path / "private-home"
    home.mkdir()
    work = tmp_path / "case"
    sessions = work / sessions_directory
    sessions.mkdir(parents=True)
    session_root = home / "cli" if agent == "traecli" else home
    session_root.mkdir(exist_ok=True)
    (session_root / "sessions").symlink_to(sessions, target_is_directory=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [
                executable,
                "exec",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--json",
                "--disable",
                "memories",
                "--disable",
                "hooks",
                "-C",
                str(work),
                "-c",
                'model="probe-model"',
                "-c",
                'model_provider="probe"',
                "-c",
                'model_providers.probe.name="probe"',
                "-c",
                f'model_providers.probe.base_url="http://127.0.0.1:{server.server_port}/v1"',
                "-c",
                'model_providers.probe.wire_api="responses"',
                "-c",
                "model_providers.probe.requires_openai_auth=false",
                "-c",
                'web_search="disabled"',
                "Reply briefly.",
            ],
            env={
                **os.environ,
                home_variable: str(home),
                "HOME": str(tmp_path / "user-home"),
                "RUST_LOG": "off",
            },
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode == 0, result.stderr
    assert requests == ["probe-model", "probe-model"]
    (work / "agent_stdout.txt").write_text(result.stdout)
    records, metrics = build_trajectory(work, agent)
    responses = [record for record in records if record.get("call_id")]
    assert [
        record["message"]["usage_metadata"]["total_tokens"] for record in responses
    ] == [13, 23]
    assert responses[0]["message"]["tool_calls"][0]["id"] == "call-probe"
    assert responses[1]["message"]["content"][0]["text"] == "probe done"
    assert metrics["tokens"]["total_tokens"] == 36
    assert metrics["tokens"]["cache_creation_input_tokens"] == 2
    assert metrics["model_responses_with_usage_count"] == 2
    assert metrics["usage_scope"] == "recorded-responses"
    assert metrics["usage_complete"] is False
