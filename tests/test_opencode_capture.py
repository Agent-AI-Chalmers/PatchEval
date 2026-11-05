"""Check the installed OpenCode CLI's persisted format without an external model."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from trajectory import build_trajectory


@pytest.mark.skipif(
    shutil.which("opencode") is None, reason="OpenCode CLI is not installed"
)
@pytest.mark.parametrize("instruction_file", ["AGENTS.md", "CLAUDE.md"])
def test_real_opencode_database_usage_and_auxiliary_coverage(
    tmp_path, instruction_file
):
    """Title generation consumes tokens but is absent from native session recording."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            identity = str(len(requests))
            item = {
                "id": "msg-" + identity,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "probe done", "annotations": []}
                ],
            }
            response = {
                "id": "response-" + identity,
                "object": "response",
                "status": "completed",
                "output": [item],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                    "input_tokens_details": {"cached_tokens": 2},
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
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "probe done",
                },
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": response},
            ]
            data = "".join(
                f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "config/opencode"
    config.mkdir(parents=True)
    (config / "opencode.json").write_text(
        json.dumps(
            {
                "model": "probe/probe-model",
                "small_model": "probe/probe-model",
                "provider": {
                    "probe": {
                        "npm": "@ai-sdk/openai",
                        "options": {
                            "baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                            "apiKey": "local-probe",
                        },
                        "models": {"probe-model": {"name": "probe-model"}},
                    }
                },
            }
        )
    )
    (tmp_path / instruction_file).write_text(
        "Repository instruction sentinel: repo-guide-731.\n"
    )
    models = tmp_path / "models.json"
    models.write_text("{}")
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        PWD=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        XDG_STATE_HOME=str(tmp_path / "state"),
        OPENCODE_DB=str(tmp_path / "opencode.db"),
        OPENCODE_MODELS_PATH=str(models),
        OPENCODE_DISABLE_MODELS_FETCH="true",
        OPENCODE_DISABLE_AUTOUPDATE="true",
        OPENCODE_DISABLE_DEFAULT_PLUGINS="true",
        OPENCODE_DISABLE_EXTERNAL_SKILLS="true",
        OPENCODE_DISABLE_CLAUDE_CODE_SKILLS="true",
    )
    for name in ("PROJECT_CONFIG", "CLAUDE_CODE", "CLAUDE_CODE_PROMPT"):
        env.pop("OPENCODE_DISABLE_" + name, None)
    try:
        result = subprocess.run(
            [
                shutil.which("opencode"),
                "run",
                "--pure",
                "--format",
                "json",
                "--auto",
                "--dir",
                str(tmp_path),
                "Reply briefly.",
            ],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode == 0, result.stderr
    assert len(requests) == 2
    assert {r["model"] for r in requests} == {"probe-model"}
    assert sum(bool(r.get("tools")) for r in requests) == 1
    main_request = next(r for r in requests if r.get("tools"))
    assert "repo-guide-731" in json.dumps(main_request)
    assert (tmp_path / "opencode.db").is_file()
    assert not (tmp_path / "opencode-session.json").exists()
    records, metrics = build_trajectory(tmp_path, "opencode")
    responses = [r for r in records if r["message"]["role"] == "assistant"]
    assert len(responses) == 1
    message = responses[0]["message"]
    assert message["usage_metadata"] == {
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 1,
    }
    assert message["response_metadata"]["token_usage"]["output"] == 4
    assert message["response_metadata"]["opencode_message"]["providerID"] == "probe"
    assert message["content"] == [{"type": "text", "text": "probe done"}]
    assert responses[0]["call_id"] == message["id"]
    assert metrics["tokens"]["total_tokens"] == 25
    assert metrics["usage_scope"] == "recorded-responses"
    assert metrics["model_responses_with_usage_count"] == 1
    assert metrics["usage_complete"] is False
