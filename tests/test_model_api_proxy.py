"""Exercise the inference-only network boundary with a local HTTP upstream."""

import http.client
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import model_api_proxy as proxy
import pytest


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


def test_inference_only_proxy_preserves_stream_and_auth(upstream):
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
def test_proxy_blocks_other_network_capabilities(upstream, path, body, method):
    url, requests = upstream
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, path, body, method)
    assert status in {403, 405}
    assert requests == []


@pytest.mark.parametrize("model", ["other-deployment", None])
def test_proxy_rejects_other_models(upstream, model):
    url, requests = upstream
    body = {} if model is None else {"model": model}
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, "/v1/responses", body, model=None)
    assert status == 403
    assert requests == []


def test_proxy_does_not_follow_upstream_redirect(upstream):
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
def test_proxy_rejects_unsupported_tool_types(upstream, path, tool):
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
def test_proxy_preserves_supported_client_tools(upstream, path, tool):
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
def test_proxy_rejects_external_media_inputs(upstream, path, block):
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
def test_proxy_preserves_inline_media_and_text(upstream, path, block):
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
def test_namespace_tools_preserve_offline_boundary(upstream, nested, expected):
    url, requests = upstream
    payload = {"tools": [{"type": "namespace", "name": "tools", "tools": [nested]}]}
    with proxy.model_api_proxy(url, expected_model=MODEL) as directory:
        status, _ = request(directory, "/v1/responses", payload)
    assert status == expected
    assert len(requests) == (1 if expected == 200 else 0)
