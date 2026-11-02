"""Forward inference requests over a Unix socket to one fixed API base URL."""

import http.client
import json
import socketserver
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit


def has_external_media(value):
    """Reject provider-fetched media while retaining inline data and prose URLs."""
    if isinstance(value, list):
        return any(has_external_media(item) for item in value)
    if not isinstance(value, dict):
        return False
    kind = value.get("type")
    if kind in {"input_image", "image_url"}:
        image = value.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        if not isinstance(url, str) or not url.startswith("data:"):
            return True
    if kind in {"input_file", "file"}:
        file = value.get("file", value)
        if not isinstance(file, dict) or file.get("file_id") or file.get("file_url"):
            return True
        if not isinstance(file.get("file_data"), str):
            return True
    if kind in {"image", "document"}:
        source = value.get("source")
        if not isinstance(source, dict) or source.get("type") not in {
            "base64",
            "text",
            "content",
        }:
            return True
    return any(has_external_media(item) for item in value.values())


class ModelAPIServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class ModelAPIHandler(BaseHTTPRequestHandler):
    # No access logs: inference URLs, headers and bodies can contain credentials.
    def log_message(self, *args):
        pass

    def do_CONNECT(self):
        self.send_error(405)

    def do_GET(self):
        self.send_error(405)

    def do_POST(self):
        upstream = self.server.upstream
        path = urlsplit(self.path)
        route = path.path
        if route.startswith("/v1/"):
            route = route[3:]
        if (
            path.scheme
            or path.netloc
            or path.query
            or path.fragment
            or route
            not in {
                "/responses",
                "/responses/compact",
                "/chat/completions",
                "/messages",
                "/messages/count_tokens",
            }
        ):
            self.send_error(403)
            return
        if self.headers.get("Transfer-Encoding"):
            self.send_error(400)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64 * 1024 * 1024:
                raise ValueError()
            body = self.rfile.read(length)
            payload = json.loads(body)
            # Only client-executed tools may cross the offline boundary.
            # Unknown provider tools fail closed rather than extending a blacklist.
            if not isinstance(payload, dict):
                raise TypeError()
            if payload.get("model") != self.server.expected_model:
                self.send_error(403)
                return
            if any(
                has_external_media(payload.get(key)) for key in ("input", "messages")
            ):
                self.send_error(403)
                return
            tools = payload.get("tools", [])
            if not isinstance(tools, list):
                raise TypeError()
            if route in {"/messages", "/messages/count_tokens"}:
                allowed_types = {"custom"}
                default_type = "custom"
            elif route == "/chat/completions":
                allowed_types = {"function"}
                default_type = None
            else:
                allowed_types = {"function", "custom", "local_shell"}
                default_type = None
            pending_tools = list(tools)
            while pending_tools:
                tool = pending_tools.pop()
                if not isinstance(tool, dict):
                    raise TypeError()
                kind = tool.get("type", default_type)
                if kind == "namespace" and route in {
                    "/responses",
                    "/responses/compact",
                }:
                    nested = tool.get("tools")
                    if not isinstance(nested, list):
                        raise TypeError()
                    pending_tools.extend(nested)
                    continue
                if not isinstance(kind, str) or kind not in allowed_types:
                    self.send_error(403)
                    return
            if payload.get("mcp_servers"):
                self.send_error(403)
                return
        except (ValueError, TypeError, AttributeError):
            self.send_error(400)
            return
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower()
            in {
                "authorization",
                "x-api-key",
                "content-type",
                "accept",
                "anthropic-version",
                "anthropic-beta",
                "openai-beta",
                "openai-organization",
                "openai-project",
            }
        }
        headers["Accept-Encoding"] = "identity"
        connection_type = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(upstream.hostname, upstream.port, timeout=600)
        started = False
        try:
            connection.request("POST", upstream.path.rstrip("/") + route, body, headers)
            response = connection.getresponse()
            # Never follow or expose redirects to an unrestricted destination.
            if 300 <= response.status < 400:
                self.send_error(502)
                return
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() in {
                    "content-type",
                    "content-encoding",
                    "retry-after",
                    "x-request-id",
                    "request-id",
                }:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            started = True
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not started:
                self.send_error(502)
        finally:
            connection.close()


@contextmanager
def model_api_proxy(base_url, *, expected_model):
    """Forward inference for one frozen model id.

    The model check keeps every run inside its manifest entry: an agent that
    asks for a different deployment fails closed instead of silently changing
    models mid-experiment.
    """
    if not isinstance(expected_model, str) or not expected_model:
        raise ValueError("Model API proxy requires the frozen model id")
    upstream = urlsplit(base_url)
    if (
        upstream.scheme not in {"https", "http"}
        or not upstream.hostname
        or upstream.username
        or upstream.password
        or upstream.query
        or upstream.fragment
    ):
        raise ValueError(
            "Model API upstream must be an HTTP(S) base URL without credentials, query or fragment"
        )
    with tempfile.TemporaryDirectory(prefix="patcheval-model-api-") as directory:
        socket_path = Path(directory) / "api.sock"
        server = ModelAPIServer(str(socket_path), ModelAPIHandler)
        server.upstream = upstream
        server.expected_model = expected_model
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield Path(directory)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
