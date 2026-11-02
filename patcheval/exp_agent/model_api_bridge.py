"""Loopback-to-Unix-socket bridge. Requires Python 3.5+."""

import json
import os
import select
import socket
import socketserver
import sys
import time


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect("/opt/model-api/api.sock")
            peers = {self.request: upstream, upstream: self.request}
            while True:
                readable, _, _ = select.select(list(peers), [], [], 600)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    peers[source].sendall(data)
        except OSError:
            return
        finally:
            upstream.close()


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    if sys.argv[1:] == ["--wait"]:
        for attempt in range(100):
            try:
                connection = socket.create_connection(("127.0.0.1", 18080), 0.1)
                connection.close()
                return
            except OSError:
                time.sleep(0.1)
        raise SystemExit("Model API bridge did not start")
    if sys.argv[1:] == ["--configure-opencode"]:
        path = "/tmp/opencode-config/opencode/opencode.json"
        with open(path) as source:
            config = json.load(source)
        provider = os.environ["PATCHAGENT_OPENCODE_PROVIDER"]
        if config.get("model", "").split("/")[0] != provider:
            raise SystemExit("OpenCode model must select PATCHAGENT_OPENCODE_PROVIDER")
        small = config.get("small_model")
        if small is not None and small.split("/")[0] != provider:
            raise SystemExit("OpenCode small_model must use PATCHAGENT_OPENCODE_PROVIDER")
        # Require the selected provider to exist rather than inventing a config.
        config["provider"][provider].setdefault("options", {})["baseURL"] = os.environ[
            "PATCHAGENT_MODEL_API_BASE_URL"
        ]
        with open(path, "w") as destination:
            json.dump(config, destination)
        return
    server = Server(("127.0.0.1", 18080), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
