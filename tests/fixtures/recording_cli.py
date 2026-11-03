#!/usr/bin/python3
"""Stand in for a CLI inside bubblewrap and record its actual invocation."""

import json
import os
import sys
from pathlib import Path


def main():
    config = Path("/tmp/opencode-config/opencode/opencode.json")
    call = {
        "argv": sys.argv[1:],
        "env": dict(os.environ),
        "stdin": sys.stdin.read(),
        "config": json.loads(config.read_text()) if config.exists() else None,
    }
    with Path("/results/calls.jsonl").open("a") as handle:
        handle.write(json.dumps(call) + "\n")

    agent = Path(sys.argv[0]).name
    if agent == "opencode":
        if sys.argv[1] == "run":
            Path(os.environ["OPENCODE_DB"]).write_text("database-sentinel")
            print(json.dumps({"sessionID": "session-sentinel"}, separators=(",", ":")))
        elif sys.argv[1] == "export":
            print(json.dumps({"info": {"id": "session-sentinel"}, "messages": []}))
    elif agent in {"codex", "traex"}:
        home = "CODEX_HOME" if agent == "codex" else "TRAE_HOME"
        sessions = Path(os.environ[home])
        if agent == "traex":
            sessions /= "cli"
        sessions /= "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        observation = {
            "type": "token_usage_record",
            "payload": {
                "response_id": "response-sentinel",
                "usage": {"input_tokens": 7},
            },
        }
        (sessions / "rollout-sentinel.jsonl").write_text(json.dumps(observation) + "\n")


if __name__ == "__main__":
    main()
