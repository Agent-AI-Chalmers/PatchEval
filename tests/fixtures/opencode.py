"""Independent opencode samples and capture builders for PatchEval tests."""

import json
import sqlite3

OPENCODE_SESSION = {
    "info": {
        "id": "ses_1",
        "cost": 0.01,
        "tokens": {
            "input": 10,
            "output": 5,
            "reasoning": 1,
            "cache": {"read": 2, "write": 0},
        },
    },
    "messages": [
        {
            "info": {"role": "user", "id": "msg_1"},
            "parts": [{"type": "text", "text": "hi"}],
        },
        {
            "info": {
                "role": "assistant",
                "id": "msg_2",
                "tokens": {
                    "input": 10,
                    "output": 5,
                    "reasoning": 1,
                    "cache": {"read": 2, "write": 0},
                },
            },
            "parts": [
                {"type": "reasoning", "text": "thinking"},
                {"type": "text", "text": "done"},
                {
                    "type": "tool",
                    "tool": "bash",
                    "callID": "call_1",
                    "state": {
                        "status": "completed",
                        "input": {"command": "ls"},
                        "output": "a.txt",
                        "metadata": {
                            "diff": "Index: a.txt\n--- a.txt\n+++ b.txt\n@@ -1 +1 @@\n-a\n+b\n",
                            "truncated": False,
                        },
                    },
                },
            ],
        },
    ],
}


def write_opencode_database(work, sessions):
    """Keep a live WAL and deliberately different session totals for accounting tests.

    The caller closes the returned connection after reading the capture. The
    session counters differ from response usage so tests detect accidental
    aggregation of both observation scopes.
    """

    db = sqlite3.connect(work / "opencode.db")
    db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE session (
            id TEXT PRIMARY KEY, parent_id TEXT, time_created INTEGER,
            cost REAL, tokens_input INTEGER, tokens_output INTEGER,
            tokens_reasoning INTEGER, tokens_cache_read INTEGER, tokens_cache_write INTEGER
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
            time_created INTEGER, data TEXT
        );
        """)
    for index, session in enumerate(sessions):
        identity = session["info"]["id"]
        db.execute(
            "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (identity, session["info"].get("parentID"), index, 0, 999, 999, 0, 0, 0),
        )
        for position, entry in enumerate(session.get("messages", [])):
            meta = entry["info"]
            db.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?)",
                (meta["id"], identity, position, json.dumps(meta)),
            )
            for part_index, part in enumerate(entry.get("parts", [])):
                db.execute(
                    "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                    (
                        f"{meta['id']}-part-{part_index}",
                        meta["id"],
                        identity,
                        part_index,
                        json.dumps(part),
                    ),
                )
    db.commit()
    return db


def opencode_database_session(identity, parent=None):
    return {
        "info": {"id": identity, "parentID": parent},
        "messages": [
            {
                "info": {
                    "id": f"{identity}-response",
                    "role": "assistant",
                    "time": {"created": 1, "completed": 2},
                    "tokens": {
                        "total": 25,
                        "input": 18,
                        "output": 4,
                        "reasoning": 1,
                        "cache": {"read": 2, "write": 0},
                    },
                },
                "parts": [{"type": "text", "text": f"answer from {identity}"}],
            }
        ],
    }
