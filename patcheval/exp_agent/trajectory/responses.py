"""Project response items shared by the observed Codex and Trae rollout formats."""

import json


def response_message(items):
    """Keep one completion's text, reasoning and tool requests in one response."""
    blocks = []
    calls = []
    for item in items:
        kind = item.get("type")
        if kind == "message":
            blocks.extend(item.get("content") or [])
        else:
            blocks.append(dict(item))
        if kind not in {"function_call", "custom_tool_call"}:
            continue
        arguments = item.get("arguments")
        if kind == "function_call" and isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"arguments": arguments}
        elif kind == "custom_tool_call":
            arguments = {"input": item.get("input")}
        calls.append(
            {
                "id": item.get("call_id"),
                "name": item.get("name"),
                "args": arguments,
                "type": "tool_call",
            }
        )
    return {
        "role": "assistant",
        "content": blocks,
        "tool_calls": calls,
        "response_metadata": {"response_items": items},
    }


def context_message(item):
    """Separate actual inputs and tool outputs from a model's response items."""
    kind = item.get("type")
    role = item.get("role")
    if role in {"system", "developer", "user"} or kind == "agent_message":
        if kind == "agent_message":
            role = "user"
        elif role == "developer":
            role = "system"
        return {
            "id": item.get("id"),
            "role": role,
            "content": item.get("content") or ([item] if kind != "message" else []),
            "response_metadata": {"response_item": item},
        }
    if kind in {"function_call_output", "custom_tool_call_output"}:
        output = item.get("output", "")
        return {
            "id": item.get("id"),
            "role": "tool",
            "tool_call_id": item.get("call_id"),
            "content": (
                output if isinstance(output, (str, list)) else json.dumps(output)
            ),
            "response_metadata": {"response_item": item},
        }
    return None
