#!/usr/bin/env python3
"""Generate patches with a CLI agent and a configured model API."""

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", choices=["codex", "opencode", "traecli"], required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-provider",
        required=True,
        help="Vendor label recorded as trajectory provenance, e.g. deepseek",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--reasoning-effort",
        required=True,
        help="Requested reasoning effort; Trae cannot currently pin this setting",
    )
    parser.add_argument("--cve", help="Select one exact CVE")
    parser.add_argument(
        "--limit", type=int, default=-1, help="Maximum cases; default: all"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Parallel cases (default: 4)"
    )
    args = parser.parse_args()
    if not args.reasoning_effort.strip():
        parser.error("--reasoning-effort must not be empty")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    executable = "traex" if args.agent == "traecli" else args.agent
    binary = shutil.which(executable) or str(Path.home() / ".local/bin" / executable)
    if not Path(binary).is_file():
        parser.error(f"Install {executable} first")
    key = (
        os.environ.get("MODEL_API_KEY") or getpass.getpass("Model API key: ")
    ).strip()
    if not key:
        parser.error("Model API key is empty")
    directory = Path(__file__).resolve().parent
    label = args.agent + "-" + uuid.uuid4().hex[:10]
    # Keep credentials out of command arguments, run metadata and durable configs.
    with tempfile.TemporaryDirectory(prefix="patcheval-model-") as temporary:
        home = Path(temporary)
        environment = dict(os.environ)
        environment.update(
            MODEL_API_UPSTREAM=args.base_url,
            MODEL_API_MODEL=args.model,
            MODEL_API_PROVIDER=args.model_provider,
            LIMIT=str(args.limit),
            CONCURRENCY=str(args.concurrency),
            CVE=args.cve or "",
        )
        if args.agent == "opencode":
            config = home / "config/opencode/opencode.json"
            config.parent.mkdir(parents=True)
            (home / "data").mkdir()
            # "@ai-sdk/openai" is the Responses transport. "small_model"
            # and "reasoningEffort" are pinned so auxiliary requests and the
            # default "medium" effort cannot slip into a frozen run.
            content = json.dumps(
                {
                    "model": "model-api/" + args.model,
                    "small_model": "model-api/" + args.model,
                    "provider": {
                        "model-api": {
                            "npm": "@ai-sdk/openai",
                            "name": "Model API",
                            "options": {"baseURL": args.base_url, "apiKey": key},
                            "models": {
                                args.model: {
                                    "name": args.model,
                                    "options": {
                                        "reasoningEffort": args.reasoning_effort
                                    },
                                }
                            },
                        }
                    },
                }
            )
            public_config = json.loads(content)
            del public_config["provider"]["model-api"]["options"]["apiKey"]
            environment.update(
                OPENCODE_BIN=binary,
                OPENCODE_CONFIG=str(config),
                OPENCODE_PROVIDER="model-api",
            )
        else:
            suffix = "traecli.toml" if args.agent == "traecli" else "config.toml"
            config = home / ("model-api." + suffix)
            # traecli 0.213.1 accepts model_reasoning_effort but still sends
            # "reasoning": null upstream, so writing it would imply an effort
            # pin that does not exist. Trae is not part of the frozen matrix.
            effort = (
                ""
                if args.agent == "traecli"
                else f"model_reasoning_effort = {json.dumps(args.reasoning_effort)}\n"
            )
            content = (
                f"model = {json.dumps(args.model)}\n"
                'model_provider = "model-api"\n'
                + effort
                + "[model_providers.model-api]\n"
                'name = "Model API"\n'
                f"base_url = {json.dumps(args.base_url)}\n"
                'wire_api = "responses"\n'
                f"experimental_bearer_token = {json.dumps(key)}\n"
            )
            public_config = {
                "model": args.model,
                "model_provider": "model-api",
                "model_providers": {
                    "model-api": {
                        "name": "Model API",
                        "base_url": args.base_url,
                        "wire_api": "responses",
                    }
                },
            }
            if args.agent == "codex":
                public_config["model_reasoning_effort"] = args.reasoning_effort
            prefix = "TRAE" if args.agent == "traecli" else "CODEX"
            environment.update(
                {
                    prefix + "_BIN": binary,
                    prefix + "_CONFIG": str(config),
                    prefix + "_MODEL_PROVIDER": "model-api",
                }
            )
        config.touch(mode=0o600)
        config.write_text(content)
        snapshot = home / "launch.json"
        snapshot.write_text(
            json.dumps(
                {
                    "source": "run_model.py",
                    "config": public_config,
                    "reasoning_effort": {
                        "requested": args.reasoning_effort,
                        "configured": (
                            None if args.agent == "traecli" else args.reasoning_effort
                        ),
                    },
                }
            )
        )
        environment["PATCHAGENT_LAUNCH_CONFIG"] = str(snapshot)
        if args.agent == "traecli":
            print(
                "Warning: Trae reasoning effort is not pinned; "
                f"requested {args.reasoning_effort!r}, configured null. "
                "This run does not establish equal reasoning effort across agents.",
                file=sys.stderr,
                flush=True,
            )
        result = subprocess.run(
            ["bash", str(directory / "run_infer.sh"), args.agent, label],
            env=environment,
            check=False,
        )
    if result.returncode == 0:
        print(f"Evaluate this run: bash {directory / 'run_eval.sh'} {label}")
    else:
        print(
            "Generation failed; inspect the run's .work/*/agent_stdout.txt and agent_stderr.txt before evaluation."
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
