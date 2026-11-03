#!/usr/bin/env python3
"""Generate patches for PatchEval cases with a configurable CLI agent.

This is a patch-generation-only runner. Evaluation is handled separately by
../evaluation/run_evaluation.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from model_api_proxy import model_api_proxy
from trajectory import write_trajectory

DEFAULT_AGENT_TIMEOUT_S = 3600
STREAM_READER_LIMIT = 16 * 1024 * 1024


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


@dataclass
class GenerationResult:
    index: int
    cve: str
    instance_id: str
    image: str
    workdir: str
    container_name: str
    status: str
    patch_generated: bool
    agent_exit_code: Optional[int]
    timed_out: bool
    duration_s: float
    patch_path: str
    error: str = ""
    runtime: dict[str, Any] | None = None


def _safe_name(value: str, max_len: int = 100) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return (value.strip(".-") or "sample")[:max_len]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _require_dataset_images(samples: list[dict[str, Any]]) -> None:
    missing: list[str] = []
    for sample in samples:
        cve = str(sample["cve_id"])
        if not sample.get("image_url"):
            missing.append(cve)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(f"dataset samples missing image_url: {preview}{suffix}")


def _image_url(sample: dict[str, Any]) -> str:
    image = str(sample.get("image_url") or "").strip()
    if not image:
        raise ValueError(f"dataset sample {sample.get('cve_id', '<unknown>')} missing image_url")
    return image


def _repo_basename(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def _prompt(sample: dict[str, Any], workdir_hint: str) -> str:
    return (
        "## USER\n\n"
        "Please fix the vulnerabilities in the code repository based on the following information:\n\n"
        + str(sample.get("cve_description") or "").strip()
        + "\n\n"
        + "Task runtime information:\n"
        + "- Target workdir: "
        + workdir_hint
        + "\n"
        + "- All tool path arguments must stay under this directory.\n"
        + "- Start exploration from this workspace root instead of guessing a path under /workspace.\n"
        + "- Before stopping, write the final repository diff to `/workspace/fix.patch` from the target workdir."
        + "\n\n"
        + "Repair-source restrictions:\n"
        + "- Do not search the web for this vulnerability, CVE, advisory, GHSA, release note, issue, pull request, or upstream patch.\n"
        + "- Do not run network commands such as curl, wget, git fetch, git pull, git ls-remote, npm view, pip index, or package/advisory lookups to find the fix.\n"
    )


async def _run(args: list[str], *, timeout_s: Optional[int] = None) -> CommandResult:
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=STREAM_READER_LIMIT,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        timed_out = False
    except asyncio.TimeoutError:
        proc.kill()
        out_b, err_b = await proc.communicate()
        timed_out = True
    return CommandResult(
        command=args,
        exit_code=int(proc.returncode if proc.returncode is not None else 124),
        stdout=(out_b or b"").decode("utf-8", errors="replace"),
        stderr=(err_b or b"").decode("utf-8", errors="replace"),
        duration_s=time.monotonic() - started,
        timed_out=timed_out,
    )


async def _docker_exec(container: str, command: str, *, workdir: str = "/", timeout_s: int = 600) -> CommandResult:
    args = ["docker", "exec", "-w", workdir, container, "bash", "-lc", command]
    return await _run(args, timeout_s=timeout_s)


def _parse_mounts(items: list[str]) -> list[tuple[str, str, str]]:
    mounts = []
    for item in items:
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"expected HOST:CONTAINER[:ro|rw], got: {item}")
        mode = parts[2] if len(parts) == 3 else "rw"
        if mode not in {"ro", "rw"}:
            raise ValueError(f"invalid mount mode: {item}")
        mounts.append((str(Path(parts[0]).expanduser().resolve()), parts[1], mode))
    return mounts


def isolated_agent_mounts(mounts, destination):
    """Expose configuration/auth and runtime dependencies, never host histories."""
    cleaned = []
    for index, (host, target, mode) in enumerate(mounts):
        source = Path(host)
        if target not in {
            "/opt/codex-home-src",
            "/opt/trae-home-src",
            "/opt/opencode-config-src",
            "/opt/opencode-data-src",
        }:
            cleaned.append((host, target, mode))
            continue
        seed = destination / str(index)
        seed.mkdir()
        if target == "/opt/opencode-data-src":
            allowed = [source / "opencode/auth.json", source / "opencode/bin"]
        elif target == "/opt/opencode-config-src":
            allowed = [source / "opencode/opencode.json"]
        else:
            allowed = [
                source / name for name in ("config.toml", "traecli.toml", "auth.json")
            ]
            allowed += list(source.glob("*.config.toml")) + list(
                source.glob("*.traecli.toml")
            )
        for item in allowed:
            if item.exists():
                output = seed / item.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                if item.is_dir():
                    shutil.copytree(item, output)
                else:
                    shutil.copyfile(item, output)
        cleaned.append((str(seed), target, "ro"))
    return cleaned


def _docker_run_args(container: str, image: str, result_dir: Path, mounts: list[tuple[str, str, str]]) -> list[str]:
    cmd = ["docker", "run", "-d", "--network", "none", "--name", container, "-v", f"{result_dir.resolve()}:/results:rw"]
    for host, dst, mode in mounts:
        cmd.extend(["-v", f"{host}:{dst}:{mode}"])
    cmd.extend([image, "bash", "-lc", "tail -f /dev/null"])
    return cmd


async def _detect_workdir(container: str, sample: dict[str, Any]) -> str:
    workdir_hint = str(sample.get("workdir") or "").rstrip("/")
    if workdir_hint:
        check = await _docker_exec(container, f"test -d {shlex.quote(workdir_hint)}/.git", timeout_s=60)
        if check.exit_code == 0:
            return workdir_hint
        raise RuntimeError(f"dataset workdir is not a git repository in image: {workdir_hint}")

    repo_name = _repo_basename(str(sample.get("repo") or ""))
    if repo_name:
        check = await _docker_exec(container, f"test -d /workspace/{shlex.quote(repo_name)}/.git", timeout_s=60)
        if check.exit_code == 0:
            return f"/workspace/{repo_name}"
        lower_repo_name = repo_name.lower()
        if lower_repo_name != repo_name:
            check = await _docker_exec(container, f"test -d /workspace/{shlex.quote(lower_repo_name)}/.git", timeout_s=60)
            if check.exit_code == 0:
                return f"/workspace/{lower_repo_name}"

    check = await _docker_exec(container, "test -d /workspace/.git", timeout_s=60)
    if check.exit_code == 0:
        return "/workspace"

    if repo_name:
        raise RuntimeError(
            f"could not locate git workdir for repo {repo_name!r}; "
            "check that dataset image_url matches this CVE or add dataset workdir"
        )

    find_repo = await _docker_exec(
        container,
        "find /workspace -mindepth 2 -maxdepth 3 -type d -name .git 2>/dev/null | head -n 2",
        timeout_s=60,
    )
    candidates = [line.rsplit("/.git", 1)[0] for line in find_repo.stdout.splitlines() if line.strip()]
    if len(candidates) == 1:
        return candidates[0]

    return "/workspace"


async def _hide_workspace_payload(container: str, workdir: str, session_key: str) -> None:
    if workdir.rstrip("/") == "/workspace":
        result = await _docker_exec(container, "rm -f /workspace/fix.patch", timeout_s=300)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or result.stdout)
        return

    if not workdir.startswith("/workspace/"):
        raise RuntimeError(f"target workdir is outside /workspace: {workdir}")

    script = f"""
set -e
rm -f /workspace/fix.patch
workdir={shlex.quote(workdir)}
top_name=${{workdir#/workspace/}}
top_name=${{top_name%%/*}}
find /workspace -mindepth 1 -maxdepth 1 ! -name "$top_name" -exec rm -rf -- {{}} + 2>/dev/null || true
"""
    result = await _docker_exec(container, script, timeout_s=300)
    if result.exit_code != 0:
        raise RuntimeError(result.stderr or result.stdout)


async def _run_agent(
    container: str,
    workdir: str,
    command_template: str,
    result_dir: Path,
    env: dict[str, str],
    timeout_s: int,
) -> CommandResult:
    env = dict(
        env,
        HOME="/tmp/patchagent-home",
        XDG_CONFIG_HOME="/tmp/patchagent-home/config",
        XDG_DATA_HOME="/tmp/patchagent-home/data",
        XDG_STATE_HOME="/tmp/patchagent-home/state",
        XDG_CACHE_HOME="/tmp/patchagent-home/cache",
        PATCHAGENT_WORKDIR=workdir,
        PATCHAGENT_PROMPT_FILE="/results/prompt.txt",
    )
    bridge = (
        "mkdir -p /tmp/patchagent-home; "
        "python3 /opt/model-api-bridge.py >/tmp/model-api-bridge.log 2>&1 & "
        "bridge_pid=$!; trap 'kill $bridge_pid 2>/dev/null || true' EXIT; "
        "python3 /opt/model-api-bridge.py --wait || { cat /tmp/model-api-bridge.log >&2; exit 1; }; "
    )
    command = bridge + command_template
    started = time.monotonic()
    stdout_path = result_dir / "agent_stdout.txt"
    stderr_path = result_dir / "agent_stderr.txt"
    # Persist output as it arrives: cancelling communicate() loses bytes it has
    # already consumed, precisely when a timed-out case needs diagnostics.
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-w",
            workdir,
            *sum((["-e", f"{k}={v}"] for k, v in env.items()), []),
            container,
            "bash",
            "-lc",
            command,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            timed_out = True
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    return CommandResult(
        ["docker", "exec", container, "bash", "-lc", command],
        int(proc.returncode if proc.returncode is not None else 124),
        stdout,
        stderr,
        time.monotonic() - started,
        timed_out,
    )


async def _collect_patch(container: str, workdir: str, result_dir: Path) -> CommandResult:
    script = f"""
set -e
cd {shlex.quote(workdir)}
if [ -s /workspace/fix.patch ]; then
  cp /workspace/fix.patch /results/llm.patch
elif git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff HEAD -U3 > /results/llm.patch
else
  echo "target workdir is not a git repository: $(pwd)" >&2
  exit 1
fi
test -s /results/llm.patch
"""
    return await _docker_exec(container, script, workdir=workdir, timeout_s=300)


async def _remove_container(container: str) -> None:
    await _run(["docker", "rm", "-f", container], timeout_s=120)


async def _capture_runtime(container: str, agent: str) -> dict[str, Any]:
    """Record the running image and mounted CLI, not a mutable tag or host CLI.

    Failed probes remain explicit unknowns without discarding a usable patch.
    Configured reasoning effort is recorded separately; these probes do not
    establish what the upstream model actually honored.
    """
    runtime = {"image_id": None, "cli_version": None, "capture_errors": []}
    try:
        image = await _run(
            ["docker", "container", "inspect", "--format", "{{.Image}}", container],
            timeout_s=30,
        )
        if image.exit_code == 0 and re.fullmatch(
            r"sha256:[a-f0-9]{64}", image.stdout.strip()
        ):
            runtime["image_id"] = image.stdout.strip()
    except OSError:
        pass
    if runtime["image_id"] is None:
        runtime["capture_errors"].append("image_id_unavailable")
    executable = "traex" if agent == "traecli" else agent
    try:
        version = await _docker_exec(
            container, f"/usr/local/bin/{executable} --version", timeout_s=30
        )
        match = re.search(r"\b\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b", version.stdout)
        if version.exit_code == 0 and match:
            runtime["cli_version"] = match.group()
    except OSError:
        pass
    if runtime["cli_version"] is None:
        runtime["capture_errors"].append("cli_version_unavailable")
    return runtime


async def _run_one(sample: dict[str, Any], index: int, args: argparse.Namespace, sem: asyncio.Semaphore, mounts: list[tuple[str, str, str]]) -> GenerationResult:
    async with sem:
        started = time.monotonic()
        cve = str(sample["cve_id"])
        instance_id = f"patcheval_{cve}"
        image = _image_url(sample)
        run_id = f"{index:05d}-{instance_id}"
        container = f"{args.container_prefix}-{_safe_name(run_id)}-{os.getpid()}"
        run_root = Path(args.run_root)
        work = run_root / ".work" / run_id
        work.mkdir(parents=True, exist_ok=True)
        patch_path = run_root / "patches" / f"{cve}.patch"
        status = "failed"
        error = ""
        workdir = "/workspace"
        agent_result: Optional[CommandResult] = None
        timed_out = False
        runtime = None
        try:
            env = {"PATCHAGENT_SESSION_ID": container, "PATCHAGENT_MODEL_API_BASE_URL": "http://127.0.0.1:18080/v1"}
            if args.opencode_provider:
                env["PATCHAGENT_OPENCODE_PROVIDER"] = args.opencode_provider
            run_result = await _run(_docker_run_args(container, image, work, mounts), timeout_s=1200)
            if run_result.exit_code != 0:
                raise RuntimeError(f"docker run failed: {run_result.stderr or run_result.stdout}")
            runtime = await _capture_runtime(container, args.agent)
            workdir = await _detect_workdir(container, sample)
            await _hide_workspace_payload(container, workdir, container)
            prompt = _prompt(sample, workdir)
            (work / "prompt.txt").write_text(prompt, encoding="utf-8")
            agent_result = await _run_agent(container, workdir, args.agent_command, work, env, args.agent_timeout)
            timed_out = agent_result.timed_out
            if agent_result.exit_code != 0:
                raise RuntimeError(f"agent failed with exit_code={agent_result.exit_code}")
            collect = await _collect_patch(container, workdir, work)
            if collect.exit_code != 0:
                raise RuntimeError(f"collect patch failed: {collect.stderr or collect.stdout}")
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "llm.patch", patch_path)
            status = "generated"
        except Exception as exc:
            error = str(exc)
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text("", encoding="utf-8")
            _log(f"{run_id}: failed: {error}")
        finally:
            await _remove_container(container)
        try:
            write_trajectory(
                work,
                run_root / "trajectories",
                cve,
                agent_kind=args.agent,
                model_provider=args.model_provider,
                model_name=args.expect_model,
            )
        except Exception as exc:
            _log(f"{run_id}: trajectory export failed: {exc}")
        result = GenerationResult(index, cve, instance_id, image, workdir, container, status, status == "generated", agent_result.exit_code if agent_result else None, timed_out, time.monotonic() - started, str(patch_path), error)
        result.runtime = runtime
        return result


async def _generate(args: argparse.Namespace) -> int:
    samples = _read_json(Path(args.input))
    indexed = list(enumerate(samples))
    if args.cve:
        indexed = [(index, sample) for index, sample in indexed if sample["cve_id"] == args.cve]
        if not indexed:
            raise ValueError(f"CVE not found in dataset: {args.cve}")
    selected = indexed if args.limit < 0 else indexed[: args.limit]
    _require_dataset_images([sample for _, sample in selected])
    _log("Selected cases: " + ", ".join(sample["cve_id"] for _, sample in selected))
    run_root = (
        Path(args.output_dir)
        / f"{time.strftime('%Y%m%d_%H%M%S')}-{_safe_name(args.run_label or 'run')}"
    )
    for sub in ["patches", ".work"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    args.run_root = str(run_root)
    _write_json(
        run_root / "run_metadata.json",
        vars(args)
        | {
            "run_root": str(run_root),
            "total_cases": len(selected),
            "network_mode": "none",
            "selected_cves": [sample["cve_id"] for _, sample in selected],
            "host_state": "config_auth_runtime_only",
            "cross_case_memory": "disabled",
            "model_api_transport": "unix_socket",
            "dataset_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            "launch_config": (
                _read_json(Path(args.launch_config)) if args.launch_config else None
            ),
        },
    )
    mounts = isolated_agent_mounts(_parse_mounts(args.mount), Path(args.seed_directory))
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(_run_one(sample, idx, args, sem, mounts))
        for idx, sample in selected
    ]
    results_path = run_root / "results.jsonl"
    results = []
    with results_path.open("w", encoding="utf-8") as f:
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            results.append(result)
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            f.flush()
            _log(f"progress {i}/{len(tasks)}: {result.cve} {result.status}")
    generated = sum(r.patch_generated for r in results)
    _write_json(
        run_root / "summary.json",
        {
            "total": len(results),
            "generated": generated,
            "failed": len(results) - generated,
        },
    )
    print(f"Run directory: {run_root}")
    print(f"Generated patches: {generated}/{len(results)}")
    return 0 if generated == len(results) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="../datasets/patcheval_verified_cleaned.json")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--cve", help="Select one exact CVE instead of the dataset prefix")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--run-label", default="")
    p.add_argument("--agent", choices=["codex", "opencode", "traecli"], default="codex")
    p.add_argument("--agent-command", required=True)
    p.add_argument(
        "--model-api-upstream",
        required=True,
        help="Fixed model API base URL, e.g. https://provider.example/v1",
    )
    p.add_argument(
        "--expect-model",
        required=True,
        help="Exact model id every inference request must use",
    )
    p.add_argument(
        "--model-provider",
        help="Vendor label recorded as trajectory provenance, e.g. deepseek",
    )
    p.add_argument("--mount", action="append", default=[])
    p.add_argument(
        "--opencode-provider", help="Provider key selected by the OpenCode adapter"
    )
    p.add_argument("--agent-timeout", type=int, default=DEFAULT_AGENT_TIMEOUT_S)
    p.add_argument("--container-prefix", default="patcheval-agent")
    p.add_argument(
        "--launch-config",
        help="Credential-free configuration snapshot supplied by run_model.py",
    )
    return p


async def _main(args: argparse.Namespace) -> int:
    with model_api_proxy(
        args.model_api_upstream, expected_model=args.expect_model
    ) as directory, tempfile.TemporaryDirectory(
        prefix="patcheval-agent-seeds-"
    ) as seeds:
        args.seed_directory = seeds
        args.mount.extend(
            [
                f"{directory}:/opt/model-api:ro",
                f"{Path(__file__).with_name('model_api_bridge.py').resolve()}:/opt/model-api-bridge.py:ro",
            ]
        )
        return await _generate(args)


def main() -> int:
    return asyncio.run(_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
