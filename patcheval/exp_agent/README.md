# PatchEval CLI Agent Patch Generation

This directory contains a lightweight patch-generation workflow for CLI coding agents. It is designed to plug into the PatchEval-Verified evaluation flow:

```text
run_infer.sh + patch_agent_runner.py
  -> generate patches
  -> process_data.py
  -> ../evaluation/run_evaluation.py
  -> evaluate generated patches
```

The supported CLI agents share the same runner and differ only in their command and mounts. The shared runner handles Docker setup, prompt generation, patch collection, and output layout.

## Quickstart

Run one smoke case with Codex, then evaluate the generated patch:

```bash
conda activate patcheval
cd patcheval/exp_agent

export CODEX_BIN=/path/to/bin/codex
export CODEX_CONFIG=/path/to/codex-home/my-profile.config.toml
export CODEX_MODEL_PROVIDER=my_provider
export MODEL_API_UPSTREAM=https://provider.example/v1
export MODEL_API_MODEL=<model-id-sent-upstream>
export MODEL_API_PROVIDER=<vendor-label>

LIMIT=1 CONCURRENCY=1 bash run_infer.sh codex codex_smoke
bash run_eval.sh codex_smoke
```

For OpenCode, set the corresponding runtime variables and replace the agent name:

```bash
# OpenCode
conda activate patcheval
cd patcheval/exp_agent

export OPENCODE_BIN=/path/to/bin/opencode
export OPENCODE_CONFIG=/path/to/opencode-home/config/opencode/opencode.json
export OPENCODE_PROVIDER=my_provider
export MODEL_API_UPSTREAM=https://provider.example/v1
export MODEL_API_MODEL=<model-id-sent-upstream>
export MODEL_API_PROVIDER=<vendor-label>
LIMIT=1 CONCURRENCY=1 bash run_infer.sh opencode opencode_smoke
bash run_eval.sh opencode_smoke
```

Generated patches are written to `agent_runs/<timestamp>-<prefix>/patches/`. Evaluation results are written to `../evaluation/evaluation_output/results/<prefix>/`.

## Model API quick start

From the repository root, select an agent, model and API base URL:

```bash
python patcheval/exp_agent/run_model.py \
  --agent traecli \
  --model MODEL \
  --model-provider VENDOR \
  --base-url https://provider.example/v1 \
  --reasoning-effort low
```

The command prompts for the API key without displaying it. For noninteractive runs, supply `MODEL_API_KEY` in the environment. By default, it processes the entire dataset with four parallel cases and prints the evaluation command afterward. Use `--concurrency` to change parallelism, `--limit N` for the first N cases, or `--cve CVE-2018-7753` for one exact case. The selected CVEs are printed and recorded.

`--agent` also accepts `codex` and `opencode`. Every adapter uses the Responses transport: Codex and Trae select it through `wire_api = "responses"`, OpenCode through the pinned `@ai-sdk/openai` provider. `--reasoning-effort` is required and is written into the Codex profile and the OpenCode model options; OpenCode's `small_model` is pinned to the same deployment so auxiliary requests cannot reach another model. Provider dependencies must be cached. Existing profile-based commands below remain available for advanced settings.

## Layout

```text
exp_agent/
├── README.md
├── agents/
│   ├── codex.sh       # Codex CLI adapter (mounts scripts/codex-run.sh)
│   ├── opencode.sh    # OpenCode adapter (mounts scripts/opencode-run.sh)
│   ├── traecli.sh     # Trae CLI adapter (mounts scripts/trae-run.sh)
│   └── scripts/       # per-agent container entrypoints, mounted at /opt/agent-run.sh
├── patch_agent_runner.py
├── process_data.py
├── run_infer.sh
├── run_eval.sh
└── trajectory.py      # CLI event streams -> trajectory.jsonl + metrics.json
```

## Data and images

<!-- doc-default: DATASET; source=patcheval/exp_agent/run_infer.sh -->
By default, scripts use:

```text
../datasets/patcheval_verified_cleaned.json
```

`patcheval_verified_cleaned.json` provides CVE metadata, prompt content, and the `image_url` used for agent execution and evaluation. `../../scripts/images.txt` contains the same 230 image references for bulk download.

`patcheval_verified.json` is retained as the pre-review snapshot for reproducing older runs. The current pipeline does not read it, and it must not be deleted.

Four description fragments destroyed by markdown stripping were restored in this file in place; see [DATASET_NOTES.md](../datasets/DATASET_NOTES.md) for the per-entry diff, the audit of the original verified data, and the comparability caveat.

## Offline execution and model access

Generation and evaluation containers use Docker `--network none`. Generation starts a loopback HTTP bridge inside each container and mounts a host Unix socket. The host proxy forwards only inference POST requests to the fixed `MODEL_API_UPSTREAM` base URL and rejects any body whose `model` differs from `MODEL_API_MODEL`, so a misconfigured agent cannot silently switch deployments. It supports Responses (including compaction), Chat Completions, and Messages/count_tokens over HTTP, including streaming; the bundled launcher configures the Responses transport only. Only client-executed tools are allowed: Responses function/custom/local-shell tools and recursively checked namespace groups, Chat Completions function tools, and Messages custom tools. Other tool types, arbitrary proxy targets, CONNECT, redirects, and remote MCP are rejected. The upstream URL must not contain credentials; authentication comes from the agent configuration. This is not a general HTTP proxy. Remote image/file inputs and provider file references are rejected; text and inline media remain supported. Evaluation has no model bridge.

Codex requires `CODEX_MODEL_PROVIDER`, and Trae CLI requires `TRAE_MODEL_PROVIDER`, each naming the provider key from its profile. Their adapters override that provider's base URL to the loopback bridge and disable web search. OpenCode requires `OPENCODE_PROVIDER` to match the provider prefix of its top-level `model`; its adapter also checks that `small_model` uses the same provider and rewrites that provider's `options.baseURL` in the per-case JSON configuration copy. Use a Responses-capable HTTP API deployment; WebSockets and non-inference provider endpoints are not supported. These overrides do not change the configured model or reasoning effort.

All dependencies must already exist in the image or mounted runtime/data seed. Automatic downloads, package installation, external MCP servers and other network-dependent tools cannot work offline. OpenCode configuration must be JSON and its provider dependencies must be cached.

Each case uses a fresh container and private HOME. Host session histories, personal instruction files and memory directories are excluded from the mounted seeds. Codex and Trae memory/import and hooks are explicitly disabled; OpenCode runs without project configuration, external plugins or external skill discovery. Both external agents therefore ignore repository instruction files such as `AGENTS.md` and `CLAUDE.md`, matching the frozen comparison protocol; the task prompt and the repository itself remain available. Model requests are the permitted network exception; direct internet access is blocked.

## Step 1: Generate patches

Use the unified inference entrypoint:

```bash
bash run_infer.sh <agent> <prefix>
```

Supported agents:

```text
codex
opencode
traecli
```

The generated run is written under:

```text
agent_runs/<timestamp>-<prefix>/
```

with patches in:

```text
agent_runs/<timestamp>-<prefix>/patches/CVE-....patch
```

Common controls:

```bash
export CONCURRENCY=5             # parallel containers / agents
export LIMIT=-1                  # -1 means all selected cases
export AGENT_TIMEOUT=3600        # per-case inference timeout in seconds
```

### Codex

Required environment variables:

```bash
export CODEX_BIN=/path/to/bin/codex
export CODEX_CONFIG=/path/to/codex-home/my-profile.config.toml
export CODEX_MODEL_PROVIDER=my_provider
export MODEL_API_UPSTREAM=https://provider.example/v1
export MODEL_API_MODEL=<model-id-sent-upstream>
export MODEL_API_PROVIDER=<vendor-label>
```

Run:

```bash
CONCURRENCY=5 bash run_infer.sh codex codex_my_profile
```

The Codex adapter mounts the executable and a filtered configuration/authentication seed into each case container. It derives the profile name from `CODEX_CONFIG` (`<profile>.config.toml`) and runs:

```text
codex exec --profile <profile> ... -C {workdir} < {prompt_file}
# The adapter also overrides the provider base URL and disables web search.
```

### OpenCode

Required environment variables:

```bash
export OPENCODE_BIN=/path/to/bin/opencode
export OPENCODE_CONFIG=/path/to/opencode-home/config/opencode/opencode.json
export OPENCODE_PROVIDER=my_provider
export MODEL_API_UPSTREAM=https://provider.example/v1
export MODEL_API_MODEL=<model-id-sent-upstream>
export MODEL_API_PROVIDER=<vendor-label>
```

Run:

```bash
CONCURRENCY=5 bash run_infer.sh opencode opencode_default
```

The configuration must select the provider through `@ai-sdk/openai` (the
Responses transport), not `@ai-sdk/openai-compatible`, set `small_model` to the
same provider, and set `reasoningEffort` in the model options. `run_model.py`
writes that configuration automatically; a hand-written config without these
keys falls back to OpenCode's default medium effort. Each turn also issues an auxiliary
request (title/context) next to the main one; `small_model` decides which model serves it,
so leaving it unset lets that request drift to another deployment, which the model-pinned
proxy then rejects.

The OpenCode adapter mounts the executable, selected JSON configuration, authentication and runtime dependencies. Session databases and historical storage are excluded. The seed is copied into a private container directory before running:

```text
opencode run --format json --auto < {prompt_file}
```

Example:

```bash
export OPENCODE_BIN=/path/to/opencode-runtime/bin/opencode
export OPENCODE_CONFIG=/path/to/opencode-home/config/opencode/opencode.json
export OPENCODE_PROVIDER=my_provider
export MODEL_API_UPSTREAM=https://provider.example/v1
export MODEL_API_MODEL=<model-id-sent-upstream>
export MODEL_API_PROVIDER=<vendor-label>
```

### Trae CLI

Install the standalone CLI (the desktop IDE package is not required):

```bash
curl -fL https://lf-cdn.trae.com.cn/obj/trae-com-cn/trae-cli/install_v2.sh -o /tmp/traecli-install.sh
sh /tmp/traecli-install.sh --no-auto-update
export PATH="$HOME/.local/bin:$PATH"
```

Use a `<profile>.traecli.toml` configuration with a custom HTTP model provider and its authentication configured:

```bash
export TRAE_BIN="$HOME/.local/bin/traex"
export TRAE_CONFIG=/path/to/trae-home/my-profile.traecli.toml
export TRAE_MODEL_PROVIDER=my_provider
export MODEL_API_UPSTREAM=https://provider.example/v1
export MODEL_API_MODEL=<model-id-sent-upstream>
export MODEL_API_PROVIDER=<vendor-label>

LIMIT=1 CONCURRENCY=1 bash run_infer.sh traecli trae_my_profile
bash run_eval.sh trae_my_profile
```

The adapter copies a filtered configuration/authentication seed into each container, overrides the selected provider's base URL, and disables web search. It preserves the configured model.

Trae is kept as an additional adapter and is not part of the frozen comparison matrix: it cannot reach the DeepSeek deployment, and its reasoning effort is not forwarded. Verified on 2026-11-02 with traecli 0.213.1: `model_reasoning_effort = "low"` passes config validation (`--strict-config` accepts it) but the outgoing Responses request still carries `"reasoning": null` for `gpt-5`, `o3`, `gpt-5-codex` and `gpt-4.1`, with both the profile key and a `-c` override. `run_model.py` therefore does not write the key for `traecli`.

Use API credentials available inside the container; account login and provider discovery that require other network endpoints are not supported by the inference proxy.

## Step 2: Evaluate patches

After patch generation, run:

```bash
bash run_eval.sh <prefix>
```

Example:

```bash
bash run_eval.sh codex_my_profile
```

`run_eval.sh` will:

1. find the latest `agent_runs/*-<prefix>` directory; 2. convert `patches/*.patch` to PatchEval evaluation JSONL with `process_data.py`; 3. call `../evaluation/run_evaluation.py`.

The converted patch file is written to:

```text
eval_inputs/<prefix>.jsonl
```

Evaluation results are written by `run_evaluation.py` under:

```text
../evaluation/evaluation_output/results/<prefix>/
```

You can also pass the run directory explicitly:

```bash
bash run_eval.sh <prefix> /path/to/agent_runs/<timestamp>-<prefix>
```

## Runner behavior

`patch_agent_runner.py` is patch-generation-only. It does **not** run evaluation.

For each sample it:

1. reads the CVE image from the sample's `image_url` field; 2. starts a Docker container with networking disabled and a restricted model API bridge; 3. selects the repository workdir from `/workspace/<repo basename>`, then `/workspace/<repo basename lower-case>`, then `/workspace`; 4. hides non-repository files under `/workspace` when a repository subdirectory is selected; 5. writes a prompt to `/results/prompt.txt`; 6. runs the selected agent command; 7. collects `/workspace/fix.patch` or `git diff HEAD -U3`; 8. writes patches, logs, and `results.jsonl`.

## Output files

A generation run contains:

```text
agent_runs/<timestamp>-<prefix>/
├── patches/           # CVE-keyed patches for process_data.py
├── .work/             # prompt, stdout/stderr, codex-sessions/, opencode-session.json
├── trajectories/      # per case: <CVE>.trajectory.jsonl + <CVE>.metrics.json
├── results.jsonl
├── run_metadata.json
└── summary.json
```

## Transcript and usage contract

`trajectory.py` implements sec-review-bot's `agent-transcript/v1` contract independently. It does not import the agents runtime, experiment reports, or their test fixtures. Trajectories are message-only JSONL recordings: each record has an increasing `seq`, `event: "message"`, and the CLI identity in `agent`. Messages have a unique nonempty `id`, a canonical `role` (`system`, `user`, `assistant`, or `tool`), and their text or content blocks. Available source IDs are retained; missing IDs receive deterministic local identities. Repeated identical messages are retained once; conflicting observations of an identity fail export.

Non-null `message.usage_metadata` contains exactly six nullable fields: `input_tokens`, `output_tokens`, `total_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, and `reasoning_tokens`. Input includes cache reads and writes; output includes reasoning. Cache and reasoning are subsets, never additional charges to these totals. Canonical total is derived from input and output when both are known. A conflicting source total produces a warning, and the original observation remains available outside canonical usage. Invalid counts and impossible subsets fail export.

Missing measurements remain null, including unreported cache creation or reasoning counts. Explicit zero stays zero. Each aggregate field is unknown when any covered observation lacks that measurement; derived uncached input requires known input and both cache subsets.

Codex writes this case's raw rollout files directly under `.work/<case>/codex-sessions/` while it runs. The files survive process failure, timeout, and container removal; config and authentication files remain in the private container home. Host histories are not imported. The canonical projection prefers these rollouts over `--json` stdout. Each `token_usage_record` identifies an upstream response and carries its own usage; sequential assistant output items before that completion form one response, including reasoning and tool requests. The response ID is retained in the message and `call_id`, with original output items and usage evidence in response metadata. Tool results remain separate. The projection does not combine output across tool results, input messages, or compaction boundaries.

Metrics distinguish the available observation scopes:

- Codex rollouts populate per-response `usage_metadata` and metrics with `usage_scope: "recorded-responses"`. Metrics count these responses once; repeated completion observations and `token_count` snapshots do not add usage. Stdout turn totals are not added to rollout totals. `model_responses_with_usage_count` counts measured responses without claiming that every model request was captured. Original response usage records are retained in `metrics.source_usage`.
- Without rollout files, Codex/Trae-compatible stdout is projected as message-only observations, with `turn.completed` usage summed in metrics under `usage_scope: "recorded-turns"`. These turn aggregates are not assigned to an individual response.
- OpenCode session totals, when available, populate metrics with `usage_scope: "session"`. They are not added to message usage. Individual message usage remains on its response, with original observations in `response_metadata.token_usage`.
- Without session totals, OpenCode metrics aggregate recorded assistant messages with `usage_scope: "recorded-messages"`. Interrupted exports recover persisted parts, retaining the latest update for an identified part. Unfinished tools retain their requests without fabricated result messages.

Codex projections retain available base instructions, system/developer inputs, original user content, tool requests/results, and compaction summaries. Thread IDs namespace message/call identities; available parent threads establish branch ancestry. Copied parent history in a child rollout is excluded from its projection and accounting. Incoming inter-agent messages are retained as user context with their original sender and recipient in response metadata. A truncated final JSON fragment is treated as interrupted capture, while malformed complete lines fail export. Pending output, unfinished turns, unmeasured compactions and truncated capture keep aggregate fields unknown. Raw rollouts remain available for inspection.

`usage_complete` is false for these external recordings: response completions do not establish complete coverage of request starts, including requests that failed before emitting a response. Turn or session totals and successful CLI execution do not prove complete call recording. The adapter does not invent call lifecycle events, system instructions, or per-call allocation. Reported CLI cost is retained as source evidence; this adapter does not calculate prices.

Run the local adapter regressions with `python -m pytest tests/test_trajectory.py` from the PatchEval root, using an environment with pytest installed. The tests use PatchEval's own CLI samples. `python -m pytest tests/test_codex_rollout_capture.py` additionally exercises an installed Codex CLI against a local mock Responses server without contacting a model provider; it skips when Codex is unavailable.

## Notes

- `patch_url` metadata should not be used by agents as a repair source.
- Each case also writes `trajectories/<CVE>.trajectory.jsonl` in the project trajectory schema (one `{seq, event, message}` record per line) plus `<CVE>.metrics.json` with token usage, tool-call count and cost when available. `trajectory.py` maps Codex item events and OpenCode session exports onto that schema; the OpenCode adapter runs `opencode export <sessionID>` before the container is removed.
- When a tool reports a real unified diff (OpenCode does for `edit` and `apply_patch`), it is stored on the tool message as `additional_kwargs.diff` with `diff_truncated`; viewers render that rather than reconstructing from arguments. Codex file-change events are retained as `apply_patch` calls with the reported file paths, change kinds and completion status; these events do not provide a unified diff.
- Trajectories carry provenance: every `assistant` message gets `response_metadata.model_provider` and `model_name` from `MODEL_API_PROVIDER` / `MODEL_API_MODEL`, and `metrics.json` repeats both. Neither CLI reports the upstream vendor in its event stream, so this is the only per-artifact provenance.
- During patch generation, the CLI prints case-level progress when a case finishes. Detailed agent logs are written under `.work/<case>/agent_stdout.txt` and `.work/<case>/agent_stderr.txt`.
- `run_infer.sh` exits with a non-zero status if any selected case fails, while preserving all generated patches and logs. Failed cases are represented by empty patch files and are counted as failed repairs during evaluation.
- Keep credentials and runtime homes outside version control.
