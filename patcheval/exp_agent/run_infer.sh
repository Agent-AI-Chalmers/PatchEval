#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <codex|opencode|traecli> [prefix]" >&2
  exit 2
fi

AGENT="$1"
if [[ "$AGENT" != codex && "$AGENT" != opencode && "$AGENT" != traecli ]]; then
  echo "Offline inference currently supports codex, opencode and traecli" >&2
  exit 2
fi
PREFIX="${2:-$AGENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_FILE="${SCRIPT_DIR}/agents/${AGENT}.sh"
if [[ ! -f "$AGENT_FILE" ]]; then
  echo "Unknown agent '$AGENT'; expected one of: codex opencode traecli" >&2
  exit 2
fi

: "${MODEL_API_UPSTREAM:?Set MODEL_API_UPSTREAM to the fixed model API base URL}"
: "${MODEL_API_MODEL:?Set MODEL_API_MODEL to the exact model id sent upstream}"
: "${MODEL_API_PROVIDER:?Set MODEL_API_PROVIDER to the vendor label, e.g. deepseek}"
# shellcheck source=/dev/null
source "$AGENT_FILE"

DATASET="${DATASET:-${SCRIPT_DIR}/../datasets/patcheval_verified_cleaned.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/agent_runs}"

if [[ -n "${CVE:-}" ]]; then
  AGENT_EXTRA_ARGS+=(--cve "$CVE")
fi
if [[ -n "${PATCHAGENT_LAUNCH_CONFIG:-}" ]]; then
  AGENT_EXTRA_ARGS+=(--launch-config "$PATCHAGENT_LAUNCH_CONFIG")
fi

mount_args=()
for mount in "${AGENT_MOUNTS[@]:-}"; do
  mount_args+=(--mount "$mount")
done

python "${SCRIPT_DIR}/patch_agent_runner.py" \
  --input "$DATASET" \
  --output-dir "$OUTPUT_BASE" \
  --limit "${LIMIT:--1}" \
  --concurrency "${CONCURRENCY:-4}" \
  --run-label "$PREFIX" \
  "${mount_args[@]}" \
  "${AGENT_EXTRA_ARGS[@]}" \
  --agent "$AGENT" \
  --agent-command "$AGENT_COMMAND" \
  --model-api-upstream "$MODEL_API_UPSTREAM" \
  --expect-model "$MODEL_API_MODEL" \
  --model-provider "$MODEL_API_PROVIDER" \
  --agent-timeout "${AGENT_TIMEOUT:-3600}" \
  --container-prefix "patcheval-${AGENT}"
