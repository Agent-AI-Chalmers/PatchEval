# OpenCode agent adapter. Source this file from run_infer.sh.
#
# Required:
#   OPENCODE_BIN=/path/to/opencode
#   OPENCODE_CONFIG=/path/to/opencode-home/config/opencode/opencode.json
#
# OPENCODE_CONFIG is the single config input. The adapter derives the XDG config
# and data roots from it. Config is mounted read-only; data is mounted read-only
# as a seed and copied to /tmp inside each case container so concurrent runs do
# not write to the same host state/log files.

: "${OPENCODE_PROVIDER:?Set OPENCODE_PROVIDER to the provider key selected by the OpenCode model configuration}"
if [[ ! "$OPENCODE_PROVIDER" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid OPENCODE_PROVIDER" >&2
  exit 1
fi

: "${OPENCODE_BIN:?Set OPENCODE_BIN to the OpenCode executable, e.g. /path/to/bin/opencode}"
: "${OPENCODE_CONFIG:?Set OPENCODE_CONFIG to opencode.json, e.g. /path/to/opencode-home/config/opencode/opencode.json}"

AGENT_MOUNTS=()
AGENT_EXTRA_ARGS=()
OPENCODE_BIN="$(realpath "$OPENCODE_BIN")"
if [[ ! -x "$OPENCODE_BIN" ]]; then
  echo "OPENCODE_BIN does not exist or is not executable: $OPENCODE_BIN" >&2
  exit 1
fi
OPENCODE_CONFIG="$(realpath "$OPENCODE_CONFIG")"
if [[ ! -f "$OPENCODE_CONFIG" ]]; then
  echo "OPENCODE_CONFIG does not exist or is not a file: $OPENCODE_CONFIG" >&2
  exit 1
fi
OPENCODE_CONFIG_HOME="$(cd "$(dirname "$OPENCODE_CONFIG")/.." && pwd)"
OPENCODE_DATA_HOME="$(cd "${OPENCODE_CONFIG_HOME}/../data" && pwd)"

AGENT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"

AGENT_MOUNTS+=("${OPENCODE_BIN}:/usr/local/bin/opencode:ro")
AGENT_MOUNTS+=("${OPENCODE_CONFIG_HOME}:/opt/opencode-config-src:ro")
AGENT_MOUNTS+=("${OPENCODE_DATA_HOME}:/opt/opencode-data-src:ro")
AGENT_MOUNTS+=("${AGENT_SCRIPT_DIR}/opencode-run.sh:/opt/agent-run.sh:ro")
AGENT_EXTRA_ARGS+=(--opencode-provider "$OPENCODE_PROVIDER")
AGENT_COMMAND="bash /opt/agent-run.sh"
