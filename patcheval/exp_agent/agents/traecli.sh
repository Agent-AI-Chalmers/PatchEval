# TraeCLI / TraeX agent adapter. Source this file from run_infer.sh.
#
# Required:
#   TRAE_BIN=/path/to/traex
#   TRAE_CONFIG=/path/to/traecli-runtime/trae-home/<profile>.traecli.toml
#
# TRAE_CONFIG is the profile/config input. The adapter derives:
#   TRAE_HOME_SRC = dirname(TRAE_CONFIG)
#   TRAE_PROFILE  = basename(TRAE_CONFIG) without .traecli.toml

: "${TRAE_MODEL_PROVIDER:?Set TRAE_MODEL_PROVIDER to the provider key in the selected Trae profile}"
if [[ ! "$TRAE_MODEL_PROVIDER" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid TRAE_MODEL_PROVIDER" >&2
  exit 1
fi

: "${TRAE_BIN:?Set TRAE_BIN to the TraeX executable, e.g. /path/to/bin/traex}"
: "${TRAE_CONFIG:?Set TRAE_CONFIG to a TraeCLI profile config file, e.g. /path/to/traecli-runtime/trae-home/doubao-8hp8j.traecli.toml}"

TRAE_BIN="$(realpath "$TRAE_BIN")"
if [[ ! -x "$TRAE_BIN" ]]; then
  echo "TRAE_BIN does not exist or is not executable: $TRAE_BIN" >&2
  exit 1
fi
TRAE_CONFIG="$(realpath "$TRAE_CONFIG")"
if [[ ! -f "$TRAE_CONFIG" ]]; then
  echo "TRAE_CONFIG does not exist or is not a file: $TRAE_CONFIG" >&2
  exit 1
fi
TRAE_HOME_SRC="$(cd "$(dirname "$TRAE_CONFIG")" && pwd)"
config_name="$(basename "$TRAE_CONFIG")"
if [[ "$config_name" == *.traecli.toml ]]; then
  TRAE_PROFILE="${config_name%.traecli.toml}"
else
  echo "TRAE_CONFIG must be named <profile>.traecli.toml: $TRAE_CONFIG" >&2
  exit 1
fi

TRAE_CODE_MODE_HOST="$(dirname "$TRAE_BIN")/traex-code-mode-host"
if [[ ! -x "$TRAE_CODE_MODE_HOST" ]]; then
  echo "Missing Trae Code Mode runtime: $TRAE_CODE_MODE_HOST; reinstall the complete Trae CLI distribution" >&2
  exit 1
fi

AGENT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"
AGENT_MOUNTS=(
  "${TRAE_CODE_MODE_HOST}:/usr/local/bin/traex-code-mode-host:ro"
  "${TRAE_BIN}:/usr/local/bin/traex:ro"
  "${TRAE_HOME_SRC}:/opt/trae-home-src:ro"
  "${AGENT_SCRIPT_DIR}/trae-run.sh:/opt/agent-run.sh:ro"
)
AGENT_EXTRA_ARGS=()
AGENT_COMMAND="PATCHAGENT_PROFILE=${TRAE_PROFILE} PATCHAGENT_PROVIDER=${TRAE_MODEL_PROVIDER} bash /opt/agent-run.sh"
