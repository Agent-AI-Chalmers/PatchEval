#!/usr/bin/env bash
# Run one OpenCode case inside the offline per-case container and export its
# session before the container is removed.
set -uo pipefail

# Private config/data roots: seeds are copied so concurrent cases never share
# host state, and /results keeps the raw stream for trajectory extraction.
rm -rf /tmp/opencode-config /tmp/opencode-data
mkdir -p /tmp/opencode-config /tmp/opencode-data
cp -a /opt/opencode-config-src/. /tmp/opencode-config/
cp -a /opt/opencode-data-src/. /tmp/opencode-data/
python3 /opt/model-api-bridge.py --configure-opencode

export OPENCODE_DISABLE_AUTOUPDATE=true
# Repository instructions remain enabled independently of skill/plugin restrictions.
unset OPENCODE_DISABLE_PROJECT_CONFIG OPENCODE_DISABLE_CLAUDE_CODE OPENCODE_DISABLE_CLAUDE_CODE_PROMPT
export OPENCODE_DISABLE_CLAUDE_CODE_SKILLS=true
export OPENCODE_DISABLE_EXTERNAL_SKILLS=true
export OPENCODE_DISABLE_DEFAULT_PLUGINS=true
export XDG_CONFIG_HOME=/tmp/opencode-config
export XDG_DATA_HOME=/tmp/opencode-data
export OPENCODE_DB=/results/opencode.db

opencode run --pure --format json --auto \
  < "$PATCHAGENT_PROMPT_FILE" > /results/agent_events.jsonl
status=$?
cat /results/agent_events.jsonl

# Keep a convenient root-session export alongside the durable database. The
# database also retains child sessions and writes made before an interruption.
session=$(grep -o '"sessionID":"[^"]*"' /results/agent_events.jsonl \
  | head -1 \
  | cut -d: -f2 \
  | tr -d '"')
if [ -n "$session" ]; then
  opencode export "$session" > /results/opencode-session.json 2>/dev/null || true
fi

exit "$status"
