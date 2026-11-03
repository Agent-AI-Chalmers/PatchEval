#!/usr/bin/env bash
# Run one Codex CLI case inside the offline per-case container.
#
# The runner provides PATCHAGENT_WORKDIR and PATCHAGENT_PROMPT_FILE; this script
# is mounted read-only at /opt/agent-run.sh.
set -euo pipefail

# Host state never enters the container: only config/auth are seeded, and they
# are copied here so concurrent cases cannot share a CODEX_HOME.
rm -rf /tmp/codex-home
cp -a /opt/codex-home-src /tmp/codex-home
# Persist this case's rollouts directly on the results mount, including partial
# records on timeout. Keep credentials/config in the private container home.
mkdir -p /results/codex-sessions
ln -s /results/codex-sessions /tmp/codex-home/sessions

CODEX_HOME=/tmp/codex-home exec /usr/local/bin/codex exec \
  --profile "${PATCHAGENT_PROFILE}" \
  --json \
  --ignore-rules \
  --disable memories \
  --disable external_agent_memory_import \
  --disable hooks \
  -c 'web_search="disabled"' \
  -c "model_provider=\"${PATCHAGENT_PROVIDER}\"" \
  -c "model_providers.${PATCHAGENT_PROVIDER}.base_url=\"http://127.0.0.1:18080/v1\"" \
  --dangerously-bypass-approvals-and-sandbox \
  -C "$PATCHAGENT_WORKDIR" \
  < "$PATCHAGENT_PROMPT_FILE"
