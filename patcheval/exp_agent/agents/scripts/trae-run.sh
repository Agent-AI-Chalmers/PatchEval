#!/usr/bin/env bash
# Run one Trae CLI case inside the offline per-case container.
set -euo pipefail

rm -rf /tmp/trae-home
cp -a /opt/trae-home-src /tmp/trae-home
mkdir -p /tmp/trae-home/cli /results/trae-sessions
ln -s /results/trae-sessions /tmp/trae-home/cli/sessions

RUST_LOG=off TRAE_HOME=/tmp/trae-home exec /usr/local/bin/traex exec \
  --profile "${PATCHAGENT_PROFILE}" \
  --json \
  --ignore-rules \
  --disable memories \
  --disable external_agent_memory_import \
  --disable hooks \
  -c 'web_search="disabled"' \
  -c "model_provider=\"${PATCHAGENT_PROVIDER}\"" \
  -c "model_providers.${PATCHAGENT_PROVIDER}.base_url=\"http://127.0.0.1:18080/v1\"" \
  --skip-git-repo-check \
  -y \
  --output-last-message /results/traex_last_message.txt \
  < "$PATCHAGENT_PROMPT_FILE"
