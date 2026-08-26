#!/usr/bin/env bash
set -euo pipefail

# Required: REPO_URL and RUNNER_TOKEN are passed at runtime (never baked into the image).
: "${REPO_URL:?Set REPO_URL, e.g. https://github.com/<owner>/<repo>}"
: "${RUNNER_TOKEN:?Set RUNNER_TOKEN (a fresh registration token from Settings > Actions > Runners)}"

RUNNER_NAME="${RUNNER_NAME:-$(hostname)-podman}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,linux,x64,podman}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-_work}"

cd /home/runner/actions-runner

cleanup() {
    echo "Removing runner registration..."
    ./config.sh remove --token "${RUNNER_TOKEN}" || true
}
# De-register on stop so GitHub does not accumulate offline runners
trap 'cleanup; exit 130' INT TERM

./config.sh \
    --unattended \
    --replace \
    --url "${REPO_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME}" \
    --labels "${RUNNER_LABELS}" \
    --work "${RUNNER_WORKDIR}" \
    ${EPHEMERAL:+--ephemeral}

./run.sh &
wait $!
