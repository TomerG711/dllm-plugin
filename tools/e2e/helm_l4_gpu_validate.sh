#!/usr/bin/env bash
# Deploy tools/helm/dllm-plugin-gpu-test on GKE L4 (jounce-tainted pool), wait for Job
# success, and retry on failure. Requires public git repo/branch for in-cluster clone.
#
# Required: KUBE_CONTEXT
# Optional: NAMESPACE (default dllm), RELEASE (default dllm-plugin-gpu-test),
#   GIT_REPO, GIT_BRANCH (default: detect from repo root or main),
#   MAX_ATTEMPTS (default 3), SLEEP_SEC (default 30, max 30), WAIT_TIMEOUT (default 55m)
#
# Example:
#   export KUBE_CONTEXT=gke_it-gcp-model-validation_us-central1_rhoai-benchmark-development-cluster
#   export GIT_REPO=https://github.com/you/dllm-plugin.git
#   export GIT_BRANCH=your-branch
#   bash tools/e2e/helm_l4_gpu_validate.sh

set -euo pipefail

: "${KUBE_CONTEXT:?set KUBE_CONTEXT to your kubectl context name}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHART="$REPO_ROOT/tools/helm/dllm-plugin-gpu-test"

NS="${NAMESPACE:-dllm}"
RELEASE="${RELEASE:-dllm-plugin-gpu-test}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"
if (( SLEEP_SEC > 30 )); then
  SLEEP_SEC=30
fi
WAIT_TIMEOUT="${WAIT_TIMEOUT:-55m}"

GIT_REPO="${GIT_REPO:-https://github.com/vllm-project/dllm-plugin.git}"
if [[ -z "${GIT_BRANCH:-}" ]]; then
  if git -C "$REPO_ROOT" rev-parse --git-dir &>/dev/null; then
    GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)" || GIT_BRANCH=main
  else
    GIT_BRANCH=main
  fi
fi

kubectl --context "$KUBE_CONTEXT" get ns "$NS" &>/dev/null || kubectl --context "$KUBE_CONTEXT" create ns "$NS"

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  echo "=== Helm GPU L4 validate attempt ${attempt}/${MAX_ATTEMPTS} ($(date -Iseconds)) ==="
  echo "    context=$KUBE_CONTEXT ns=$NS release=$RELEASE"
  echo "    git.repoUrl=$GIT_REPO git.branch=$GIT_BRANCH"

  kubectl --context "$KUBE_CONTEXT" -n "$NS" delete job "$RELEASE" --ignore-not-found

  helm upgrade --install "$RELEASE" "$CHART" \
    --kube-context "$KUBE_CONTEXT" \
    --namespace "$NS" \
    --set "git.repoUrl=$GIT_REPO" \
    --set "git.branch=$GIT_BRANCH" \
    --set-json 'scheduling.nodeSelector={"cloud.google.com/gke-accelerator":"nvidia-l4"}' \
    --set-json 'scheduling.extraTolerations=[{"key":"jounce.io/nodetype","operator":"Equal","value":"L4","effect":"NoSchedule"}]'

  if kubectl --context "$KUBE_CONTEXT" -n "$NS" wait --for=condition=complete "job/$RELEASE" --timeout="$WAIT_TIMEOUT"; then
    echo ">>> JOB SUCCEEDED"
    kubectl --context "$KUBE_CONTEXT" -n "$NS" logs "job/$RELEASE" --tail=200
    exit 0
  fi

  echo ">>> JOB did not complete (attempt ${attempt})"
  kubectl --context "$KUBE_CONTEXT" -n "$NS" logs "job/$RELEASE" --tail=800 2>&1 || true
  POD="$(kubectl --context "$KUBE_CONTEXT" -n "$NS" get pods -l "app.kubernetes.io/instance=$RELEASE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "${POD:-}" ]]; then
    kubectl --context "$KUBE_CONTEXT" -n "$NS" describe pod "$POD" 2>&1 | tail -60 || true
  fi

  if ((attempt < MAX_ATTEMPTS)); then
    echo ">>> Sleeping ${SLEEP_SEC}s before retry..."
    sleep "$SLEEP_SEC"
  fi
done

echo ">>> Exhausted ${MAX_ATTEMPTS} attempt(s)"
exit 1
