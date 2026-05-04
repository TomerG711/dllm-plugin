#!/usr/bin/env bash
# Deploy tools/helm/dllm-plugin-gpu-test on GKE L4 (jounce-tainted pool), wait for Job
# success, and retry on failure. Requires public git repo/branch for in-cluster clone.
#
# Required: KUBE_CONTEXT
# Optional: NAMESPACE (default dllm), RELEASE (default dllm-plugin-gpu-test),
#   GIT_REPO, GIT_BRANCH (default: detect from repo root or main),
#   MAX_ATTEMPTS (default 3), SLEEP_SEC (default 30, max 30), WAIT_TIMEOUT (default 55m),
#   POLL_INTERVAL (default 45s): while waiting, print job/pod status and log tail this often
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
POLL_INTERVAL="${POLL_INTERVAL:-45}"

parse_wait_seconds() {
  local t="${1:-55m}"
  if [[ "$t" =~ ^([0-9]+)m$ ]]; then echo $((${BASH_REMATCH[1]} * 60))
  elif [[ "$t" =~ ^([0-9]+)s$ ]]; then echo "${BASH_REMATCH[1]}"
  else echo 3300
  fi
}
MAX_WAIT_SEC="$(parse_wait_seconds "$WAIT_TIMEOUT")"

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

  job_ok=0
  t0="$(date +%s)"
  while true; do
    now="$(date +%s)"
    elapsed=$((now - t0))
    if ((elapsed >= MAX_WAIT_SEC)); then
      echo ">>> TIMEOUT after ${elapsed}s (limit ${MAX_WAIT_SEC}s)"
      break
    fi

    succeeded="$(kubectl --context "$KUBE_CONTEXT" -n "$NS" get job "$RELEASE" -o jsonpath='{.status.succeeded}' 2>/dev/null || echo 0)"
    failed="$(kubectl --context "$KUBE_CONTEXT" -n "$NS" get job "$RELEASE" -o jsonpath='{.status.failed}' 2>/dev/null || echo 0)"
    phase="$(kubectl --context "$KUBE_CONTEXT" -n "$NS" get pods -l "app.kubernetes.io/instance=$RELEASE" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")"
    pod="$(kubectl --context "$KUBE_CONTEXT" -n "$NS" get pods -l "app.kubernetes.io/instance=$RELEASE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    term_reason="" exit_code="" wait_reason=""
    if [[ -n "${pod:-}" ]]; then
      term_reason="$(kubectl --context "$KUBE_CONTEXT" -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.terminated.reason}' 2>/dev/null || true)"
      exit_code="$(kubectl --context "$KUBE_CONTEXT" -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)"
      wait_reason="$(kubectl --context "$KUBE_CONTEXT" -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)"
    fi

    echo ""
    echo ">>> poll +${elapsed}s succeeded=${succeeded:-0} failed=${failed:-0} pod_phase=${phase:-n/a} waiting=${wait_reason:-} terminated=${term_reason:-} exit=${exit_code:-}"

    kubectl --context "$KUBE_CONTEXT" -n "$NS" get job,pods -l "app.kubernetes.io/instance=$RELEASE" -o wide 2>&1 || true
    echo ">>> log tail (job/$RELEASE):"
    kubectl --context "$KUBE_CONTEXT" -n "$NS" logs "job/$RELEASE" --tail=80 2>&1 || true

    if [[ "${succeeded:-0}" == "1" ]]; then
      echo ">>> JOB SUCCEEDED"
      kubectl --context "$KUBE_CONTEXT" -n "$NS" logs "job/$RELEASE" --tail=200
      job_ok=1
      break
    fi
    if [[ "${failed:-0}" == "1" ]] || [[ "$phase" == "Failed" ]]; then
      echo ">>> JOB OR POD FAILED (detected during poll)"
      kubectl --context "$KUBE_CONTEXT" -n "$NS" logs "job/$RELEASE" --tail=800 2>&1 || true
      if [[ -n "${pod:-}" ]]; then
        kubectl --context "$KUBE_CONTEXT" -n "$NS" describe pod "$pod" 2>&1 | tail -80 || true
      fi
      job_ok=0
      break
    fi
    if [[ -n "${pod:-}" && -n "$term_reason" && "${exit_code:-}" != "" && "$exit_code" != "0" ]]; then
      echo ">>> CONTAINER TERMINATED non-zero (exit ${exit_code})"
      kubectl --context "$KUBE_CONTEXT" -n "$NS" logs "job/$RELEASE" --tail=800 2>&1 || true
      job_ok=0
      break
    fi
    if [[ -n "${pod:-}" ]] && { [[ "$wait_reason" == "CrashLoopBackOff" ]] || [[ "$wait_reason" == "ImagePullBackOff" ]] || [[ "$wait_reason" == "ErrImagePull" ]]; }; then
      echo ">>> POD STUCK: $wait_reason"
      kubectl --context "$KUBE_CONTEXT" -n "$NS" describe pod "$pod" 2>&1 | tail -80 || true
      job_ok=0
      break
    fi

    sleep "$POLL_INTERVAL"
  done

  if [[ "$job_ok" == "1" ]]; then
    exit 0
  fi

  echo ">>> JOB did not complete successfully (attempt ${attempt})"
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
