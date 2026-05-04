# Helm chart: GPU integration job (template)

This chart runs **GPU pytest targets** from `tests.pytestPaths` (see `values.yaml`): by default that includes **EngineCore draft-hook shim** (`test_engine_core_draft_hook_patch.py`), **runtime scheduler drafts** (`test_runtime_scheduler_draft_output.py`), **GPU semantics** (`test_dllm_gpu_integration_semantics.py`, with optional skips via `testEnv`), **mock-stack end-to-end** (`test_vllm_mock_integration.py`), **selected MRV2 GPU grammar tests** (`test_vllm_gpu_mrv2_monkeypatch_grammar.py`—see `values.yaml` for the exact nodes), and **two-phase** (`test_two_phase_dllm.py`). When **`tests.runServeHttpSmoke`** is `true` (default), the Job also runs **`tools/e2e/serve_http_smoke.sh`** after pytest (`vllm serve` on the mock HF config + `curl` to `/health` and `/v1/chat/completions`). Set `tests.runServeHttpSmoke: false` to skip that step. The Job template always tolerates `nvidia.com/gpu` (NoSchedule). **Default `values.yaml`** also includes the **jounce.io L4 GPU pool** toleration (`jounce.io/nodetype=L4:NoSchedule`); if your nodes are not tainted that way, set `scheduling.extraTolerations: []` or replace with your fleet’s taints. Use `scheduling.nodeSelector` when you need a specific accelerator label.

The job **clones `git.repoUrl` / `git.branch` from GitHub** inside the container; only commits **pushed** to that remote branch are exercised.

## Fork-and-adjust

1. Copy or vendor this chart under your infra repo.
2. Set `scheduling.nodeSelector` / `scheduling.extraTolerations` in `values.yaml` to match **your** accelerator nodes (cloud provider labels, spot tiers, etc.).
3. Override `image`, `git.repoUrl`, `git.branch`, and resource requests as needed.

Example for GKE L4-style scheduling (illustrative—verify against your cluster labels):

```yaml
scheduling:
  nodeSelector:
    cloud.google.com/gke-accelerator: "nvidia-l4"
  extraTolerations: []
```

The default `extraTolerations` target that **jounce.io L4** pattern. If your pool uses a different taint, edit `values.yaml` or the pod stays `Pending`. Confirm with `kubectl describe node <gpu-node> | grep -A2 Taints`.

## Maintainer: L4 deploy + validate loop

From the repository root, [`tools/e2e/helm_l4_gpu_validate.sh`](../../e2e/helm_l4_gpu_validate.sh) runs `helm upgrade --install` against this chart with **`cloud.google.com/gke-accelerator=nvidia-l4`** and the **jounce.io/nodetype=L4** toleration, waits for the Job to complete (default timeout **55m**), prints tail logs on success, dumps more logs and `describe pod` on failure, and retries up to **3** times with **30s** sleep between attempts.

Required env: **`KUBE_CONTEXT`**. Optional: **`GIT_REPO`** / **`GIT_BRANCH`** (defaults: `https://github.com/vllm-project/dllm-plugin.git` and the current `git` branch when the script is run from a checkout, else `main`). The cluster must be able to `git clone` the URL over HTTPS without credentials (public repo or equivalent).

Default chart **`testEnv`** sets **`DLLM_SKIP_GPU_SEMANTICS_MULTI_STEP=1`** so the Helm Job skips one GPU case that can hit illegal CUDA access on some L4 nodes; unset via Helm overrides to run the full file on hardware where it passes.

Default **`tests.pytestPaths`** runs a **single** MRV2 GPU case (`test_gpu_injects_dllm_mrv2_via_monkeypatch_stock_worker`); regex structured-output nodes are omitted on Helm L4 because decoded text can be empty on this stack—override `tests.pytestPaths` locally or via Helm when validating on hardware where they pass.

```bash
export KUBE_CONTEXT='gke_it-gcp-model-validation_us-central1_rhoai-benchmark-development-cluster'
export GIT_REPO='https://github.com/you/dllm-plugin.git'   # optional
export GIT_BRANCH='your-feature-branch'                    # optional
bash tools/e2e/helm_l4_gpu_validate.sh
```

See also `docs/OPERATOR_LLaDA2.md` (Helm subsection).
