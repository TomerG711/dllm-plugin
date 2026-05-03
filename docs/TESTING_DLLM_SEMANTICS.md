# dLLM semantics testing (issue #35)

This document is the **test plan matrix** for behavioral coverage of the dLLM
plugin stack: scheduler draft state, worker/runner handoff, remasking contracts,
structured output, and the **EngineCore** draft-token hook alignment with
[vllm#36155](https://github.com/vllm-project/vllm/issues/36155) and
[vLLM PR #36391](https://github.com/vllm-project/vllm/pull/36391).

Upstream design references: [DESIGN_MVP.md](DESIGN_MVP.md) §4 / §6–7,
[CONTRACTS.md](CONTRACTS.md). Milestone mapping: [#19](https://github.com/vllm-project/dllm-plugin/issues/19).
Hook pins and wheel drift: [#2](https://github.com/vllm-project/dllm-plugin/issues/2).

## EngineCore test shim

Stock PyPI vLLM in the `0.20.x` range may still gate
`take_draft_token_ids()` / `update_draft_token_ids*` on `use_spec_decode`.
[`tests/support/engine_core_draft_hook.py`](../tests/support/engine_core_draft_hook.py)
applies a **test-only** patch matching PR **#36391** while the context manager is
active, and **no-ops** when the installed engine already matches that behavior.

Disable all patching (for debugging) with:

```bash
export VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH=1
```

## Pytest markers

| Marker | Meaning |
|--------|---------|
| `dllm_engine_patch` | CPU-oriented tests for the PR **#36391** shim and related scheduler draft helpers; require `pytest.importorskip("vllm")`. |
| `dllm_gpu_integration` | GPU behavioral tests (CUDA); skip on CPU-only runners. |

## Where tests run

| Tier | Typical files | Default CI (`ci` matrix) | `vllm-extra` job (`.github/workflows/ci.yml`) | Helm GPU job |
|------|----------------|-------------------------|-----------------------------------------------|----------------|
| A — component (CPU) | `tests/test_scheduler.py`, `tests/test_runtime_scheduler_draft_output.py` | yes | yes (when `DraftTokenIds` import works) | optional (same image) |
| B — EngineCore shim (CPU) | `tests/test_engine_core_draft_hook_patch.py` | skipped (no vLLM) | **yes** | optional |
| C — GPU mock-stack + patch | `tests/test_dllm_gpu_integration_semantics.py` | skip (no CUDA) | skip (no CUDA) | add path when promoting |
| D — GPU grammar / MRV2 | `tests/test_vllm_gpu_mrv2_monkeypatch_grammar.py` | skip | skip | **yes** (default chart) |
| E — mock `LLM.generate` | `tests/test_vllm_mock_integration.py` | skip | skip | **yes** |

Manual optional smoke: `.github/workflows/optional-vllm-smoke.yml` (`workflow_dispatch`).

## Local commands

```bash
uv sync --group dev --extra vllm   # Linux/CUDA-capable host
uv run pytest -q -m dllm_engine_patch
uv run pytest -q -m dllm_gpu_integration   # needs GPU
```

## Out of scope / skipped for MVP

Per [DESIGN_MVP.md](DESIGN_MVP.md) §6.1, **async scheduling + structured outputs + dLLM**
is not CI-validated; tests remain absent or must be explicitly `xfail`/`skip`
with a pointer here and in [#19](https://github.com/vllm-project/dllm-plugin/issues/19).

## Phase mapping (milestone #19)

| Tier | Phases exercised |
|------|------------------|
| A | 1 / 4 — contracts and runtime scheduler draft padding |
| B | 0 / 4 — engine hook semantics vs upstream PR **#36391** |
| C | 4 / 6 — multi-step mock stack on GPU with hook shim |
| D | 4 / 6 — grammar + bitmask path on GPU |
| E | 6 — end-to-end `LLM.generate` smoke |
