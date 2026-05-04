# dLLM semantics testing (issue #35)

This document is the **test plan matrix** for behavioral coverage of the dLLM
plugin stack: scheduler draft state, worker/runner handoff, remasking contracts,
structured output, and the **EngineCore** draft-token hook alignment with
[vllm#36155](https://github.com/vllm-project/vllm/issues/36155) and
[vLLM PR #36391](https://github.com/vllm-project/vllm/pull/36391).

Upstream design references: [DESIGN_MVP.md](DESIGN_MVP.md) §4 / §6–7,
[CONTRACTS.md](CONTRACTS.md). Milestone mapping: [#19](https://github.com/vllm-project/dllm-plugin/issues/19).
Hook pins and wheel drift: [#2](https://github.com/vllm-project/dllm-plugin/issues/2).

## Issue [#35](https://github.com/vllm-project/dllm-plugin/issues/35) coverage checklist

This matrix is the **intended** semantic surface for closing #35 alongside the
markers above (not every cell needs a dedicated file; some items share one test).

| Area | What is covered | Primary tests |
|------|-----------------|---------------|
| Scheduler block state | `spec_token_ids` matches `scheduled_spec_decode_tokens` across partial commits and `update_draft_token_ids` | `tests/test_scheduler.py` |
| Scheduler negatives | Overlong `sampled_token_ids`, wrong draft length, grammar-constrained draft rewrite rejection | `tests/test_scheduler.py` |
| Worker handoff | `take_draft_token_ids` / `DllmWorkerStep` length checks, v2 runner gate | `tests/test_worker.py`, `tests/test_validation.py` |
| Runtime logits | Row count, vocab width consistency, missing mapping vs mock arch | `tests/test_runtime_adapters.py` |
| Runtime scheduler drafts | Pad/truncate in `update_draft_token_ids_in_output`, empty `num_invalid_spec_tokens`, `_validate_draft_lengths`, `update_draft_token_ids` | `tests/test_runtime_scheduler_draft_output.py` |
| EngineCore hook | PR **#36391**-aligned patch (pytest context manager + optional runtime via `register_dllm`) | `dllm_plugin/engine_core_draft_hook.py`, `tests/test_engine_core_draft_hook_patch.py` |
| HTTP OpenAI smoke | `vllm serve` + `/health` + `/v1/chat/completions` (`curl`); **not** default GitHub `ci` | `tools/e2e/serve_http_smoke.sh`; Helm when `tests.runServeHttpSmoke` |
| GPU mock stack | Multi-step `LLM.generate` with hook; **v1 model runner** rejected under strict validation | `tests/test_dllm_gpu_integration_semantics.py` |
| GPU SO / grammar | MRV2 monkeypatch path | `tests/test_vllm_gpu_mrv2_monkeypatch_grammar.py` |
| End-to-end mock | `LLM.generate` on CUDA | `tests/test_vllm_mock_integration.py` |

## EngineCore draft hook (tests + runtime)

Stock PyPI vLLM in the `0.20.x` range may still gate
`take_draft_token_ids()` / `update_draft_token_ids*` on `use_spec_decode`.

- **Canonical implementation:** [`dllm_plugin/engine_core_draft_hook.py`](../dllm_plugin/engine_core_draft_hook.py) (`patch_engine_core_draft_hook_semantics`, `apply_engine_core_draft_hook_patch_if_needed`, `engine_core_draft_hook_patch_needed`).
- **Tests:** [`tests/support/engine_core_draft_hook.py`](../tests/support/engine_core_draft_hook.py) re-exports the same API for backward-compatible imports and doc links.
- **Runtime (`vllm serve`):** set `VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK=1` so `register_dllm()` applies a **permanent** in-process patch when the wheel still needs PR **#36391** semantics. See [`docs/OPERATOR_LLaDA2.md`](OPERATOR_LLaDA2.md) and [`docs/CONTRACTS.md`](CONTRACTS.md).

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
| C — GPU mock-stack + patch | `tests/test_dllm_gpu_integration_semantics.py` | skip (no CUDA) | skip (no CUDA) | **yes** (default chart) |
| D — GPU grammar / MRV2 | `tests/test_vllm_gpu_mrv2_monkeypatch_grammar.py` | skip | skip | **yes** (default chart) |
| E — mock `LLM.generate` | `tests/test_vllm_mock_integration.py` | skip | skip | **yes** |
| F — HTTP `vllm serve` + `curl` | `tools/e2e/serve_http_smoke.sh` | skip (no GPU / no long-lived server) | skip | **yes** when `tests.runServeHttpSmoke` (default) |

Manual optional smoke: `.github/workflows/optional-vllm-smoke.yml` (`workflow_dispatch`).

## Local commands

```bash
uv sync --group dev --extra vllm   # Linux/CUDA-capable host
uv run pytest -q -m dllm_engine_patch
uv run pytest -q -m dllm_gpu_integration   # needs GPU
bash tools/e2e/serve_http_smoke.sh          # GPU + same env as operator doc
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
| F | 6 — OpenAI HTTP server smoke (`vllm serve` + `curl`) |
