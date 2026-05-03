# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU integration: multi-step mock-stack invariants with EngineCore draft-hook shim."""

from __future__ import annotations

from pathlib import Path

import pytest

from dllm_plugin.config import DRAFT_SIZE
from tests.gpu_memory import gpu_memory_utilization, kv_cache_memory_bytes
from tests.support.engine_core_draft_hook import patch_engine_core_draft_hook_semantics

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.dllm_gpu_integration


@pytest.fixture
def mock_llada2_model_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "mock_llada2_hf_config"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_mock_stack_multi_step_respects_max_tokens_with_engine_patch(
    monkeypatch: pytest.MonkeyPatch,
    mock_llada2_model_dir: Path,
) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    max_new = 5
    with patch_engine_core_draft_hook_semantics():
        llm = LLM(
            model=str(mock_llada2_model_dir),
            tokenizer=str(mock_llada2_model_dir),
            skip_tokenizer_init=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=128,
            max_num_seqs=1,
            gpu_memory_utilization=gpu_memory_utilization(),
            kv_cache_memory_bytes=kv_cache_memory_bytes(),
            load_format="dummy",
            scheduler_cls="dllm_plugin.Scheduler",
            worker_cls="dllm_plugin.Worker",
        )
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])],
            SamplingParams(
                max_tokens=max_new,
                temperature=0.0,
                detokenize=False,
            ),
        )

    toks = outputs[0].outputs[0].token_ids
    assert toks, "expected at least one generated token id"
    assert len(toks) <= max_new * DRAFT_SIZE + max_new, (
        "sanity bound: bounded cumulative ids under mock dLLM + greedy decode"
    )
