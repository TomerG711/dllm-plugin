# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for EngineCore draft-hook shim (issue #35, vLLM PR #36391)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.support.engine_core_draft_hook import (
    engine_core_draft_hook_patch_needed,
    patch_engine_core_draft_hook_semantics,
)

pytest.importorskip("vllm")

pytestmark = pytest.mark.dllm_engine_patch


def _make_engine_core_stub(
    *, use_spec_decode: bool, async_scheduling: bool = False
) -> MagicMock:
    stub = MagicMock()
    stub.use_spec_decode = use_spec_decode
    stub.async_scheduling = async_scheduling
    stub.model_executor = MagicMock()
    stub.scheduler = MagicMock()
    return stub


@pytest.fixture
def _draft_hook_semantics():
    """Apply PR #36391 semantics when the wheel still uses legacy spec-decode gates."""

    if engine_core_draft_hook_patch_needed():
        with patch_engine_core_draft_hook_semantics():
            yield
    else:
        yield


def test_post_step_calls_hook_without_spec_decode(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=False)
    draft_ids = MagicMock()
    stub.model_executor.take_draft_token_ids.return_value = draft_ids

    EngineCore.post_step(stub, model_executed=True)

    stub.model_executor.take_draft_token_ids.assert_called_once()
    stub.scheduler.update_draft_token_ids.assert_called_once_with(draft_ids)


def test_post_step_noop_when_draft_ids_none(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=False)
    stub.model_executor.take_draft_token_ids.return_value = None

    EngineCore.post_step(stub, model_executed=True)

    stub.model_executor.take_draft_token_ids.assert_called_once()
    stub.scheduler.update_draft_token_ids.assert_not_called()


def test_post_step_noop_when_model_not_executed(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=False)

    EngineCore.post_step(stub, model_executed=False)

    stub.model_executor.take_draft_token_ids.assert_not_called()


def test_post_step_noop_with_async_scheduling(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=False, async_scheduling=True)

    EngineCore.post_step(stub, model_executed=True)

    stub.model_executor.take_draft_token_ids.assert_not_called()


def test_post_step_still_works_with_spec_decode(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=True)
    draft_ids = MagicMock()
    stub.model_executor.take_draft_token_ids.return_value = draft_ids

    EngineCore.post_step(stub, model_executed=True)

    stub.model_executor.take_draft_token_ids.assert_called_once()
    stub.scheduler.update_draft_token_ids.assert_called_once_with(draft_ids)


def test_patch_compiles_when_legacy_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check that source rewrite succeeds for the supported layout."""

    monkeypatch.delenv("VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH", raising=False)
    if not engine_core_draft_hook_patch_needed():
        pytest.skip("installed vLLM already matches PR #36391 post_step layout")

    from tests.support import engine_core_draft_hook as mod

    compile_fn = mod._compile_patched_engine_core_methods
    post_fn, step_fn = compile_fn()
    assert callable(post_fn)
    assert callable(step_fn)


def test_skip_env_disables_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.engine.core import EngineCore

    monkeypatch.setenv("VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH", "1")
    stub = _make_engine_core_stub(use_spec_decode=False)
    draft_ids = MagicMock()
    stub.model_executor.take_draft_token_ids.return_value = draft_ids

    with patch_engine_core_draft_hook_semantics():
        EngineCore.post_step(stub, model_executed=True)

    if engine_core_draft_hook_patch_needed():
        stub.model_executor.take_draft_token_ids.assert_not_called()
    else:
        stub.model_executor.take_draft_token_ids.assert_called_once()
