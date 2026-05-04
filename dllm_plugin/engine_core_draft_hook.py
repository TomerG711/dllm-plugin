# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore draft-token hook alignment with vLLM PR #36391 (runtime + tests).

Stock PyPI vLLM in the ``0.20.x`` range may still gate
``take_draft_token_ids()`` / ``update_draft_token_ids*`` on ``use_spec_decode``.
This module applies a **string-fragile** source rewrite matching
https://github.com/vllm-project/vllm/pull/36391 until releases include that
behavior natively.

**Runtime (``vllm serve`` / engine process):** set
:data:`~dllm_plugin.config.DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK_ENV_VAR` so
:func:`register_dllm` calls :func:`apply_engine_core_draft_hook_patch_if_needed`
after plugin load (early enough because vLLM invokes ``load_general_plugins()``
at the start of ``EngineCore.__init__``).

**Tests:** use :func:`patch_engine_core_draft_hook_semantics` for a temporary
patch with teardown.

Disable all patching with ``VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH=1``.
Track wheel drift in https://github.com/vllm-project/dllm-plugin/issues/2 .
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager

_logger = logging.getLogger(__name__)

# Legacy v0.20.x ``post_step`` gates the hook on ``use_spec_decode``.
_LEGACY_POST_STEP_PATTERN = re.compile(
    r"not\s+self\.async_scheduling\s+and\s+self\.use_spec_decode\s+and\s+model_executed",
)

# Deferred structured-output branch in ``step_with_batch_queue`` (v0.20.x).
_LEGACY_DEFERRED_DRAFT_BLOCK = """        if self.use_spec_decode:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            assert draft_token_ids is not None
            # Update the draft token ids in the scheduler output to
            # filter out the invalid spec tokens, which will be padded
            # with -1 and skipped by the grammar bitmask computation.
            self.scheduler.update_draft_token_ids_in_output(
                draft_token_ids, deferred_scheduler_output
            )"""

_PATCHED_DEFERRED_DRAFT_BLOCK = """        draft_token_ids = (
            self.model_executor.take_draft_token_ids()
        )
        if draft_token_ids is not None:
            # Update the draft token ids in the scheduler output to
            # filter out the invalid spec tokens, which will be padded
            # with -1 and skipped by the grammar bitmask computation.
            self.scheduler.update_draft_token_ids_in_output(
                draft_token_ids, deferred_scheduler_output
            )"""

_runtime_patch_applied: bool = False


def _skip_patch_env_set() -> bool:
    raw = os.environ.get("VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH", "")
    return raw.lower() in ("1", "true", "yes", "on")


def engine_core_draft_hook_patch_needed() -> bool:
    """Return True if installed vLLM still uses the legacy spec-decode gate."""

    import inspect

    from vllm.v1.engine.core import EngineCore

    try:
        post_src = inspect.getsource(EngineCore.post_step)
    except (OSError, TypeError):
        return False
    return bool(_LEGACY_POST_STEP_PATTERN.search(post_src))


def _compile_patched_engine_core_methods() -> tuple[object, object]:
    """Build replacement functions by AST-safe string edit of upstream sources."""

    import inspect

    import vllm.v1.engine.core as ec_module
    from vllm.v1.engine.core import EngineCore

    ns: dict[str, object] = dict(vars(ec_module))

    post_src = textwrap.dedent(inspect.getsource(EngineCore.post_step))
    post_fixed = post_src.replace(
        "    if not self.async_scheduling and self.use_spec_decode and model_executed:",
        "    if not self.async_scheduling and model_executed:",
    )
    step_src = textwrap.dedent(inspect.getsource(EngineCore.step_with_batch_queue))
    if _LEGACY_DEFERRED_DRAFT_BLOCK not in step_src:
        msg = (
            "dllm_plugin.engine_core_draft_hook: deferred draft block not found "
            "in EngineCore.step_with_batch_queue; vLLM revision may have diverged."
        )
        raise RuntimeError(msg)
    step_fixed = step_src.replace(
        _LEGACY_DEFERRED_DRAFT_BLOCK,
        _PATCHED_DEFERRED_DRAFT_BLOCK,
    )

    if post_fixed == post_src:
        msg = (
            "engine_core_draft_hook: post_step guard string not found; "
            "vLLM revision may have diverged."
        )
        raise RuntimeError(msg)
    if step_fixed == step_src:
        raise RuntimeError("deferred draft block replace produced no change")

    exec(compile(post_fixed, ec_module.__file__, "exec"), ns)
    exec(compile(step_fixed, ec_module.__file__, "exec"), ns)
    post_fn = ns["post_step"]
    step_fn = ns["step_with_batch_queue"]
    return post_fn, step_fn


def apply_engine_core_draft_hook_patch_if_needed() -> None:
    """Apply permanent ``EngineCore`` patch in-process when a legacy wheel needs it.

    Idempotent: safe to call multiple times (e.g. from :func:`register_dllm`).

    Respects ``VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH`` (no-op when set).
    Does **not** require ``VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK`` — callers
    (``register_dllm``) should gate on that env before invoking.
    """

    global _runtime_patch_applied

    if _skip_patch_env_set():
        _logger.debug(
            "dLLM EngineCore draft-hook: skip env set; not applying runtime patch.",
        )
        return

    if _runtime_patch_applied:
        return

    if not engine_core_draft_hook_patch_needed():
        _logger.info(
            "dLLM EngineCore draft-hook: installed vLLM already matches PR #36391 "
            "semantics (no runtime string patch applied).",
        )
        _runtime_patch_applied = True
        return

    from vllm.v1.engine.core import EngineCore

    patched_post, patched_step = _compile_patched_engine_core_methods()
    EngineCore.post_step = patched_post
    EngineCore.step_with_batch_queue = patched_step
    _runtime_patch_applied = True
    _logger.warning(
        "dLLM EngineCore draft-hook: applied string-fragile runtime patch matching "
        "vLLM PR #36391 (temporary until your vLLM pin includes native hook "
        "behavior). Track pins and upstream in "
        "https://github.com/vllm-project/dllm-plugin/issues/2 — see also "
        "https://github.com/vllm-project/vllm/pull/36391",
    )


@contextmanager
def patch_engine_core_draft_hook_semantics() -> Iterator[None]:
    """Temporarily patch ``EngineCore`` post-step / batch-queue draft semantics."""

    if _skip_patch_env_set():
        yield
        return

    from vllm.v1.engine.core import EngineCore

    if not engine_core_draft_hook_patch_needed():
        yield
        return

    orig_post = EngineCore.post_step
    orig_step = EngineCore.step_with_batch_queue
    patched_post, patched_step = _compile_patched_engine_core_methods()
    try:
        EngineCore.post_step = patched_post
        EngineCore.step_with_batch_queue = patched_step
        yield
    finally:
        EngineCore.post_step = orig_post
        EngineCore.step_with_batch_queue = orig_step


def _reset_runtime_patch_applied_for_tests() -> None:
    """Reset idempotency flag (tests only; not part of the public API)."""

    global _runtime_patch_applied
    _runtime_patch_applied = False


__all__ = [
    "apply_engine_core_draft_hook_patch_if_needed",
    "engine_core_draft_hook_patch_needed",
    "patch_engine_core_draft_hook_semantics",
]
