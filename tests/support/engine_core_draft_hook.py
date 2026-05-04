# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility re-export for :mod:`dllm_plugin.engine_core_draft_hook`.

Prefer importing from ``dllm_plugin.engine_core_draft_hook`` in new code.
"""

from __future__ import annotations

from dllm_plugin.engine_core_draft_hook import (
    apply_engine_core_draft_hook_patch_if_needed,
    engine_core_draft_hook_patch_needed,
    patch_engine_core_draft_hook_semantics,
)

__all__ = [
    "apply_engine_core_draft_hook_patch_if_needed",
    "engine_core_draft_hook_patch_needed",
    "patch_engine_core_draft_hook_semantics",
]
