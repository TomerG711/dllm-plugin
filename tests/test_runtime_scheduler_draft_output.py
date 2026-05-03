# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Component tests for ``DllmRuntimeScheduler`` draft output padding (issue #35)."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm")

from vllm.v1.outputs import DraftTokenIds

from dllm_plugin.config import DRAFT_SIZE
from dllm_plugin.runtime_scheduler import DllmRuntimeScheduler
from dllm_plugin.scheduler import DllmScheduler

pytestmark = pytest.mark.dllm_engine_patch


def _req() -> MagicMock:
    r = MagicMock()
    r.is_finished.return_value = False
    r.is_prefill_chunk = False
    return r


def test_update_draft_token_ids_in_output_pads_short_draft_with_neg_one() -> None:
    host = SimpleNamespace(
        requests={"r1": _req()},
        _dllm_helper=DllmScheduler(),
    )
    placeholder = [10, 11, 12, 13]
    sched = SimpleNamespace(scheduled_spec_decode_tokens={"r1": list(placeholder)})
    draft = DraftTokenIds(req_ids=["r1"], draft_token_ids=[[7, 8]])

    DllmRuntimeScheduler.update_draft_token_ids_in_output(
        cast(Any, host),
        draft,
        sched,
    )

    assert sched.scheduled_spec_decode_tokens["r1"] == [7, 8, -1, -1]
    assert sched.num_invalid_spec_tokens == {}


def test_update_draft_token_ids_in_output_truncates_long_draft() -> None:
    host = SimpleNamespace(
        requests={"r1": _req()},
        _dllm_helper=DllmScheduler(),
    )
    placeholder = [1, 2, 3]
    sched = SimpleNamespace(scheduled_spec_decode_tokens={"r1": list(placeholder)})
    long_row = list(range(10))
    draft = DraftTokenIds(req_ids=["r1"], draft_token_ids=[long_row])

    DllmRuntimeScheduler.update_draft_token_ids_in_output(
        cast(Any, host),
        draft,
        sched,
    )

    assert sched.scheduled_spec_decode_tokens["r1"] == [0, 1, 2]


def test_validate_draft_lengths_rejects_wrong_block_width() -> None:
    host = SimpleNamespace(_dllm_helper=DllmScheduler())
    bad = DraftTokenIds(req_ids=["r1"], draft_token_ids=[[1] * (DRAFT_SIZE + 1)])
    with pytest.raises(ValueError, match="draft token block length mismatch"):
        DllmRuntimeScheduler._validate_draft_lengths(cast(Any, host), bad)


def test_update_draft_token_ids_rejects_wrong_block_width() -> None:
    """``update_draft_token_ids`` rejects drafts via ``_validate_draft_lengths``."""

    host = SimpleNamespace(
        requests={"r1": _req()},
        _dllm_helper=DllmScheduler(),
    )
    host._validate_draft_lengths = MethodType(
        DllmRuntimeScheduler._validate_draft_lengths,
        host,
    )
    bad = DraftTokenIds(req_ids=["r1"], draft_token_ids=[[7, 8]])
    with pytest.raises(ValueError, match="draft token block length mismatch"):
        DllmRuntimeScheduler.update_draft_token_ids(cast(Any, host), bad)


def test_update_draft_token_ids_in_output_clears_num_invalid_spec_tokens() -> None:
    host = SimpleNamespace(
        requests={"r1": _req()},
        _dllm_helper=DllmScheduler(),
    )
    placeholder = [10, 11, 12, 13]
    sched = SimpleNamespace(
        scheduled_spec_decode_tokens={"r1": list(placeholder)},
        num_invalid_spec_tokens={"r1": 99},
    )
    draft = DraftTokenIds(req_ids=["r1"], draft_token_ids=[[1, 2, 3, 4]])

    DllmRuntimeScheduler.update_draft_token_ids_in_output(
        cast(Any, host),
        draft,
        sched,
    )

    assert sched.num_invalid_spec_tokens == {}
