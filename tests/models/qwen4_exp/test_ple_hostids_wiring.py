# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wiring tests for the PLE host-ids work (docs/d1-host-ids-design.md).

These cover the *plumbing* half of the design: the pinned mirror of each step's
draft tokens (``DraftTokensHandler`` with ``host_draft_staging=True``) and the
config-time refusal of the one scheduling mode that makes that mirror stale.
The numerical half -- that the host computes the same ngram ids the device kernel
would have -- is ``test_ple_hostids.py``.

Three properties are worth more here than a happy-path assertion, because each
one is a way the knob could be silently wrong or silently destructive:

* **The scheduler channel does not change.** Upstream reports ``[[-1] * k]`` to
  the scheduler unless the batch has structured-output requests. If staging
  changed that, a perf knob would be altering scheduling.
* **Row mapping travels with the values.** Consuming staged rows by batch
  position under a different mapping is a plausible ngram id for the wrong
  request.
* **Freshness is provable.** Two decode steps over an unchanged batch have the
  same row mapping and different values, so ``gen`` -- not the mapping -- is
  what tells a consumer whether it is holding this step's drafts.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.vllm import ENV_PLE_HOSTIDS, ple_hostids_async_conflict
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the staging path allocates a copy stream and a blocking event",
)


def _batch(num_reqs: int, structured: bool = False, mapping: list[int] | None = None):
    """The three ``InputBatch`` fields ``set_draft_tokens`` touches."""
    return SimpleNamespace(
        req_ids=[f"req{i}" for i in range(num_reqs)],
        has_structured_output_reqs=structured,
        idx_mapping_np=np.array(
            mapping if mapping is not None else range(num_reqs), dtype=np.int32
        ),
    )


# --------------------------------------------------------------------------
# The config-time refusal: async scheduling takes away the wait this depends on
# --------------------------------------------------------------------------


def test_async_conflict_refuses_knob_under_async():
    env = {ENV_PLE_HOSTIDS: "1"}
    msg = ple_hostids_async_conflict(True, environ=env)
    assert msg is not None
    # The message has to name both the knob and the way out, or the operator
    # reading a ValueError at boot cannot act on it.
    assert ENV_PLE_HOSTIDS in msg
    assert "--no-async-scheduling" in msg


def test_async_conflict_silent_when_sync_or_unset():
    # Red control for the test above: the refusal must track *both* inputs, so
    # a guard that always fires would pass the first test and fail this one.
    assert ple_hostids_async_conflict(False, environ={ENV_PLE_HOSTIDS: "1"}) is None
    assert ple_hostids_async_conflict(True, environ={}) is None
    assert ple_hostids_async_conflict(True, environ={ENV_PLE_HOSTIDS: "0"}) is None
    assert ple_hostids_async_conflict(False, environ={}) is None


def test_async_conflict_covers_every_nonzero_value():
    """Differential mode ("2") reads the same mirror, so async breaks it too.

    It would not produce wrong output -- the device path still answers -- but the
    comparison would run against a one-step-old row set and report mismatches
    that are the scheduler's doing, not the arithmetic's. A guard that matches
    only "1" leaves that mode unguarded.
    """
    from vllm.models.qwen4_exp.amd import ple_hostids

    assert ple_hostids_async_conflict(True, environ={ENV_PLE_HOSTIDS: "2"}) is not None
    assert ple_hostids_async_conflict(True, environ={ENV_PLE_HOSTIDS: "7"}) is not None
    assert ple_hostids_async_conflict(True, environ={ENV_PLE_HOSTIDS: ""}) is None
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(ENV_PLE_HOSTIDS, "2")
        assert ple_hostids.hostids_mode() == 2
        assert ple_hostids.hostids_enabled() is False, (
            "differential mode must not be treated as the authoritative path"
        )
    finally:
        monkey.undo()


def test_env_name_agrees_between_guard_and_worker():
    """The guard, the worker and the model must all read one environment name.

    They cannot share a function call (the worker deliberately avoids importing
    model code), so the name is the thing that can drift -- and a drifted name is
    a knob that silently never engages, which is indistinguishable from one that
    worked. This reads the sources instead of importing them so the test costs
    nothing and cannot itself perturb the import graph.
    """
    import inspect
    import pathlib

    from vllm.models.qwen4_exp.amd import ple_hostids
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    literal = f'"{ENV_PLE_HOSTIDS}"'
    runner_src = pathlib.Path(inspect.getfile(model_runner_module)).read_text()
    assert literal in runner_src, "worker reads a different name than the guard"
    # hostids_enabled() must answer to the same name, not a private copy.
    monkey = pytest.MonkeyPatch()
    try:
        monkey.delenv(ENV_PLE_HOSTIDS, raising=False)
        assert ple_hostids.hostids_enabled() is False
        monkey.setenv(ENV_PLE_HOSTIDS, "1")
        assert ple_hostids.hostids_enabled() is True
    finally:
        monkey.undo()


def test_accepted_values_agree_between_mode_and_worker():
    """``hostids_mode()`` and the worker must accept the same set of values.

    The worker cannot import model code (it would drag the model into every
    worker of every model), so it re-implements the read of the same variable. A
    value one side accepts and the other ignores is the worst drift there is:
    differential mode would boot with no draft mirror, refuse every step, and
    report a clean run.
    """
    import inspect
    import pathlib

    from vllm.models.qwen4_exp.amd import ple_hostids
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    src = pathlib.Path(inspect.getfile(model_runner_module)).read_text()
    assert 'in ("1", "2")' in src, (
        "the worker's accepted value set moved; update it and this test together"
    )

    monkey = pytest.MonkeyPatch()
    try:
        for raw, expected in (
            ("0", False),
            ("", False),
            ("1", True),
            ("2", True),
            ("3", False),
            ("yes", False),
        ):
            monkey.setenv(ENV_PLE_HOSTIDS, raw)
            assert (ple_hostids.hostids_mode() != 0) is expected, raw
    finally:
        monkey.undo()


# --------------------------------------------------------------------------
# Staging off: the default must be the known-good path, unchanged
# --------------------------------------------------------------------------


@requires_cuda
def test_staging_off_by_default():
    handler = DraftTokensHandler(None)
    assert handler.host_draft_staging is False
    assert handler.draft_tokens_host is None
    assert handler.host_draft_gen == 0

    handler.set_draft_tokens(_batch(2), torch.zeros(2, 3, dtype=torch.int32))
    values, row_idx, gen = handler.wait_host_draft_tokens()
    assert values is None and row_idx is None
    assert gen == 0, "off must mean nothing was staged, not something unstaged"
    assert handler.draft_tokens_host is None


@requires_cuda
def test_scheduler_channel_is_untouched_by_staging():
    """``get_draft_tokens()`` must not learn anything new from the knob.

    Upstream only hands the scheduler real drafts when the batch contains
    structured-output requests; otherwise it reports -1 for every slot. If
    enabling a host-side mirror changed that report, the knob would be changing
    scheduling behaviour, which is a different (and much larger) change.
    """
    drafts = torch.arange(6, dtype=torch.int32).reshape(2, 3)

    reports = {}
    for staging in (False, True):
        handler = DraftTokensHandler(None, host_draft_staging=staging)
        handler.set_draft_tokens(_batch(2, structured=False), drafts)
        out = handler.get_draft_tokens()
        reports[staging] = out.draft_token_ids
        assert out.req_ids == ["req0", "req1"]

    assert reports[False] == [[-1, -1, -1], [-1, -1, -1]]
    assert reports[True] == reports[False]


@requires_cuda
def test_structured_output_channel_still_works_with_staging_on():
    """The two channels are independent: both populated, both correct."""
    drafts = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    handler = DraftTokensHandler(None, host_draft_staging=True)
    handler.set_draft_tokens(_batch(2, structured=True), drafts.to("cuda"))
    staged, row_idx, gen = handler.wait_host_draft_tokens()
    assert staged[:2, :3].tolist() == drafts.numpy().tolist()
    assert row_idx.tolist() == [0, 1]
    assert handler.get_draft_tokens().draft_token_ids == drafts.cpu().tolist()
    assert gen == 1


# --------------------------------------------------------------------------
# Staging on: values, mapping, buffer reuse, and the freshness counter
# --------------------------------------------------------------------------


@requires_cuda
def test_cpu_staging_values_mapping_and_gen():
    handler = DraftTokensHandler(None, host_draft_staging=True)
    drafts = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int32)
    handler.set_draft_tokens(_batch(2, mapping=[5, 2]), drafts)

    staged, row_idx, gen = handler.wait_host_draft_tokens()
    assert gen == 1
    assert staged[:2, :3].tolist() == [[7, 8, 9], [10, 11, 12]]
    assert row_idx.tolist() == [5, 2], "row order must follow idx_mapping, not position"

    # Waiting twice must be idempotent and must not invent a new generation:
    # a consumer that sees the same gen twice has to take the device path.
    again = handler.wait_host_draft_tokens()
    assert again[2] == gen
    assert again[0][:2, :3].tolist() == staged[:2, :3].tolist()


@requires_cuda
def test_gen_only_advances_when_something_was_staged():
    handler = DraftTokensHandler(None, host_draft_staging=True)
    handler.set_draft_tokens(_batch(1), torch.tensor([[1, 2]], dtype=torch.int32))
    _, _, gen1 = handler.wait_host_draft_tokens()
    assert gen1 == 1
    # A step where the drafter did not run calls nothing here, so the gen sits
    # still and the next consumer refuses. This is the assertion that makes the
    # consumer's staleness check possible at all.
    assert handler.wait_host_draft_tokens()[2] == gen1


@requires_cuda
def test_buffer_is_reused_then_grown():
    handler = DraftTokensHandler(None, host_draft_staging=True)
    handler.set_draft_tokens(_batch(2), torch.zeros(2, 3, dtype=torch.int32))
    buf = handler.draft_tokens_host
    assert buf is not None and buf.is_pinned()

    # Smaller request: same buffer, no new pinned allocation.
    handler.set_draft_tokens(_batch(1), torch.ones(1, 2, dtype=torch.int32))
    assert handler.draft_tokens_host is buf
    staged, _, gen = handler.wait_host_draft_tokens()
    assert gen == 2
    assert staged[0, :2].tolist() == [1, 1]

    # Larger request: grown, and the previously staged rows are gone from it.
    big = torch.full((4, 5), 9, dtype=torch.int32)
    handler.set_draft_tokens(_batch(4), big)
    assert handler.draft_tokens_host.shape >= (4, 5)
    staged, row_idx, gen = handler.wait_host_draft_tokens()
    assert gen == 3
    assert staged[:4, :5].tolist() == big.numpy().tolist()
    assert row_idx.tolist() == [0, 1, 2, 3]


@requires_cuda
def test_staging_dtype_follows_source_without_casting():
    """A cast would be a device kernel per step to achieve nothing."""
    handler = DraftTokensHandler(None, host_draft_staging=True)
    handler.set_draft_tokens(_batch(1), torch.tensor([[1, 2, 3]], dtype=torch.int32))
    assert handler.draft_tokens_host.dtype == torch.int32
    handler.set_draft_tokens(_batch(1), torch.tensor([[1, 2, 3]], dtype=torch.int64))
    # Different dtype means a new buffer, not a cast into the old one.
    assert handler.draft_tokens_host.dtype == torch.int64
    staged, _, _ = handler.wait_host_draft_tokens()
    assert staged[:1, :3].tolist() == [[1, 2, 3]]


@requires_cuda
def test_gpu_staging_lands_before_wait_returns():
    """The real path: D2H on the copy stream, visible after the event wait."""
    handler = DraftTokensHandler(None, host_draft_staging=True)
    drafts = torch.arange(12, dtype=torch.int32).reshape(3, 4).cuda()
    handler.set_draft_tokens(_batch(3, mapping=[9, 4, 77]), drafts)
    staged, row_idx, gen = handler.wait_host_draft_tokens()
    assert staged[:3, :4].tolist() == drafts.cpu().tolist()
    assert row_idx.tolist() == [9, 4, 77]
    assert gen == 1

    # Second step, same shape: the buffer must carry the new values, proving the
    # copy is re-issued rather than the first snapshot being reused.
    handler.set_draft_tokens(_batch(3, mapping=[9, 4, 77]), drafts * 100)
    staged, _, gen = handler.wait_host_draft_tokens()
    assert gen == 2
    assert staged[:3, :4].tolist() == (drafts * 100).cpu().tolist()
