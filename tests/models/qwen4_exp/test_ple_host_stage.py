# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-staged PLE ids: the producer, the consumer, and the differential gate.

E1 Phase 1 (P1.3 + P1.4). A captured FULL decode graph replays the H2D out of
``_pinned_output`` and contains neither the blocking device->host copy of the
n-gram ids nor the host gather that produced them, so the host has to fill that
buffer itself, every step, from ids it obtained without asking the device. These
tests cover the three new behaviours that claim makes true:

* ``stage_host_step`` -- compute ids on the host, gather, publish the row count;
* the custom op skipping its D2H and gather when the row count agrees;
* ``compare_host_ids`` -- differential mode, the only thing that can validate the
  position arithmetic against the device's own answer.

Construction note: a real ``Qwen4ExpNGramEmbedding`` wants a model config, a
VllmConfig and mmap'd shards totalling tens of GiB, and a real
``Qwen4ExpModelState`` wants an engine. Both are exercised here as unbound
methods against a stand-in ``self`` that carries exactly the attributes the
methods touch. That is not a shortcut around the code under test -- the method
bodies are the real ones (assigned from the class), so a change in them shows up
here; it only removes the machinery that cannot go wrong in a different way.
Every attribute the fake carries is one the production ``__init__`` sets.
"""

from __future__ import annotations

import contextlib
import pathlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.amd import model_state as model_state_module
from vllm.models.qwen4_exp.amd import ple_hostids
from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.amd.ple_hostids import HostTokenHistory
from vllm.models.qwen4_exp.amd.ple_layer import (  # noqa: F401
    Qwen4ExpNGramEmbedding,
    qwen4_exp_amd_ple_ngram_embedding,
)

NGRAM_SIZE = 3
HEADS_PER_NGRAM = 1
HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM  # 2
EMBED_DIM = 8
HEAD_DIM = EMBED_DIM // HEADS  # 4
ROWS = 6
CAPACITY = 8
SENTINEL = -12345
EOS = 0
CTX = NGRAM_SIZE - 1


def _layout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fixed, plausible hash layout: 3 multipliers, one range per head."""
    multipliers = np.array([-12345, 67891, -3333], dtype=np.int64)
    sizes = np.array([1000003, 2000003], dtype=np.int64)
    offsets = (np.cumsum(sizes) - sizes).astype(np.int64)
    return multipliers, sizes, offsets


class FakeEvent:
    """Stand-in for ``torch.cuda.Event`` whose readiness the test owns.

    ``synchronize`` raises on purpose: "the tripwire never waits" is then asserted
    by every test that uses the event, not only by the one written to check it.
    """

    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.records = 0
        self.queries = 0

    def record(self) -> None:
        self.records += 1

    def query(self) -> bool:
        self.queries += 1
        return self.ready

    def synchronize(self) -> None:  # pragma: no cover - must never run
        raise AssertionError("the generation tripwire synchronised on the device")


class StagingEmbedding:
    """Stand-in ``self`` for the real staging methods of the embedding module.

    ``ngram_embedding`` is a recording gather so "did the host gather run" is
    observable, and the pinned buffers are ordinary (unpinned) CPU tensors with
    the production shapes and dtypes; pinning is a DMA property and nothing here
    does DMA.
    """

    # The real implementations, not copies.
    _host_hash_buffers = Qwen4ExpNGramEmbedding._host_hash_buffers
    _stage_step_impl = Qwen4ExpNGramEmbedding.stage_host_step
    take_host_stage = Qwen4ExpNGramEmbedding.take_host_stage

    #: Whether a staged step also puts its generation where the op would have put
    #: it. See :meth:`stage_host_step` below.
    _mirror_op_gen = True

    def stage_host_step(self, *args: Any, **kwargs: Any) -> bool:
        """The real staging method, plus the one thing only the custom op does.

        The staged branch of the op copies this step's generation into the device
        scratch on every staged step, eager and replayed alike
        (``ple_layer.py:2806``), and the generation tripwire reads that scratch back
        later. A test that drives ``stage_host_step`` never runs the op, so the
        scratch would stay at zero and the tripwire would correctly report a handoff
        that never landed -- a true statement about the fake, arriving as a layer
        disabled in the middle of a test about something else. Boot #27 is what that
        looks like from the outside: a ``staged_steps`` counter that climbed while
        the layer underneath it had stopped taking steps.

        A test that wants the wire to report something -- stale, ahead, in flight --
        says so itself with :func:`_replay`, which turns this mirror off.
        """
        accepted = self._stage_step_impl(*args, **kwargs)
        if accepted and self._mirror_op_gen and self._gen_device is not None:
            self._gen_device.copy_(self._pinned_gen)
        return accepted
    compare_host_ids = Qwen4ExpNGramEmbedding.compare_host_ids
    note_draft_provenance = Qwen4ExpNGramEmbedding.note_draft_provenance
    _id_dump = Qwen4ExpNGramEmbedding._id_dump
    disable_hostids = Qwen4ExpNGramEmbedding.disable_hostids
    _alloc_gen_scratch = Qwen4ExpNGramEmbedding._alloc_gen_scratch
    _gen_scratch = Qwen4ExpNGramEmbedding._gen_scratch
    check_host_gen = Qwen4ExpNGramEmbedding.check_host_gen
    _fire_gen = Qwen4ExpNGramEmbedding._fire_gen
    _inputs_agree = Qwen4ExpNGramEmbedding._inputs_agree
    stage_inputs_agree = Qwen4ExpNGramEmbedding.stage_inputs_agree
    _note_stage_refusal = Qwen4ExpNGramEmbedding._note_stage_refusal
    _ctx_leg = Qwen4ExpNGramEmbedding._ctx_leg
    _packed_dump = Qwen4ExpNGramEmbedding._packed_dump
    _consumed_report = Qwen4ExpNGramEmbedding._consumed_report
    _note_input_refusal = Qwen4ExpNGramEmbedding._note_input_refusal

    def __init__(self, rows: int = ROWS, capacity: int = CAPACITY) -> None:
        multipliers, sizes, offsets = _layout()
        self.lookup_calls = 0
        self.ngram_size = NGRAM_SIZE
        self.heads_per_ngram = HEADS_PER_NGRAM
        self.eos_token_id = EOS
        self.layer_name = "ple"
        self.runtime_dtype = torch.float16
        self.layer_multipliers = torch.from_numpy(multipliers)
        self.ngram_heads_vocab_sizes = torch.from_numpy(sizes)
        self.ngram_heads_offsets = torch.from_numpy(offsets)
        self._pinned_ngram_ids = torch.full(
            (capacity, HEADS), SENTINEL, dtype=torch.long
        )
        self._pinned_output = torch.zeros(
            (capacity, EMBED_DIM), dtype=self.runtime_dtype
        )
        self._host_stage_rows = -1
        self._host_check_rows = -1
        self._host_check_ids = None
        self._host_hash_cache = None
        self._hostids_disabled = False
        self._host_stage_steps = 0
        self._host_id_mismatches = 0
        self._host_id_matches = 0
        self._host_check_inputs: tuple[Any, ...] | None = None
        # The compute's own packed workspace, as the real layer holds it: written
        # only by _compute_ngram_ids, which P1.12 reads at dump time.
        self.padded_buffer = torch.full((4, 64), EOS, dtype=torch.long)
        self._host_check_draft: tuple[Any, Any, int] | None = None
        self._host_check_dev_inputs: Any = None
        # P1.14: the qsl view the forward is handed. The stand-in has to carry it
        # because the per-step report reads it, and a stand-in whose log lines
        # cannot be rendered is a stand-in whose tests are not testing the line.
        self._host_consumed_qsl: Any = None
        self._rows = rows
        # Generation tripwire state, mirroring the production __init__. The event
        # is a FakeEvent so `query()` is controllable and a synchronize() is a
        # test failure rather than a wait.
        self._host_stage_gen = 0
        self._pinned_gen = torch.zeros(1, dtype=torch.long)
        self._pinned_gen_read = torch.zeros(1, dtype=torch.long)
        self._gen_device: torch.Tensor | None = torch.zeros(1, dtype=torch.long)
        self._gen_event = FakeEvent()
        self._gen_pending_expected = 0
        self._gen_check_countdown = 1
        self._gen_stuck = 0
        self._gen_checks_ok = 0
        self._gen_trips = 0
        self._staged_branch_captured = False
        # Input-agreement gate (P1.10b) counters, mirroring the production __init__.
        # Nothing else from P1.9 survives: the op-entry stamps and the input snapshot
        # were both observers of a region the AOT cache decides how to run.
        self._host_input_checks = 0
        self._host_input_skips = 0
        self._host_stage_refusals = 0
        # P1.11: steps where the device window was never handed over, so the third
        # leg stayed dark. Mirrors the production __init__ for the same reason the
        # other counters do -- the gate prints it, so a stand-in without it is a
        # stand-in whose log lines cannot be rendered.
        self._host_ctx_unchecked = 0

    @property
    def ngram_embedding(self):  # noqa: ANN201
        def _gather(ids: torch.Tensor) -> torch.Tensor:
            self.lookup_calls += 1
            # Deterministic function of the ids, so a wrong buffer is visible.
            out = torch.zeros((ids.shape[0], HEAD_DIM, HEADS), dtype=torch.float32)
            out[...] = ids.abs().sum(dim=-1).reshape(-1, 1, 1)
            return out

        return _gather


def _step_inputs(num_reqs: int = 2, q: int = 3) -> tuple:
    """Pure-decode step: ``q`` rows per request, all query_len == q."""
    input_ids = np.arange(num_reqs * q, dtype=np.int64) + 100
    query_start_loc = np.arange(0, num_reqs * q + 1, q, dtype=np.int64)
    ngram_context = np.tile(
        (np.arange(num_reqs, dtype=np.int64) + 7).reshape(-1, 1), (1, HEADS)
    )
    return input_ids, query_start_loc, ngram_context


def _expected_ids(input_ids, query_start_loc, ngram_context) -> np.ndarray:
    multipliers, sizes, offsets = _layout()
    return ple_hostids.compute_ngram_ids_host(
        input_ids,
        query_start_loc,
        ngram_context,
        multipliers=multipliers,
        vocab_sizes=sizes,
        head_offsets=offsets,
        eos_token_id=EOS,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
    )


# --------------------------------------------------------------------------
# stage_host_step: the producer's per-layer half
# --------------------------------------------------------------------------


def test_stage_host_step_fills_ids_and_gathered_rows() -> None:
    emb = StagingEmbedding()
    inp, qsl, ctx = _step_inputs()
    assert emb.stage_host_step(inp, qsl, ctx, gather=True) is True

    ids = _expected_ids(inp, qsl, ctx)
    assert emb._host_stage_rows == ROWS
    # The ids really went into the buffer the graph's H2D does not read but the
    # gather does -- proving the pinned write happened, not just the flag.
    assert np.array_equal(emb._pinned_ngram_ids[:ROWS].numpy(), ids)
    assert emb.lookup_calls == 1, "the host gather did not run"
    # Row content is the gather's output in the runtime dtype, per row.
    expect = (
        torch.from_numpy(ids)
        .abs()
        .sum(dim=-1, keepdim=True)
        .expand(-1, EMBED_DIM)
        .to(torch.float16)
    )
    assert torch.equal(emb._pinned_output[:ROWS], expect)
    assert emb._host_stage_steps == 1


def test_stage_host_step_check_mode_leaves_the_device_path_untouched() -> None:
    emb = StagingEmbedding()
    inp, qsl, ctx = _step_inputs()
    assert emb.stage_host_step(inp, qsl, ctx, gather=False) is True

    assert emb._host_check_rows == ROWS
    assert emb._host_stage_rows == -1, "check mode must not arm the skip"
    assert emb.lookup_calls == 0, "check mode must not gather"
    assert torch.all(emb._pinned_ngram_ids == SENTINEL), (
        "check mode wrote the device's id buffer"
    )
    assert torch.all(emb._pinned_output == 0)
    assert emb._host_check_ids is not None
    assert np.array_equal(
        emb._host_check_ids[:ROWS].numpy(), _expected_ids(inp, qsl, ctx)
    )


def test_disabled_embedding_stages_nothing() -> None:
    emb = StagingEmbedding()
    emb._hostids_disabled = True
    inp, qsl, ctx = _step_inputs()
    assert emb.stage_host_step(inp, qsl, ctx, gather=True) is False
    assert emb._host_stage_rows == -1
    assert emb.lookup_calls == 0


def test_more_rows_than_the_pinned_buffer_refuses() -> None:
    """A padded batch cannot be staged: the buffer is max_num_batched_tokens."""
    emb = StagingEmbedding(capacity=4)
    inp, qsl, ctx = _step_inputs(num_reqs=4, q=3)  # 12 rows > 4
    assert emb.stage_host_step(inp, qsl, ctx, gather=True) is False
    assert emb._host_stage_rows == -1
    assert emb.lookup_calls == 0, "refused late: it gathered anyway"


def test_hash_buffers_are_read_once() -> None:
    """A per-step ``.cpu()`` would be a synchronising D2H on the hot path."""
    emb = StagingEmbedding()
    first = emb._host_hash_buffers()
    assert emb._host_hash_cache is not None
    assert emb._host_hash_buffers() is first, "snapshot is not cached"
    # And invalidation (what load_weights does) makes it re-read.
    emb._host_hash_cache = None
    second = emb._host_hash_buffers()
    assert second is not first
    assert np.array_equal(second[0], first[0])


def test_load_weights_drops_the_hash_cache() -> None:
    """The cache must not outlive a reload of the weights it read.

    Checked against source because load_weights needs real checkpoint tensors:
    what can rot is the *absence* of the invalidation, and a missing line is
    exactly what a source assertion is good at.
    """
    src = (
        pathlib.Path(__file__).resolve().parents[3]
        / "vllm/models/qwen4_exp/amd/ple_layer.py"
    ).read_text()
    assert "def load_weights" in src
    after = src[src.index("def load_weights") :]
    assert "self._host_hash_cache = None" in after, (
        "load_weights no longer drops the host hash-cache snapshot"
    )


# --------------------------------------------------------------------------
# take_host_stage / compare_host_ids: the rules the op relies on
# --------------------------------------------------------------------------


def test_take_host_stage_requires_exact_row_agreement() -> None:
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS
    assert emb.take_host_stage(ROWS) is True
    # Consumed: a second step must not inherit the previous step's answer.
    assert emb._host_stage_rows == -1
    assert emb.take_host_stage(ROWS) is False

    emb._host_stage_rows = ROWS
    assert emb.take_host_stage(ROWS - 1) is False, (
        "row count disagreed and the flag was still trusted"
    )
    assert emb._host_stage_rows == -1, "a stale flag survived"


def test_compare_disables_on_the_first_mismatch() -> None:
    emb = StagingEmbedding()
    ids = np.full((ROWS, HEADS), 42, dtype=np.int64)
    emb._host_check_ids = torch.from_numpy(ids.copy())
    emb._host_check_rows = ROWS
    device = torch.from_numpy(np.full((ROWS, HEADS), 43, dtype=np.int64))
    # Input agreement (P1.10b): these tests hand the compare an ids buffer directly,
    # so they have to state the tokens the compute read as well. Without that the
    # gate refuses, which is what it exists to do, and the test would be asserting
    # nothing about mismatch handling.
    _agree_inputs(emb)

    emb.compare_host_ids(device, emb._host_check_dev_inputs)

    assert emb._hostids_disabled is True
    assert emb._host_id_mismatches == 1
    assert emb._host_check_rows == -1, "the check was not consumed"


def test_compare_passes_and_is_consumed_on_agreement() -> None:
    emb = StagingEmbedding()
    ids = np.full((ROWS, HEADS), 42, dtype=np.int64)
    emb._host_check_ids = torch.from_numpy(ids.copy())
    emb._host_check_rows = ROWS
    # Input agreement (P1.10b), as in the mismatch test above.
    _agree_inputs(emb)
    emb.compare_host_ids(torch.from_numpy(ids.copy()), emb._host_check_dev_inputs)
    assert emb._host_input_checks == 1
    assert emb._hostids_disabled is False
    assert emb._host_id_mismatches == 0
    assert emb._host_check_rows == -1


def test_compare_ignores_a_row_count_disagreement() -> None:
    """A check left over from a step that never reached the op is not a mismatch.

    Row counts differ legitimately: the op runs per forward with the padded-or-
    real token count of that forward, and the producer staged for a different
    step. Declaring that a mismatch would disable the knob on the first padded
    batch, which is a much more common event than an arithmetic error.
    """
    emb = StagingEmbedding()
    emb._host_check_ids = torch.zeros((CAPACITY, HEADS), dtype=torch.long)
    emb._host_check_rows = ROWS
    emb.compare_host_ids(torch.zeros((ROWS - 1, HEADS), dtype=torch.long))
    assert emb._hostids_disabled is False
    assert emb._host_id_mismatches == 0
    assert emb._host_check_rows == -1
    # The row-count gate runs first, so the input gate never speaks and no inputs are
    # even read: a step the host did not build costs the harness nothing.
    assert emb._host_input_skips == 0
    assert emb._host_input_checks == 0


def test_disable_is_logged_once_and_idempotent() -> None:
    emb = StagingEmbedding()
    emb.disable_hostids("first reason")
    emb.disable_hostids("second reason")
    assert emb._hostids_disabled is True


# --------------------------------------------------------------------------
# the custom op: skip the D2H, keep the H2D
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _op_env(emb, monkeypatch, *, capturing: bool = False):
    layer = ple_layer_module.Qwen4ExpPLELayer.__new__(ple_layer_module.Qwen4ExpPLELayer)
    layer.ple_embedding = emb  # type: ignore[attr-defined]
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    monkeypatch.delenv("VLLM_GFX906_PLE_CAPTURE_PROBE", raising=False)
    monkeypatch.delenv("VLLM_GFX906_PLE_PROF", raising=False)
    real = torch.cuda.is_current_stream_capturing
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    try:
        yield emb
    finally:
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", real)


def _run_op(n: int = ROWS) -> torch.Tensor:
    ids = torch.arange(n * HEADS, dtype=torch.long).reshape(n, HEADS)
    output = torch.full((n, EMBED_DIM), float("nan"), dtype=torch.float16)
    qwen4_exp_amd_ple_ngram_embedding(ids, output, "ple")
    return output


def test_op_skips_d2h_and_gather_when_the_host_staged_this_step(monkeypatch):
    emb = StagingEmbedding()
    staged = torch.full((ROWS, EMBED_DIM), 3.5, dtype=torch.float16)
    emb._pinned_output[:ROWS] = staged
    emb._host_stage_rows = ROWS

    with _op_env(emb, monkeypatch) as emb:
        out = _run_op()

    assert emb.lookup_calls == 0, "the host gather ran although the host staged"
    # No D2H: the id buffer still holds the sentinel it was built with, so the
    # device's ids never landed. This is the wait the whole change removes.
    assert torch.all(emb._pinned_ngram_ids == SENTINEL), "the D2H ran"
    assert torch.equal(out, staged), "the H2D did not deliver the staged rows"
    assert emb._host_stage_rows == -1


def test_op_runs_the_device_path_when_row_counts_disagree(monkeypatch):
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS - 1

    with _op_env(emb, monkeypatch) as emb:
        out = _run_op()

    assert emb.lookup_calls == 1, "the device path was skipped on a disagreement"
    assert not torch.all(out == 0)
    assert emb._host_stage_rows == -1, "a rejected flag survived to the next step"


def test_a_rejected_flag_does_not_skip_the_next_step(monkeypatch):
    """The failure mode being guarded: stage once, skip forever."""
    emb = StagingEmbedding()
    emb._host_stage_rows = 1  # staged for a different step than the op sees

    with _op_env(emb, monkeypatch) as emb:
        _run_op()
        _run_op()

    assert emb.lookup_calls == 2, "a stale flag skipped a later step's gather"


def test_op_compares_after_the_d2h_in_check_mode(monkeypatch):
    emb = StagingEmbedding()
    ids = torch.arange(ROWS * HEADS, dtype=torch.long).reshape(ROWS, HEADS)
    emb._host_check_ids = ids.clone()
    emb._host_check_rows = ROWS
    # The op hands the compare whatever ``_host_check_dev_inputs`` holds, so the
    # agreeing case has to be staged for this test to be about the comparison rather
    # than about the refusal (which the tests below cover on their own).
    _agree_inputs(emb)

    with _op_env(emb, monkeypatch) as emb:
        _run_op()

    assert emb.lookup_calls == 1, "check mode must not change what runs"
    assert emb._hostids_disabled is False, "a matching check disabled the knob"
    assert emb._host_id_mismatches == 0
    assert emb._host_input_checks == 1, "the op never got as far as the comparison"


def test_op_disables_after_a_check_mismatch(monkeypatch):
    emb = StagingEmbedding()
    emb._host_check_ids = torch.full((ROWS, HEADS), -7, dtype=torch.long)
    emb._host_check_rows = ROWS
    # Input agreement (P1.10b): see the two compare tests above.
    _agree_inputs(emb)

    with _op_env(emb, monkeypatch) as emb:
        _run_op()

    assert emb.lookup_calls == 1, "the device path must still produce the answer"
    assert emb._hostids_disabled is True
    assert emb._host_id_mismatches == 1


def test_host_stage_wins_over_check_mode(monkeypatch):
    """Mode 1 stages and skips; nothing was compared, so nothing can disable."""
    emb = StagingEmbedding()
    emb._pinned_output[:ROWS] = 2.0
    emb._host_stage_rows = ROWS
    emb._host_check_rows = ROWS
    emb._host_check_ids = torch.zeros((ROWS, HEADS), dtype=torch.long)

    with _op_env(emb, monkeypatch) as emb:
        _run_op()

    assert emb.lookup_calls == 0
    assert emb._hostids_disabled is False, "the skipped path compared nothing"


# --------------------------------------------------------------------------
# the producer in Qwen4ExpModelState: what decides coverage
# --------------------------------------------------------------------------


class FakeModelState:
    """Stand-in ``self`` for the real producer methods."""

    _stage_host_ple_ids = model_state_module.Qwen4ExpModelState._stage_host_ple_ids
    set_host_draft_source = model_state_module.Qwen4ExpModelState.set_host_draft_source
    stage_host_tokens = model_state_module.Qwen4ExpModelState.stage_host_tokens
    disable_host_staging = model_state_module.Qwen4ExpModelState.disable_host_staging
    _note_host_refusal = model_state_module.Qwen4ExpModelState._note_host_refusal
    _staged_graph_in_play = model_state_module.Qwen4ExpModelState._staged_graph_in_play
    _ple_host_coverage = model_state_module.Qwen4ExpModelState._ple_host_coverage
    _ple_resync_mirror = model_state_module.Qwen4ExpModelState._ple_resync_mirror
    _ple_resync_mirror_inner = (
        model_state_module.Qwen4ExpModelState._ple_resync_mirror_inner
    )
    _note_resync_disagreement = (
        model_state_module.Qwen4ExpModelState._note_resync_disagreement
    )
    _ple_host_layer_steps = model_state_module.Qwen4ExpModelState._ple_host_layer_steps
    # The budgets are read off the real class rather than restated here, so a
    # cap changed in production changes what these tests assert.
    PLE_RESYNC_PER_SLOT = model_state_module.Qwen4ExpModelState.PLE_RESYNC_PER_SLOT
    PLE_RESYNC_TOTAL = model_state_module.Qwen4ExpModelState.PLE_RESYNC_TOTAL
    PLE_DISAGREE_TOTAL = model_state_module.Qwen4ExpModelState.PLE_DISAGREE_TOTAL
    PLE_DISAGREE_PER_SLOT = model_state_module.Qwen4ExpModelState.PLE_DISAGREE_PER_SLOT

    def __init__(self, slots: int = 8) -> None:
        self.uses_ngram_embedding = True
        self.ngram_context_len = CTX
        self.ngram_eos_token_id = EOS
        self._host_drafts = None
        self._ple_host_layers: list | None = None
        self._ple_host_gen_layers: list = []
        self._ple_host_refusals: dict[str, int] = {}
        self._ple_host_refusal_logged: set[str] = set()
        self._ple_host_steps = 0
        # E1 Phase 1: the captured-graph refusal tracking the real class does. The
        # stub borrows ``_note_host_refusal``, so it has to carry the state that
        # method reads -- which is the point of borrowing rather than copying.
        self._ple_host_captured = False
        self._ple_host_captured_refusals = 0
        self._last_draft_gen = -1
        self._ple_host_tokens = None
        # The real class builds this in __init__ and seeds it at add time, so the
        # stub has to as well: an unseeded window is a refused step, and these
        # tests are about which steps get through.
        self._ple_host_history = HostTokenHistory(slots, CTX + 1)
        for slot in range(slots):
            self._ple_host_history.seed(slot, _slot_tokens(slot)[: _POSITION + 1])
        # Repair state, because ``_stage_host_ple_ids`` now reaches for it on a
        # count mismatch. The stub has no device, so the sync is a counter: the
        # repair's correctness does not depend on it, only its ordering does, and
        # the resync tests assert that it was called.
        self._ple_resyncs = 0
        self._ple_resync_disagree = 0
        self._ple_resync_refused: set[str] = set()
        self._ple_resync_slots: dict[int, int] = {}
        # Boot #27's lesson, carried as state: a cross-check is only meaningful
        # against the request that owns the slot, so the stub has to know owners,
        # and it has to match the mapping ``_batch_and_states`` puts in
        # ``req_states.index_to_req_id``.
        self._ple_host_slot_req = {slot: _req_id(slot) for slot in range(slots)}
        self._ple_resync_disagree_slots: dict[int, int] = {}
        self._ple_resync_owner_skip = 0
        self._ple_syncs = 0
        self._ple_resync_sync = self._count_sync
        self.slots = slots

    def _count_sync(self) -> None:
        self._ple_syncs += 1

    def staged_rows(self) -> list[int]:
        return [layer._host_stage_rows for layer in self._ple_host_layers or []]


class FakeDrafts:
    def __init__(self, values: np.ndarray, row_idx: np.ndarray) -> None:
        self._result = (values, row_idx, 1)

    def wait_host_draft_tokens(self):
        return self._result

    def bump_generation(self, gen: int) -> None:
        """Simulate an epilogue that staged a fresh block."""
        values, row_idx, _ = self._result
        self._result = (values.copy(), row_idx.copy(), gen)


def _req_id(slot: int) -> str:
    """The request id a slot carries, shared by the stub's owner map and the batch.

    The resync cross-check compares a mirror against a device row, and that is
    only a question when both describe the same request -- so the fixture has to
    name requests, not just slots.
    """
    return f"req-{slot}"


def _slot_tokens(slot: int, width: int = 64) -> np.ndarray:
    """The token history a slot carries, shared by the batch and the stub."""
    return np.arange(width, dtype=np.int64) + 1000 * (slot + 1)


_POSITION = 5  # every request in _batch_and_states sits here


def _step_device_ids(
    slots, tokens: np.ndarray, num_reqs: int, q: int, position: int | None = None
) -> list[int]:
    """The runner's ``input_ids`` for the step ``_batch_and_states`` describes.

    Row 0 of each request is the newest committed token from the host mirror; the
    rest are the staged drafts. Same construction the tests spell out by hand, in
    one place so the fixture and the assertions cannot drift apart.
    """
    position = _POSITION if position is None else position
    drafts = _drafts(num_reqs, q).wait_host_draft_tokens()[0]
    ids: list[int] = []
    for order, slot in enumerate(slots):
        ids.append(int(tokens[slot, position]))
        ids.extend(int(v) for v in drafts[order])
    return ids


def _advance_device_ids(
    input_batch,
    st: FakeModelState,
    *,
    num_reqs: int = 2,
    q: int = 3,
    row: np.ndarray | None = None,
    computed: np.ndarray | None = None,
) -> None:
    """Set the fixture's device ``input_ids`` to what the runner holds for this step.

    Row 0 is the mirror's newest committed token and the rest are the staged
    drafts -- read from the state rather than recomputed from ``_slot_tokens``,
    because after a commit the mirror is the ground truth for row 0 and the raw
    token grid is not (``_commit`` lands values like 7 and 11, which are not in the
    grid's synthetic id space).

    In production the two move together: the post-update kernel advances
    ``num_computed`` and the next step's ids start from the newly committed token.
    A test that bumps only the counter describes a device that advanced and a
    runner that did not, and the mode-1 input gate now refuses exactly that -- so a
    test that simulates a commit landing has to advance the ids too, or it is
    testing the refusal path while its name claims it tests the commit path. The
    refusal itself is what ``test_mode_one_refuses_a_device_array_that_lags`` is
    for. One exception, in
    ``test_a_lost_commit_is_repaired_not_poisoning``: there the
    device really has advanced past the mirror, and this helper makes the two
    agree; the mismatch that test simulates is caught by the history-length check,
    which runs before the gate, so the difference never reaches the assertion.
    """
    hist = st._ple_host_history
    # Drain first: the build drains inside ``_stage_host_ple_ids``, and row 0 is
    # whatever that drain appends, so a helper that has to know the ids BEFORE the
    # build has to do the same drain. It is idempotent -- landed commits are popped
    # once -- so the build's own drain then finds nothing pending and the step
    # proceeds exactly as it would have.
    hist.drain()
    drafts = _drafts(num_reqs, q).wait_host_draft_tokens()[0]
    ids: list[int] = []
    for order, slot in enumerate([0, 3][:num_reqs]):
        # ``-1``, not ``length - 1``: the window is a rolling tail of
        # ``ngram_context_len + 1`` cells while ``length`` counts every token the
        # request has accepted. The newest cell is the last one, which is what the
        # builder reads as row 0.
        if row is not None:
            # ... unless the test is describing a device that moved and a mirror
            # that did not hear about it. Then row 0 is the token sitting at this
            # step's position in the device's own row; taking it from the mirror
            # would describe a device that agrees with the mirror the test just
            # made wrong, and the repair would trip the input gate on the
            # fixture rather than on anything real.
            ids.append(int(row[slot, int(computed[slot])]))
        else:
            ids.append(int(hist.window[slot][-1]))
        ids.extend(int(v) for v in drafts[order])
    input_batch.input_ids = torch.tensor(
        ids + [0] * (16 - len(ids)), dtype=torch.int32
    )


def _step_qsl(num_reqs: int, q: int) -> list[int]:
    """``query_start_loc`` for that step, padded the way the runner pads it."""
    num_tokens = num_reqs * q
    qsl = list(range(0, num_tokens + 1, q))
    return qsl + [num_tokens] * (4 - len(qsl))


def _batch_and_states(num_reqs: int = 2, q: int = 3, *, uva: bool = True):
    """A pure-decode batch over slots 0 and 3, with a UVA-shaped host mirror.

    The mirror is deliberately still here after P1.6: the id path no longer reads
    it, and one test below asserts that a batch with no host view at all now
    stages instead of disabling itself.
    """
    slots = [0, 3][:num_reqs]
    idx_mapping = np.array(slots, dtype=np.int64)
    scheduled = np.array([q] * num_reqs, dtype=np.int32)
    computed = np.zeros(8, dtype=np.int64)
    computed[slots] = _POSITION  # context is available for every row
    tokens = np.zeros((8, 64), dtype=np.int64)
    for slot in slots:
        tokens[slot, :] = _slot_tokens(slot)
    # Production exposes the same cells twice: ``.gpu`` is the view the device
    # writes and ``._uva_buf.np`` the host alias of it. The resync reads ``.gpu``,
    # which is also the only view a non-UVA StagedWriteTensor has, so the stub
    # carries both in both configurations.
    gpu = torch.from_numpy(tokens)
    all_token_ids = (
        SimpleNamespace(_uva_buf=SimpleNamespace(np=tokens), gpu=gpu)
        if uva
        else SimpleNamespace(_uva_buf=None, gpu=gpu)
    )
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping_np=idx_mapping,
        num_scheduled_tokens=scheduled,
        # The runner's own device arrays, padded to capacity. The differential
        # dump reads these (not anything forward() stashed) because torch.compile
        # runs that body at trace time only.
        #
        # Filled to agree with the host mirror, because a healthy step is what
        # this fixture claims to describe. They used to be all zeros, which was
        # harmless while only the differential read them: its gate refused and the
        # tests asserted on staging, never on the verdict. Mode 1 now asks the same
        # question before it serves, so zeros would turn every mode-1 test into a
        # test of the refusal path -- and a fixture whose device arrays disagree
        # with the batch it names tests the wrong thing. The gate's refusal on a
        # mismatch is asserted directly, by corrupting these, below.
        input_ids=torch.tensor(
            _step_device_ids(slots, tokens, num_reqs, q) + [0] * (16 - num_reqs * q),
            dtype=torch.int32,
        ),
        query_start_loc=torch.tensor(_step_qsl(num_reqs, q), dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens_np=computed,
        all_token_ids=all_token_ids,
        # The runner's own slot -> request map, which is the only authority on who
        # is in a slot. The resync's cross-check reads it before comparing a
        # mirror against a row.
        index_to_req_id={slot: _req_id(slot) for slot in slots},
    )
    return input_batch, req_states, tokens


def _drafts(num_reqs: int = 2, q: int = 3) -> FakeDrafts:
    slots = np.array([0, 3][:num_reqs], dtype=np.int64)
    values = np.stack(
        [
            np.arange(1001 + 1000 * s + 1, 1001 + 1000 * s + q, dtype=np.int64)
            for s in slots
        ]
    )
    return FakeDrafts(values, slots)


@pytest.fixture
def hostids_env(monkeypatch):
    """Force a mode and an empty registry; return setters for both."""
    state: dict[str, Any] = {"mode": 1, "layers": []}
    monkeypatch.setattr(ple_hostids, "hostids_mode", lambda: state["mode"])
    monkeypatch.setattr(
        ple_layer_module, "host_stageable_layers", lambda: list(state["layers"])
    )
    return state


def test_mode_zero_does_nothing(hostids_env) -> None:
    hostids_env["mode"] = 0
    st = FakeModelState()
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_layers is None
    assert st._ple_host_refusals == {}
    assert st._ple_host_steps == 0


def test_model_without_ple_does_nothing(hostids_env) -> None:
    st = FakeModelState()
    st.uses_ngram_embedding = False
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_layers is None
    assert st._ple_host_steps == 0


def test_no_registered_layers_is_reported(hostids_env) -> None:
    """The knob is set but no embedding asked for it -- not a silent no-op."""
    st = FakeModelState()
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert "no-registered-layers" in st._ple_host_refusals


def test_no_draft_source_is_reported(hostids_env) -> None:
    hostids_env["layers"] = [StagingEmbedding()]
    st = FakeModelState()
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert "no-draft-source" in st._ple_host_refusals
    assert st._ple_host_steps == 0


def test_host_stage_source_is_wired_once(hostids_env) -> None:
    st = FakeModelState()
    handler = FakeDrafts(np.zeros((1, 1), dtype=np.int64), np.array([0]))
    st.set_host_draft_source(handler)
    assert st._host_drafts is handler


def test_id_path_survives_a_platform_with_no_host_view(hostids_env) -> None:
    """P1.6: the ids no longer depend on a host view of ``all_token_ids``.

    Before the token mirror existed, a ``req_states.all_token_ids`` without a
    ``_uva_buf`` meant the host could never know a step's history, so staging
    disabled itself permanently. That is exactly the buffer whose newest cells are
    written by a GPU kernel unordered before the host read, so depending on it was
    both fragile and unnecessary. The prefetcher still disables itself on such a
    platform -- that path reads the device array by design.
    """
    hostids_env["layers"] = [StagingEmbedding()]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states(uva=False)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert st._ple_host_steps == 1
    assert st.staged_rows() == [ROWS]


def test_pure_decode_batch_reaches_the_layer(hostids_env) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()
    # A decoy with the *batch-aligned* shape the same-named array has on
    # InputBatch: reading that one instead of req_states' slot-aligned array is a
    # plausible mistake and must not be invisible.
    input_batch.num_computed_tokens_np = np.array([5, 0], dtype=np.int64)

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert st.staged_rows() == [ROWS]
    assert emb.lookup_calls == 1
    assert st._ple_host_steps == 1
    # Not just "something was staged": the ids must be the ones this batch's
    # slots, positions and drafts imply. Row 0 of each request is the newest
    # committed token, from the host token mirror at the slot's computed position;
    # the rest are the staged drafts.
    expected_rows = []
    staged_drafts = _drafts().wait_host_draft_tokens()[0]
    for order, slot in enumerate((0, 3)):
        row = [int(tokens[slot, 5])]
        row += [int(v) for v in staged_drafts[order]]
        expected_rows.append(row)
    expected_input_ids = np.array(
        [v for row in expected_rows for v in row], dtype=np.int64
    )
    expected = _expected_ids(
        expected_input_ids,
        np.array([0, 3, 6], dtype=np.int64),
        # The n-gram context the builder reads out of the same mirror: the CTX
        # tokens just before each row's position.
        np.array(
            [
                [tokens[0, 5 - CTX + i] for i in range(CTX)],
                [tokens[3, 5 - CTX + i] for i in range(CTX)],
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(emb._pinned_ngram_ids[:ROWS].numpy(), expected)


def test_mode_one_refuses_a_device_array_that_lags(hostids_env) -> None:
    """The gate boot #14 said the serving path needed.

    Mode 1 used to stage whatever the mirror handed it. Boot #14 (differential
    mode, same builder, same mirror) caught the mirror returning
    ``host=4161103993993705554`` where a token id belongs -- 8 refused comparisons
    across 4 workers in 45 minutes. Mode 2 refused those steps and the device served
    them; mode 1 would have gathered rows out of a nonexistent id and handed the
    model a plausible answer built from garbage.

    So: when the runner's arrays and the mirror do not describe the same tokens,
    nothing stages. The layer's flag stays down, the op falls through to the device
    path, and the refusal is counted. Staging nothing is the whole safety argument;
    asserting it is the point of the test.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()
    # Row 0 of each request holds the token BEFORE the newest committed one: the
    # shape of a step whose sides disagree, whether the mirror ran ahead or the
    # runner's array is stale.
    input_batch.input_ids[0] = int(tokens[0, _POSITION - 1])
    input_batch.input_ids[3] = int(tokens[3, _POSITION - 1])

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals == {"stage-inputs-disagree": 1}
    assert st._ple_host_steps == 0
    assert st.staged_rows() == [-1], "a refused step armed the D2H skip"
    # -1 is the layer's "nothing staged" sentinel: ``take_host_stage`` will not
    # answer yes, so the op falls through to the device path for this step.
    assert emb.lookup_calls == 0
    assert emb._host_stage_refusals == 1


def test_mode_one_refuses_a_draft_column_the_device_does_not_have(hostids_env) -> None:
    """A draft position is as load-bearing as row 0.

    The staged rows are [newest committed, draft, draft]; a wrong draft column
    gathers embedding rows for a token the device never fed to its own embedding
    lookup. The gate covers the whole id vector, not just the boundary the builder
    derived its positions from -- which is the same reach the differential's gate
    has, and the reason both now share one implementation.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    input_batch.input_ids[1] = 987654  # a draft token the device does not have

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals == {"stage-inputs-disagree": 1}
    assert emb.lookup_calls == 0
    assert emb._host_stage_refusals == 1


def test_a_staging_refusal_is_worded_as_a_refusal_to_stage(
    hostids_env, caplog
) -> None:
    """Two different claims, so two different counters and two different verbs.

    ``SKIPPED a comparison`` means the device served the step and the harness
    declined to judge it. ``REFUSED to stage`` means the serving path declined the
    step and the device served it. The first is a hole in coverage; the second is a
    statement about served output. Blurring them is how a diagnostic starts lying,
    and this file has a counter for each because of it.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    input_batch.input_ids[0] = 999

    st._stage_host_ple_ids(input_batch, req_states)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "REFUSED to stage" in text, text
    assert "SKIPPED a comparison" not in text, text
    assert "input_ids differ at 0 of 6" in text, text
    assert emb._host_input_skips == 0, "a staging refusal counted as a comparison skip"
    assert emb._host_stage_refusals == 1


def test_the_comparison_gate_still_counts_comparisons(hostids_env) -> None:
    """The shared implementation must not merge the two counters.

    ``_inputs_agree`` now serves both paths, so a refactor that routed every
    refusal to one counter would keep every other assertion in this file green while
    quietly deleting the distinction the boot reads. A mode-2 refusal still lands on
    ``_host_input_skips`` and leaves the staging counter alone.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    other = ran_ids.copy()
    other[1] = 4
    emb._host_check_dev_inputs = (
        torch.from_numpy(other),
        torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    assert emb._host_input_skips == 1, "the input-agreement gate did not fire"
    assert emb._host_stage_refusals == 0, "a comparison skip counted as a refusal"


def test_the_stage_gate_stays_out_of_graph_capture(hostids_env, monkeypatch) -> None:
    """During a capture there is no step to attribute, and a D2H would kill the boot.

    The gate reads three device arrays synchronously. Inside a stream capture that
    is ``hipErrorStreamCaptureUnsupported``: the capture aborts and the engine dies
    at startup -- the same class of failure that ended boot #26 at config time, one
    layer down. And it would protect nothing: replay never reads the host arrays,
    and the rows a capture leaves behind are zeroed on purpose
    (``test_ple_capture_stage``). So the gate asks the capture question first.
    """
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    input_batch.input_ids[0] = 999  # would refuse if the gate were consulted

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert emb.lookup_calls == 1
    assert emb._host_stage_refusals == 0


def test_mode_two_stages_for_comparison_only(hostids_env) -> None:
    hostids_env["mode"] = 2
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)

    assert emb._host_check_rows == ROWS
    assert emb._host_stage_rows == -1, "differential mode armed the D2H skip"
    assert emb.lookup_calls == 0, "differential mode gathered"
    assert st._ple_host_steps == 1


def test_mixed_batch_refuses_and_the_layer_is_untouched(hostids_env) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    # One request is a chunked prefill chunk: not a decode step at all.
    input_batch.num_scheduled_tokens = np.array([3, 12], dtype=np.int32)

    st._stage_host_ple_ids(input_batch, req_states)

    assert "not-pure-decode" in st._ple_host_refusals
    assert emb._host_stage_rows == -1
    assert emb.lookup_calls == 0
    assert st._ple_host_steps == 0


def test_missing_draft_row_refuses(hostids_env) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    # The mirror describes a different batch than the one being prepared.
    st.set_host_draft_source(
        FakeDrafts(
            np.arange(2 * 2, dtype=np.int64).reshape(2, 2),
            np.array([0, 1], dtype=np.int64),
        )
    )
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)

    assert "row-missing" in st._ple_host_refusals
    assert emb._host_stage_rows == -1


def test_unexpected_failure_disables_and_survives(hostids_env) -> None:
    """A knob must never be able to kill a step, let alone a boot."""

    class Exploding:
        def wait_host_draft_tokens(self):
            raise RuntimeError("mirror is broken")

    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(Exploding())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)  # must not raise

    assert st._ple_host_layers == []
    assert emb._host_stage_rows == -1


def test_layer_failure_disables_midflight(hostids_env) -> None:
    class ExplodingLayer(StagingEmbedding):
        def stage_host_step(self, *a, **kw):  # noqa: ANN002, ANN003
            raise RuntimeError("gather blew up")

    good = StagingEmbedding()
    hostids_env["layers"] = [good, ExplodingLayer()]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_layers == []
    assert st._ple_host_steps == 0, "a partial stage was counted as a success"


def test_refusals_are_counted_and_logged_once(hostids_env, caplog) -> None:
    st = FakeModelState()
    with caplog.at_level("INFO"):
        st._note_host_refusal("not-pure-decode")
        st._note_host_refusal("not-pure-decode")
        st._note_host_refusal("no-drafts")
    assert st._ple_host_refusals["not-pure-decode"] == 2
    assert st._ple_host_refusals["no-drafts"] == 1
    logged = [r.getMessage() for r in caplog.records if "HOSTIDS" in r.getMessage()]
    assert len(logged) == 2, "a repeated refusal reason logged repeatedly"


def test_a_refusal_under_a_captured_graph_denies_the_device_path(
    hostids_env, caplog
) -> None:
    """Under FULL, "device path used for that step" is a lie -- so say the truth.

    This count is the shipping gate for ``GRAPHMODE=full`` + ``HOSTIDS=1``: a step
    the host refused still replays, and the graph hands back the PREVIOUS step's
    embedding rows. The generation tripwire cannot see a sustained refusal (nothing
    advances the counter, so nothing looks stale), which is why the counter exists.
    A FULL boot that leaves this at zero never reached the cliff; a non-zero number
    means FULL needs the runner to pick a non-FULL descriptor for unstaged steps.
    """
    emb = StagingEmbedding()
    emb._staged_branch_captured = True
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_gen_layers = [emb]

    with caplog.at_level("WARNING"):
        st._note_host_refusal("history-pending")
        st._note_host_refusal("history-pending")

    assert st._ple_host_captured_refusals == 2
    msgs = [
        r.getMessage() for r in caplog.records if "captured graph" in r.getMessage()
    ]
    assert len(msgs) == 1, "first, then every 32nd -- not one per step"
    assert "NO device path" in msgs[0]
    assert "previous step's embedding rows" in msgs[0]


def test_disable_reports_the_coverage_it_is_leaving(hostids_env, caplog) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1

    with caplog.at_level("WARNING"):
        st.disable_host_staging("test reason")

    assert st._ple_host_layers == []
    # Engagement and coverage have to be recoverable from the log alone: a
    # disable that does not say how much it is switching off is not auditable.
    assert any("after 1 step" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# P1.5: the generation tripwire
#
# A frozen CUDA graph is silent: it replays its captured work and prints nothing,
# so the only way to know it is carrying this step's PLE rows is to have the graph
# itself carry a number. These tests cover the whole chain -- the counter written
# with the rows, the copy inside the graph, the non-blocking read, the verdict --
# because the failure it exists to catch (2026-09-20: wrong embeddings for five
# and a half hours, found by eye) is invisible to every other check here.
# --------------------------------------------------------------------------


def _stage(emb, *, gather: bool = True, num_reqs: int = 2) -> bool:
    inp, qsl, ctx = _step_inputs(num_reqs=num_reqs)
    dev = None
    if not gather:
        # Differential staging hands the runner's own arrays over, and the gate reads
        # them the next time the op runs. Padding to the buffer capacity is the
        # runner's shape, not a decoration: the real array is padded too, and a test
        # fixture that handed an unpadded one would be asserting a shape production
        # never produces.
        pad = max(0, CAPACITY - qsl.shape[0])
        dev = (
            torch.from_numpy(inp.copy()),
            torch.from_numpy(
                np.concatenate([qsl, np.full(pad, int(qsl[-1]), dtype=np.int64)])
            ),
            None,
        )
    return emb.stage_host_step(inp, qsl, ctx, gather=gather, device_inputs=dev)


def _agree_inputs(emb, *, rows: int = ROWS) -> None:
    """Say what both sides of a hand-driven comparison read.

    ``compare_host_ids`` refuses anything whose inputs it cannot match up (P1.10b),
    so a test that hands it an ids buffer directly has to state the tokens the compute
    consumed as well as the ones the host mirrored. This is the agreeing case; the
    refusing cases build the two arrays by hand so the disagreement is visible in the
    test rather than produced by a helper's arithmetic.
    """
    host_ids = np.arange(rows, dtype=np.int64) + 100
    host_qsl = np.array([0, rows // 2, rows], dtype=np.int64)
    host_ctx = np.zeros((rows, CTX), dtype=np.int64)
    emb._host_check_inputs = (host_ids, host_qsl, host_ctx)
    pad = max(0, CAPACITY - host_qsl.shape[0])
    emb._host_check_dev_inputs = (
        torch.from_numpy(host_ids.copy()),
        torch.from_numpy(
            np.concatenate([host_qsl, np.full(pad, rows, dtype=np.int64)])
        ),
        # P1.11: a window the device agrees with, padded to buffer capacity the way
        # ``Qwen4ExpModelState.ngram_context`` is. Handing None here instead would
        # exercise the unchecked leg, not the compared one, and every green test in
        # this file would silently be testing two legs out of three.
        torch.from_numpy(
            np.concatenate([host_ctx, np.zeros((CAPACITY - rows, CTX), np.int64)])
        ),
    )


def _interval(monkeypatch, n: int) -> None:
    monkeypatch.setattr(ple_hostids, "hostids_check_interval", lambda: n)


def _replay(emb, gen: int) -> None:
    """Stand in for the graph: the captured copy put ``gen`` into the scratch.

    Call this *before* the check that issues the read, not after. The read is a
    copy that runs in stream order, so what it samples is the scratch's value at
    the moment the issue happens -- setting the scratch afterwards would model a
    copy that reads the future, and would have made the frozen-handoff tests pass
    for the wrong reason.
    """
    assert emb._gen_device is not None
    emb._mirror_op_gen = False  # this test drives the scratch itself from here on
    emb._gen_device.fill_(gen)


def test_gen_verdict_covers_every_direction() -> None:
    """The pure half of the tripwire, testable with no torch and no device."""
    assert ple_hostids.gen_verdict(5, 5) == "ok"
    assert ple_hostids.gen_verdict(4, 5) == "frozen"
    assert ple_hostids.gen_verdict(0, 5) == "frozen"
    assert ple_hostids.gen_verdict(6, 5) == "ahead"
    # "never staged" must not borrow "ok"'s authority.
    assert ple_hostids.gen_verdict(0, 0) == "no-staging"


def test_check_interval_defaults_clamps_and_ignores_junk(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_GFX906_PLE_HOSTIDS_CHECK", raising=False)
    assert ple_hostids.hostids_check_interval() == 64
    monkeypatch.setenv("VLLM_GFX906_PLE_HOSTIDS_CHECK", "1")
    assert ple_hostids.hostids_check_interval() == 1
    # 0 would sample every step and, worse, make `countdown > 0` never true in a
    # way that reads like a bug; a negative interval is not a thing.
    monkeypatch.setenv("VLLM_GFX906_PLE_HOSTIDS_CHECK", "0")
    assert ple_hostids.hostids_check_interval() == 1
    monkeypatch.setenv("VLLM_GFX906_PLE_HOSTIDS_CHECK", "yes")
    assert ple_hostids.hostids_check_interval() == 64


def test_staging_advances_the_generation_with_the_rows() -> None:
    emb = StagingEmbedding()
    assert _stage(emb) is True
    assert emb._host_stage_gen == 1
    assert int(emb._pinned_gen[0]) == 1
    _stage(emb)
    assert emb._host_stage_gen == 2 and int(emb._pinned_gen[0]) == 2


def test_a_refused_stage_does_not_advance_the_generation() -> None:
    """The same-act rule, observed from the refusing side.

    ``stage_host_step`` returns before writing anything when the rows will not
    fit, so the counter must not move either. A counter that advanced past a
    refusal would tell the next check "the graph owes you generation N" for rows
    that were never staged -- inventing a stale handoff out of a healthy step.
    """
    emb = StagingEmbedding()
    assert _stage(emb) is True
    # 3 requests x 3 tokens = 9 rows > CAPACITY: refused, nothing written.
    assert _stage(emb, num_reqs=3) is False
    assert emb._host_stage_gen == 1, "a refused step advanced the generation"
    assert int(emb._pinned_gen[0]) == 1


def test_differential_mode_never_advances_the_generation() -> None:
    """Mode 2 keeps the device in charge, so the tripwire stays inert by design.

    Not a mode check inside the op: the counter simply never leaves zero, which
    makes "frozen" unreachable for a path that never claimed to hand anything off.
    """
    emb = StagingEmbedding()
    assert _stage(emb, gather=False) is True
    assert emb._host_stage_gen == 0
    assert int(emb._pinned_gen[0]) == 0


def test_the_check_issues_a_read_only_every_nth_step(monkeypatch) -> None:
    """Sampling is throttled, and the sampled step's answer arrives on the next."""
    emb = StagingEmbedding()
    _interval(monkeypatch, 3)
    emb._gen_check_countdown = 3
    _stage(emb)
    verdicts = []
    for _ in range(9):
        _replay(emb, emb._host_stage_gen)  # the graph keeps up
        verdicts.append(emb.check_host_gen())
    assert verdicts[:6] == ["", "", "issued", "ok", "", ""], verdicts
    # Two samples in nine steps of interval 3 (the step that reads a verdict back
    # also occupies a step, which is why this is not every third call exactly).
    assert verdicts.count("issued") == 2
    assert emb._gen_event.records == 2
    assert emb._gen_checks_ok == 2
    assert emb._hostids_disabled is False


def test_nothing_staged_yet_is_not_reported_as_ok() -> None:
    """A vacuous pass is reported as what it is.

    "ok" for a boot that has staged nothing would make the first interval of every
    run look verified, which is the same class of lie as an instrument that cannot
    go red.
    """
    emb = StagingEmbedding()
    assert emb.check_host_gen() == "no-staging"
    assert emb._gen_checks_ok == 0


def test_a_fresh_handoff_passes_and_is_counted(monkeypatch) -> None:
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    _stage(emb)
    _replay(emb, 1)  # the graph delivered generation 1
    assert emb.check_host_gen() == "issued"
    assert emb.check_host_gen() == "ok"
    assert emb._gen_checks_ok == 1
    assert emb._hostids_disabled is False
    assert emb._gen_trips == 0


def test_a_frozen_handoff_disables_while_the_device_path_is_reachable(
    monkeypatch,
) -> None:
    """No captured graph => the device path really does run, so back off."""
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    _stage(emb)
    _replay(emb, 0)  # the scratch is stuck at capture's value
    assert emb.check_host_gen() == "issued"
    assert emb.check_host_gen() == "frozen"
    assert emb._hostids_disabled is True
    assert emb._gen_trips == 1


def test_a_frozen_handoff_is_fatal_once_a_graph_captured_the_staged_branch(
    monkeypatch,
) -> None:
    """The case where "fall back" is not a thing a replay can do.

    Inside a captured graph the op body does not run, so the device path is
    unreachable and every further step embeds the wrong step's tokens. Continuing
    is the 2026-09-20 failure; raising is the only correct remaining action.
    """
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    emb._staged_branch_captured = True
    _stage(emb)
    _replay(emb, 0)  # capture's value: the graph never advanced it
    assert emb.check_host_gen() == "issued"
    with pytest.raises(RuntimeError, match="frozen embeddings"):
        emb.check_host_gen()


def test_a_device_value_ahead_of_the_host_is_not_treated_as_ok(monkeypatch) -> None:
    """One scratch and one producer cannot get ahead; if it does, stop anyway."""
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    _stage(emb)
    _replay(emb, 9)
    assert emb.check_host_gen() == "issued"
    assert emb.check_host_gen() == "ahead"
    assert emb._hostids_disabled is True


def test_a_read_that_never_lands_fails_instead_of_hanging(monkeypatch) -> None:
    """If the scratch is never written the check must still terminate loudly."""
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    emb._gen_event = FakeEvent(ready=False)
    _stage(emb)
    verdicts = [emb.check_host_gen() for _ in range(6)]
    assert verdicts[0] == "issued"
    assert "in-flight" in verdicts
    assert emb._hostids_disabled is True, verdicts
    assert emb._gen_trips == 1


def test_the_check_never_waits_on_the_device(monkeypatch) -> None:
    """The instrument must not cost the thing it protects.

    ``FakeEvent.synchronize`` raises, so reaching it anywhere in this file fails
    the test that happened to be running; this test asserts the non-blocking path
    was actually exercised rather than skipped.
    """
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    emb._gen_event = FakeEvent(ready=False)
    _stage(emb)
    assert emb.check_host_gen() == "issued"
    assert emb.check_host_gen() == "in-flight"
    assert emb._gen_event.records == 1
    assert emb._gen_event.queries == 1
    assert emb._hostids_disabled is False


def test_the_op_copies_the_generation_into_the_scratch(monkeypatch) -> None:
    """The graph node that makes the tripwire a measurement instead of a tautology.

    RED CONTROL: delete the ``scratch.copy_`` in the staged branch and this fails,
    because the scratch would still hold capture's value while the host counter
    marched on.
    """
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS
    emb._pinned_gen.fill_(7)
    assert emb._gen_device is not None
    assert int(emb._gen_device[0]) == 0
    with _op_env(emb, monkeypatch):
        _run_op()
    assert int(emb._gen_device[0]) == 7


def test_the_op_records_that_the_staged_branch_was_captured(monkeypatch) -> None:
    """What makes a tripwire fire fatal rather than a quiet disable."""
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS
    with _op_env(emb, monkeypatch, capturing=True):
        _run_op()
    assert emb._staged_branch_captured is True

    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS
    with _op_env(emb, monkeypatch, capturing=False):
        _run_op()
    assert emb._staged_branch_captured is False


def test_the_device_path_never_touches_the_generation_scratch(monkeypatch) -> None:
    """The copy is a node of the staged branch only.

    If the device path also copied, a frozen staged graph would be "fixed" by the
    eager path writing the counter, and the tripwire would report a healthy handoff
    for a graph that delivered nothing.
    """
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS - 1  # disagreement -> device path
    emb._pinned_gen.fill_(7)
    with _op_env(emb, monkeypatch):
        _run_op()
    assert emb.lookup_calls == 1
    assert emb._gen_device is not None
    assert int(emb._gen_device[0]) == 0, "the device path advanced the tripwire"


def test_scratch_allocation_refuses_to_happen_inside_a_capture(monkeypatch) -> None:
    emb = StagingEmbedding()
    emb._gen_device = None
    emb._host_stage_rows = ROWS
    with (
        _op_env(emb, monkeypatch, capturing=True),
        pytest.raises(RuntimeError, match="never allocated"),
    ):
        _run_op()


def test_scratch_is_created_lazily_outside_a_capture(monkeypatch) -> None:
    """A module that never ran load_weights still works, and reports honestly."""
    emb = StagingEmbedding()
    emb._gen_device = None
    emb._host_stage_rows = ROWS
    emb._pinned_gen.fill_(4)
    with _op_env(emb, monkeypatch, capturing=False):
        _run_op()
    assert emb._gen_device is not None
    assert int(emb._gen_device[0]) == 4


# --------------------------------------------------------------------------
# Wiring: the model state runs the check even on steps it refused
# --------------------------------------------------------------------------


class WatchedLayer:
    """A registered embedding that records tripwire samples."""

    ngram_size = NGRAM_SIZE

    def __init__(self) -> None:
        self.samples = 0
        self._staged_branch_captured = False
        self._host_stage_rows = -1

    def check_host_gen(self) -> str:
        self.samples += 1
        return "ok"

    def take_host_stage(self):
        return None

    def stage_host_step(self, *a: Any, **kw: Any) -> bool:  # noqa: ANN002, ANN003
        return False


def test_the_check_runs_on_steps_the_host_refused(hostids_env) -> None:
    """A refusal is the interesting case, so the check sits above the refusals."""
    hostids_env["layers"] = [WatchedLayer()]
    st = FakeModelState()
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert "no-draft-source" in st._ple_host_refusals
    assert st._ple_host_steps == 0
    layer = hostids_env["layers"][0]
    assert layer.samples == 1, "the tripwire was skipped on a refused step"


def test_the_check_runs_once_per_registered_layer(hostids_env) -> None:
    hostids_env["layers"] = [WatchedLayer(), WatchedLayer()]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    # Resolution is what arms the watchers; nothing sets this by hand.
    assert st._ple_host_gen_layers == hostids_env["layers"]
    assert all(layer.samples == 1 for layer in hostids_env["layers"])


def test_disabling_keeps_the_tripwire_armed_after_capture(hostids_env, caplog) -> None:
    """Disabling the host path must not disarm the instrument that proves it broke.

    Once a graph captured the staged branch, a disabled host path *is* the
    frozen-embedding state; and the disable message must not claim a device path
    that a replay cannot reach.
    """
    layer = WatchedLayer()
    hostids_env["layers"] = [layer]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)

    layer._staged_branch_captured = True
    with caplog.at_level("WARNING"):
        st.disable_host_staging("test reason")

    assert st._ple_host_layers == []
    assert st._ple_host_gen_layers == [layer], "disarming hid the frozen graph"
    assert any("does NOT handle" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


def test_disabling_reports_a_real_fallback_when_nothing_was_captured(
    hostids_env, caplog
) -> None:
    layer = WatchedLayer()
    hostids_env["layers"] = [layer]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)

    with caplog.at_level("WARNING"):
        st.disable_host_staging("test reason")

    assert st._ple_host_layers == []
    assert st._ple_host_gen_layers == [layer]
    assert any("handles every step" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The producer side: the model runner's epilogue handing over accepted tokens
# ---------------------------------------------------------------------------


class FakeCopyEvent:
    """``AsyncOutput.copy_event``, as far as the mirror is concerned."""

    def __init__(self, done: bool = False) -> None:
        self.done = done
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.done


def _commit(counts=(1, 1), done=False):
    """A commit for slots 0 and 3: one accepted token each by default."""
    rows = np.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=np.int32)
    return (
        np.array([0, 3], dtype=np.int32),
        rows,
        np.array(counts, dtype=np.int32),
        FakeCopyEvent(done),
    )


def test_epilogue_commit_is_free_when_the_knob_is_off(hostids_env) -> None:
    """The epilogue runs on every step of every boot, knob or no knob.

    So with the knob off it must queue nothing, count nothing and -- because a
    stray log line here would be read as engagement -- say nothing.
    """
    hostids_env["mode"] = 0
    st = FakeModelState()
    assert st.stage_host_tokens(*_commit()) is False
    assert st._ple_host_history.pending == 0
    assert st._ple_host_refusals == {}


def test_landed_commit_advances_the_step(hostids_env) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()

    assert st.stage_host_tokens(*_commit(done=True)) is True
    # The commit carries one accepted token per request; the device's post-update
    # kernel advances num_computed by the same count, and so does the scheduler's
    # host copy that the batch carries.
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert st._ple_host_steps == 1
    assert st._ple_host_history.pending == 0
    # The token the commit carried is now the newest cell of each slot's window,
    # which is what the builder reads as row 0. (That the ids then follow is
    # test_pure_decode_batch_reaches_the_layer's job, with hashed values.)
    assert st.staged_rows() == [ROWS]
    assert emb.lookup_calls == 1
    hist = st._ple_host_history
    assert (hist.window[0][-1], hist.window[3][-1]) == (7, 11)
    assert (hist.length[0], hist.length[3]) == (_POSITION + 2, _POSITION + 2)


def test_unlanded_commit_refuses_the_step_rather_than_waiting(hostids_env) -> None:
    """The design's central bargain: no per-step host sync, so refuse instead.

    A commit whose D2H has not landed means the newest token is not host-visible
    yet. Blocking for it is the sync this whole path exists to remove; skipping it
    would build ids one token short. Refusing the step costs the device path one
    step and keeps the promise.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()

    slots, rows, counts, event = _commit(done=False)
    assert st.stage_host_tokens(slots, rows, counts, event) is True
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals.get("history-pending") == 1
    assert st._ple_host_steps == 0
    assert emb.lookup_calls == 0

    # The copy lands; the same step now goes through, with the counter advanced
    # by the commit it was waiting on.
    event.done = True
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1
    assert st._ple_host_history.pending == 0


def test_a_lost_commit_is_repaired_not_poisoning(hostids_env) -> None:
    """A commit that fails validation costs a step and a sync, not the request.

    The device advanced its counter whether or not the host mirror got the tokens,
    so the mirror is one token short afterwards. Until 2026-10-04 that was
    terminal: nothing could rebuild a slot -- ``seed`` ran at add and
    ``invalidate`` at remove and never again -- so the exact-length check refused
    that request for the rest of its life. Boot #26 measured the result: 411
    ``history-length`` refusals per worker and six consecutive windows of 32/32,
    with the drops that caused them all clustered behind them at batch transitions.

    What the repair does NOT give up: it reads nothing until the device has been
    synced (the row's generated cells are written by a kernel that is unordered
    before a host read), it cross-checks the row against the cells the old window
    still claims to hold, and it is budgeted per request and per boot. A mirror it
    cannot vouch for still refuses the step rather than building ids from a
    history with a hole in it.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, tokens = _batch_and_states()

    slots, rows, counts, event = _commit(done=True)
    # A commit claiming more accepted tokens than the row holds.
    assert (
        st.stage_host_tokens(slots, rows, np.array([9, 1], dtype=np.int32), event)
        is False
    )
    assert "stage-tokens-rejected" in st._ple_host_refusals
    # Boot #26 could not say why a commit was lost, only how many. The detail now
    # names the check that lost it.
    assert "num_sampled" in st._ple_host_history.last_drop

    # The device advanced anyway, so the mirror now lags by exactly one token.
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals.get("history-length") is None, "not repaired"
    assert st._ple_host_steps == 1
    assert st._ple_resyncs == 1, "the repair did not run"
    assert st._ple_syncs == 1, "the repair read the row without ordering the device"
    hist = st._ple_host_history
    for slot in (0, 3):
        assert hist.length[slot] == req_states.num_computed_tokens_np[slot] + 1
        assert int(hist.window[slot, -1]) == int(tokens[slot, hist.length[slot] - 1])

    # The budget is per request and finite. A third loss is a systematic problem,
    # and paying another device sync per step for it turns a coverage bug into a
    # latency bug that produces entirely plausible text.
    for step in range(FakeModelState.PLE_RESYNC_PER_SLOT - 1):
        # A fresh draft block each step, or the staleness check refuses before the
        # build and the loop measures the wrong thing.
        drafts.bump_generation(10 + step)
        req_states.num_computed_tokens_np[[0, 3]] += 1
        _advance_device_ids(
            input_batch,
            st,
            row=req_states.all_token_ids.gpu.numpy(),
            computed=req_states.num_computed_tokens_np,
        )
        st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resyncs == FakeModelState.PLE_RESYNC_PER_SLOT
    drafts.bump_generation(99)
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals.get("history-length") == 1
    assert st._ple_resyncs == FakeModelState.PLE_RESYNC_PER_SLOT, "budget exceeded"
    assert "per-slot cap" in st._ple_resync_refused


def test_resync_refuses_to_serve_when_the_row_disagrees_with_the_mirror(
    hostids_env,
) -> None:
    """A repair that takes its length from the row it read cannot notice a bad read.

    So it cross-checks the row against the cells the old window still claims --
    ``num_sampled`` commits on one side, the device's own token buffer on the
    other. Disagreement over the overlap means one of those two is lying about the
    request's history, and boot #14 already caught the host mirror handing back
    ``4161103993993705554`` where a token id belongs. The step is refused; the
    worker does not serve from a read it cannot vouch for. What it does with a
    *pattern* of these is ``test_a_disagreement_refuses_the_step_not_the_boot``.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    hist = st._ple_host_history
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    old_len = int(hist.length[0])
    req_states.all_token_ids.gpu[0, old_len - 1] = 4161103993993705554 % (2**31)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resync_disagree == 1
    assert st._ple_host_steps == 0, "ids built from a row that contradicts the mirror"
    assert st._ple_host_refusals.get("history-length") == 1


def test_the_cross_check_only_compares_the_request_that_owns_the_slot(
    hostids_env,
) -> None:
    """Boot #27: two requests in one slot is not a disagreement, it is two requests.

    At 02:30:55 -- three seconds before its first real step -- boot #27 compared a
    mirror holding ``[4, 5, 10]`` against a row holding ``[4, 5, 19959]`` on slot 2
    and turned the knob off on all four workers. The mirror had been seeded by a
    warmup dummy that never went through ``add_request``; the row held the previous
    occupant's tokens, which ``invalidate`` does not clear because it is not the
    mirror's to clear. Neither side was wrong and the boot was dead.

    So the row is compared against the window only when the runner says the slot
    still belongs to the request the mirror describes. Otherwise the post-sync read
    of the device's own token buffer is taken on trust, which is what it is for.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    hist = st._ple_host_history
    req_states.num_computed_tokens_np[[0, 3]] += 1
    old_len = int(hist.length[0])
    # A token the mirror has never seen, in the cell the overlap would compare.
    req_states.all_token_ids.gpu[0, old_len - 1] = 4161103993993705554 % (2**31)
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    # Slot 0 now holds somebody else; slot 3 still holds who the mirror says. The
    # per-slot distinction is the whole point, so the test keeps one of each.
    req_states.index_to_req_id = {0: "the-previous-occupant", 3: _req_id(3)}

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_resync_disagree == 0, "a stranger's tokens were read as a defect"
    assert st._ple_syncs == 1
    assert st._ple_resync_owner_skip == 1
    assert st._ple_host_steps == 1, "the slot was rebuilt but the step was lost anyway"
    assert int(hist.window[0, -1]) == int(
        req_states.all_token_ids.gpu[0, hist.length[0] - 1]
    ), "the mirror did not take the row it was told to trust"
    assert st._ple_host_slot_req[0] == "the-previous-occupant"
    # The slot whose owner matched was still checked, and passed.
    assert st._ple_host_slot_req[3] == _req_id(3)


def test_a_disagreement_with_the_owner_refuses_the_step_not_the_boot(
    hostids_env,
) -> None:
    """One bad read is a refused step; a pattern of them is a dead knob.

    Under a captured graph the step has no device fallback either way, so the old
    immediate disable traded every other request in the boot for coverage it was
    not going to get on that step anyway. Disabling stays correct once the
    disagreement repeats, because then the thing that is broken is the host's
    ability to read its own memory.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    hist = st._ple_host_history
    req_states.num_computed_tokens_np[[0, 3]] += 1
    old_len = int(hist.length[0])
    req_states.all_token_ids.gpu[0, old_len - 1] = 4161103993993705554 % (2**31)
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resync_disagree == 1
    assert st._ple_host_steps == 0, "ids built from a row that contradicts the mirror"
    assert st._ple_host_layers == [emb], "one disagreement disabled the whole boot"
    assert emb._hostids_disabled is False

    # The same slot again. The mirror was deliberately not re-seeded after a
    # disagreement, so it still lags and the next step asks the same question --
    # which is the point: two on one request means the two sides will not agree.
    drafts.bump_generation(7)
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_resync_disagree == 2
    assert st._ple_host_layers == [], "a repeating disagreement kept the knob on"
    assert st._ple_resync_disagree_slots[0] == FakeModelState.PLE_DISAGREE_PER_SLOT


def test_staged_steps_stops_when_staging_is_disabled(hostids_env) -> None:
    """A counter that survives the thing it counts cannot be the shipping signal.

    Boot #27 logged ``staged_steps=5120`` for a worker that had disabled staging at
    02:30:55, whose mirror had taken 1,140 commits and whose layer had been handed
    one row -- the capture dummy. ``_stage_host_ple_ids`` tested only whether the
    layer list had been *built*, and an empty list built just as well as a full
    one, so the counter advanced through a staging loop that ran zero times. Every
    number on that line now comes from at or below the layer that does the work.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1
    assert st._ple_host_layer_steps() == 1

    st.disable_host_staging("test: what the counter does afterwards")
    before = st._ple_host_layer_steps()

    for step in range(3):
        drafts.bump_generation(10 + step)
        st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_steps == 1, "the counter kept rising after the disable"
    assert st._ple_host_layer_steps() == before
    assert st._ple_host_refusals.get("staging-disabled") == 3
    coverage = st._ple_host_coverage()
    assert "staged_steps=1" in coverage
    assert f"layers_staged={before}" in coverage
    assert "disabled=yes" in coverage


def test_a_layer_that_declines_the_ids_is_not_counted_as_staged(
    hostids_env,
) -> None:
    """``stage_host_step`` answers whether it staged. The answer is the fact.

    The layer returns False when its own hostids are off or the step is wider than
    its pinned buffer, which means these rows are not in the buffer this step's
    replay reads. The return value used to be dropped and the step counted as
    staged anyway -- the same one-line class of slip as the empty layer list, and
    invisible for exactly the same reason: nothing downstream consumed the answer.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1
    assert st._ple_host_layer_steps() == 1

    emb.disable_hostids("test: the layer stopped taking steps")
    drafts.bump_generation(11)
    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_steps == 1, "a step that staged nothing was counted"
    assert st._ple_host_layer_steps() == 1
    assert st._ple_host_refusals.get("layer-declined-staging") == 1
    coverage = st._ple_host_coverage()
    assert "staged_steps=1" in coverage
    # A layer that stopped taking steps is a boot that is not staging, whatever
    # the worker's own knob says.
    assert "disabled=yes" in coverage


def test_resync_never_runs_inside_a_capture(hostids_env, monkeypatch) -> None:
    """A device sync inside a capture aborts it, and that kills the boot."""
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    monkeypatch.setattr(
        model_state_module.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_syncs == 0
    assert st._ple_host_steps == 0
    assert "capturing" in st._ple_resync_refused
    assert st._ple_host_refusals.get("history-length") == 1


def test_refusal_detail_says_which_side_of_the_count_is_wrong(
    hostids_env, caplog
) -> None:
    """Boot #26 raised 411 of these and could not answer the question it posed.

    The exception carried ``positions [...] != committed-1 [...]`` and only
    ``exc.reason`` was logged, so the boot could establish that the mirror and the
    scheduler disagreed on every step and could not say which one was wrong or by
    how much. ``grep -c committed-1`` on that log is zero.
    """
    import logging

    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    # No repair possible: the budget is gone, so the refusal reaches the log.
    st._ple_resync_slots.update({0: st.PLE_RESYNC_PER_SLOT, 3: st.PLE_RESYNC_PER_SLOT})
    with caplog.at_level(logging.INFO):
        st._stage_host_ple_ids(input_batch, req_states)
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "committed-1" in line, line
    # And the line that counts refusals now has the denominator next to it.
    assert "staged_steps=" in line and "dropped=" in line


def test_replaying_a_stale_draft_block_refuses(hostids_env) -> None:
    """A draft block is consumed once: the same generation is last step's tokens.

    This is the one property of the mirror that no shape check can infer. Two
    consecutive decode steps over an unchanged batch produce an IDENTICAL
    slot->row map with different values, so a map that lines up is not evidence
    of freshness -- and a block that is one step old produces exactly the
    signature the 2026-10-01 14:10 boot showed: the rows whose tokens come from
    history and draft column 0 agree with the device, the row that needs column
    1 does not.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    src = _drafts()  # generation fixed at 1, as an epilogue that staged nothing
    st.set_host_draft_source(src)
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1, "first step should stage"

    emb._host_stage_rows = -1  # the op consumed this step's rows
    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals.get("draft-gen-stale") == 1
    assert emb._host_stage_rows == -1, "a stale block was staged anyway"
    assert st._ple_host_steps == 1

    # The control: the very same batch proceeds the moment a new block exists, so
    # the refusal above is about the generation and not about "second step ever".
    src.bump_generation(src.wait_host_draft_tokens()[2] + 1)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 2, st._ple_host_refusals


def test_mismatch_dump_names_its_inputs(caplog) -> None:  # noqa: ANN001
    """A mismatch that cannot name its tokens is not a diagnosis.

    The arithmetic is proven against the device separately
    (tools/gfx906/cpu-diff-ngram.py: 8 shapes, including the failing one, all match),
    so what a production mismatch can mean is "the inputs differed". That is only
    falsifiable from a log line if the line carries both sides of the inputs and
    the draft block the host built from.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[ROWS - 1, HEADS - 1] = -1  # the LAST head: a [:4] dump would not show it
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ids_arr = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    qsl_arr = np.array([0, 3, 6], dtype=np.int64)
    ctx_arr = np.arange(ROWS * CTX, dtype=np.int64).reshape(ROWS, CTX)
    emb._host_check_inputs = (ids_arr, qsl_arr, ctx_arr)
    emb.note_draft_provenance(
        np.array([[11, 12, 999], [14, 15, 998]], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
        7,
    )

    emb.compare_host_ids(
        torch.from_numpy(dev_ids),
        (
            torch.from_numpy(ids_arr),
            torch.from_numpy(qsl_arr),
            torch.from_numpy(ctx_arr),
        ),
    )

    assert emb._host_input_checks == 1, "the dump ran on a comparison nobody attributed"
    assert emb._host_id_mismatches == 1
    assert emb._hostids_disabled is True
    text = "\n".join(r.getMessage() for r in caplog.records)
    for marker in (
        "first mismatch at 6 rows",
        # The dump now has to admit which witness it consulted: a line saying the
        # inputs agreed is what separates "the formula is wrong" from "the inputs
        # differed and nobody could tell".
        "input gate: agreed",
        "inputs host=",
        "device=",
        "qsl host=",
        "ctx host=",
        "draft gen=7",
        "rows differing=[5]",
        # The heads that actually differ, not the first four: the 14:54 boot
        # truncated every row to [:4] and its second bad row therefore looked
        # identical on both sides in the line meant to explain it.
        f"row {ROWS - 1} differs in 1 head(s) [{HEADS - 1}]",
    ):
        assert marker in text, f"missing {marker!r} in:\n{text}"
    # Consumed, so a later step cannot be compared against this step's inputs.
    assert emb._host_check_dev_inputs is None


def test_a_dump_that_cannot_name_the_device_side_says_so(
    caplog,
) -> None:
    """Silence about a missing half is worse than a line admitting it.

    The 14:54 dump dropped the input half without comment, which is why it read
    as a complete diagnosis for a whole round of analysis. A dump that is missing
    a side has to say which one.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[1, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb._host_check_inputs = (
        np.arange(ROWS, dtype=np.int64),
        np.arange(ROWS + 1, dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )

    emb.compare_host_ids(torch.from_numpy(dev_ids))  # no device inputs anywhere

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "inputs missing on the device side" in text, text
    assert "inputs host=" not in text, text
    # P1.10b changed what "cannot name the device side" costs: it used to disable the
    # knob, which meant a harness with no visibility took the feature away. Now it
    # refuses, so the mirror is neither blamed nor cleared.
    assert emb._hostids_disabled is False
    assert emb._host_id_mismatches == 0
    assert emb._host_input_skips == 1


def test_the_dump_prints_the_heads_that_differ_not_the_first_four(
    caplog,
) -> None:
    """``[:4]`` is only informative when the disagreement starts at head 0.

    The 14:54 boot reported two bad rows and printed four columns of each, so
    the row whose disagreement lived past column 3 looked identical on both
    sides -- the log line said "row 3 differs" and then showed row 3 matching.
    Eight heads here, disagreement at 5 and 7, so a first-four dump cannot pass.
    """
    emb = StagingEmbedding()
    heads = 8
    host = np.zeros((3, heads), dtype=np.int64)
    dev = host.copy()
    dev[2, 5] = 7
    dev[2, 7] = 9

    line = emb._id_dump(3, host, dev, None)

    assert "row 2 differs in 2 head(s) [5, 7]" in line, line
    assert "host row2=[0, 0]" in line, line
    assert "dev  row2=[7, 9]" in line, line
    assert "host row2=[0, 0, 0, 0]" not in line, line


def test_staging_captures_the_device_side_of_the_inputs(  # noqa: ANN201
    hostids_env, caplog
):
    """Staging is the only place both sides of the comparison are in scope.

    The 14:54 boot printed the draft half of its mismatch and silently dropped the
    input half. The compare read ``_host_check_dev_inputs`` and the only writer was
    ``forward()``, whose Python body dynamo runs at trace time and never again on a
    compiled path -- so on the boot that mattered the attribute was whatever tracing
    left behind. Staging is plain Python in the worker, so it hands over the runner's
    own arrays, and the gate reads those.

    What changed in P1.10b is what the compare does with them: it checks them. The
    ids under comparison below are hand-made, so the host half is restated to a step
    that agrees with them -- production stages both halves of the same step, and the
    agreeing path is what a mismatch dump needs. The capture itself is the subject.
    """
    hostids_env["mode"] = 2
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1, st._ple_host_refusals
    captured = emb._host_check_dev_inputs
    assert captured is not None, "staging did not capture the device inputs"
    assert captured[0] is input_batch.input_ids, (
        "staging captured a copy where it should capture the runner's buffer"
    )
    assert emb._host_check_inputs is not None, "staging kept no host side"

    # Both halves now describe the step whose ids the test hand-builds, so the gate
    # agrees and the dump is the thing under test.
    _agree_inputs(emb)
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[0, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb.compare_host_ids(torch.from_numpy(dev_ids))

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "inputs host=" in text, f"the input half is missing again:\n{text}"
    assert "device=" in text, text
    assert "SKIPPED a comparison" not in text, text
    assert emb._host_check_dev_inputs is None, "captured inputs were not consumed"


# ---------------------------------------------------------------------------
# P1.10b: a comparison is attributed by what the two sides read, not by counting
# entries into the op.
#
# P1.9 decided attribution from inside ``forward()``, by stamping an op-entry
# counter and copying the compute's inputs aside as captured graph nodes. Both
# halves were wrong. The counter counts op entries, and under piecewise graphs the
# main pass is graph-replayed and never enters the op: a boot's own dump printed
# ``op entry=42`` against ``stage seq=2``, so "gap == 1" described the harness's
# bookkeeping and not the step. And the snapshot sits inside the compiled region,
# which the AOT cache owns -- the key directory for this shape did not change for
# eight days across dozens of source edits, so whatever was traced is what ran,
# knob or no knob. The one boot that did execute those copies (a drafter-off boot,
# therefore a brand-new key) died of ``hipErrorIllegalAddress`` in CUDA-graph
# memory profiling.
#
# What replaces it asks the question directly, in eager code, at the split point:
# are the arrays the compute read the arrays the host mirrored? If not, there is
# nothing to attribute, so the comparison is refused and counted -- refused, not
# disabled, because a harness that cannot see the inputs has no business taking
# away the feature it is harnessing.
# ---------------------------------------------------------------------------


def test_the_differential_harness_keeps_nothing_in_the_compiled_region() -> None:
    """P1.9's probes stay deleted; this is what keeps them deleted.

    Asserted on the class's surface rather than on source text, because a formatter
    owns the text and dynamo owns ``forward()``. The methods below existed to observe
    the compute from inside the region the AOT cache decides how to run, and the boot
    that first executed one of them is the boot that crashed.
    """
    for gone in (
        "_snapshot_compute_inputs",
        "_read_input_snapshot",
        "_alloc_snap_scratch",
        "_begin_invocation",
    ):
        assert not hasattr(Qwen4ExpNGramEmbedding, gone), gone


def test_a_disagreement_in_the_inputs_refuses_the_comparison() -> None:
    """Two halves that describe different tokens are not a comparison at all.

    Boot #6's device ids invert -- by CRT, exactly -- to a token that appears nowhere
    in the staged step, because under MTP four passes per step write their own rows
    into one static ids buffer. Against the row-count gate that arrived as
    ``first mismatch at 6 rows`` and disabled a mirror which every offline shape
    test since has proven correct.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[2, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    other = ran_ids.copy()
    other[1] = 4  # the token the other MTP pass had put in the runner's array
    emb._host_check_dev_inputs = (
        torch.from_numpy(other),
        torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(dev_ids), emb._host_check_dev_inputs)

    assert emb._host_input_skips == 1, "the input-agreement gate did not fire"
    assert emb._host_id_mismatches == 0, "a refused comparison was counted as a verdict"
    assert emb._hostids_disabled is False, (
        "a harness limitation disabled the feature it was harnessing"
    )


def test_the_refusal_names_the_position_and_both_values(caplog) -> None:
    """ "Inputs differed" is only evidence if a reader can check it.

    The CRT work on boot #6 turned a dump into an exact token pair because the line
    carried the values; a refusal that said only "inputs differ" would need the same
    inversion all over again, this time to prove the gate was right.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    other = ran_ids.copy()
    other[1] = 4
    emb._host_check_dev_inputs = (
        torch.from_numpy(other),
        torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "SKIPPED a comparison" in text, text
    assert "input_ids differ at 1 of 6: device=4 host=101" in text, text
    # The other half of the honesty rule: the line says what the gate checked. It
    # used to confess that the window was never compared; P1.11 compares it when the
    # worker hands it over, and counts the steps where it does not.
    assert "ngram context is never compared" not in text, text
    assert "handed over no device window" in text, text


def test_the_refusal_prints_both_id_arrays_so_rows_can_be_attributed(caplog) -> None:
    """P1.13: a refusal has to name rows, not only a position.

    Boot #11's single sign of the per-pass patchwork was
    ``input_ids differ at 0 of 8 ... (2 of 8 positions)``. That proves the ids buffer
    the op read is not the one the worker staged, but it cannot say *which row* is
    foreign, and "which row was authored by which pass" is the only question the
    differential has left. Rows are contiguous groups of ``heads_per_ngram`` in the
    flat array, so printing both arrays turns the refusal into the map -- and it is
    free at the only size this path runs (a compared step is pure decode, so the
    array is ``requests * (num_spec + 1)`` long).
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    other = ran_ids.copy()
    other[1] = 4
    other[4] = 7
    emb._host_check_dev_inputs = (
        torch.from_numpy(other),
        torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "(2 of 6 positions)" in text, text
    assert "device_ids=[100, 4, 102, 103, 7, 105]" in text, text
    assert "host_ids=[100, 101, 102, 103, 104, 105]" in text, text


def test_the_id_array_render_is_bounded_and_says_so() -> None:
    """A diagnostic on this path must not be able to grow with the batch.

    The cap is only honest if the line admits it fired: a truncation that looks like
    the whole array is worse than no array, because the reader counts rows in what
    they can see.
    """
    short = ple_layer_module._bounded_ids(np.arange(6))
    assert short == "[0, 1, 2, 3, 4, 5]", short
    long_ = ple_layer_module._bounded_ids(np.arange(20), cap=16)
    assert long_.endswith(", ... +4 more]"), long_
    assert ", 16]" not in long_, long_


def test_a_padded_query_boundary_beyond_the_host_rows_is_not_a_disagreement() -> None:
    """The gate has to be able to go green, or every refusal test is vacuous.

    The runner's ``query_start_loc`` is padded to the batch capacity and the host's
    is the rows it built, so the two arrays are routinely different lengths and
    agreement means equal over the rows. A gate that refused that shape would refuse
    every captured step -- and would pass every red control written for it.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    emb._host_check_dev_inputs = (
        torch.from_numpy(ran_ids),
        torch.tensor([0, 3, 6, 6, 6], dtype=torch.long),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    assert emb._host_input_checks == 1, (
        f"refused a padded but agreeing step: skips={emb._host_input_skips}"
    )
    assert emb._host_input_skips == 0
    assert emb._hostids_disabled is False


def test_a_device_array_shorter_than_the_staged_rows_is_not_a_green_light(
    caplog,
) -> None:
    """A prefix compare of a short array compares nothing and reports agreement.

    The runner's array is the padded one, so being shorter than the staged rows
    means the two sides are not describing the same work. Slicing to the shorter
    length -- which is what a careless equality check does -- would call that a
    match and hand the mismatch a green light it never earned.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb._host_check_inputs = (
        np.arange(ROWS, dtype=np.int64) + 100,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    emb._host_check_dev_inputs = (
        torch.from_numpy(np.arange(3, dtype=np.int64) + 100),
        torch.from_numpy(np.array([0, 3], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    assert emb._host_input_skips == 1
    assert emb._host_input_checks == 0
    assert emb._hostids_disabled is False
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "shorter than the staged rows (3 < 6)" in text, text


def test_the_gate_is_not_a_blanket_refusal() -> None:
    """A comparison the harness *can* attribute still has to reach a verdict.

    Every refusal test above is also satisfied by a compare that never compares,
    which is how a diagnostic ends up believing its own silence. The mirror being
    correct is proven elsewhere; what this proves is that a real mismatch is still
    a real finding.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[2, 0] = -1
    _agree_inputs(emb)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS

    emb.compare_host_ids(torch.from_numpy(dev_ids), emb._host_check_dev_inputs)

    assert emb._host_input_checks == 1, "the agreeing step was never compared"
    assert emb._host_input_skips == 0
    assert emb._host_id_mismatches == 1
    assert emb._hostids_disabled is True


def test_the_device_query_boundaries_are_printed_whole(caplog) -> None:
    """A padded qsl is a suspect, so it cannot be printed cut to the host's size.

    The runner pads ``query_start_loc`` to the batch capacity; a padded boundary makes
    the compute split the token stream into requests the host never saw. Slicing the
    array to the host's length, which is what the 15:37 dump did, hides the only
    evidence of that. The gate is satisfied here (equal over the rows, per the test
    above) so the mismatch dump is the thing under test.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[0, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb._host_check_inputs = (
        np.arange(ROWS, dtype=np.int64),
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )

    emb.compare_host_ids(
        torch.from_numpy(dev_ids),
        (
            torch.from_numpy(np.arange(ROWS, dtype=np.int64)),
            # Capacity 5, two real requests: the padded boundaries are the
            # thing being looked for.
            torch.tensor([0, 3, 6, 6, 6], dtype=torch.long),
            torch.zeros((ROWS, CTX), dtype=torch.long),
        ),
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "qsl host=[0, 3, 6] device=[0, 3, 6, 6, 6]" in text, text
    assert emb._host_input_checks == 1


# ---------------------------------------------------------------------------
# P1.11: the window leg
#
# Boot #8's CRT inversion put the single wrong term exactly in the cell the
# per-request ngram window feeds: the host mirrored 1156 where the device read 4.
# Every other input agreed, and the gate could not say anything about the window
# because it had convinced itself the worker had no device-side counterpart for
# it. It has one -- Qwen4ExpModelState owns the buffer, fills it, and hands a view
# of it to the forward -- so these tests pin the third leg: window agreement is
# required for a comparison to run, a window refusal is refused rather than
# blamed, and a window nobody handed over is announced instead of assumed.
# ---------------------------------------------------------------------------


def _window_case(
    emb: StagingEmbedding,
    *,
    host_ctx: np.ndarray,
    dev_ctx: Any,
    qsl: Any = None,
    ids_differ: bool = False,
) -> None:
    """Line up ids and qsl so only the window can disagree.

    The gate checks ids and query boundaries first and returns on the first
    disagreement, so a window test that also had a token difference would be
    testing the token check. Everything here is equal except what is passed in.
    """
    ids = np.arange(ROWS, dtype=np.int64) + 100
    bounds = np.array([0, 3, 6], dtype=np.int64) if qsl is None else qsl
    ngram = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    if ids_differ:
        ngram = ngram.copy()
        ngram[2, 0] = -1
    emb._host_check_ids = torch.from_numpy(ngram)
    emb._host_check_rows = ROWS
    emb._host_check_inputs = (ids, bounds, host_ctx)
    emb._host_check_dev_inputs = (
        torch.from_numpy(ids.copy()),
        torch.from_numpy(np.asarray(bounds, dtype=np.int64)),
        dev_ctx,
    )


def test_a_window_the_device_does_not_have_refuses_the_comparison() -> None:
    """One cell of the window differs: the host mirrored a different prompt.

    The ids the host computed are then a function of tokens the device never read,
    so calling the difference a mirror bug would be blaming the wrong suspect --
    which is precisely the mistake the boot #6 analysis almost made.
    """
    emb = StagingEmbedding()
    host_ctx = np.zeros((ROWS, CTX), dtype=np.int64)
    dev_ctx = np.zeros((CAPACITY, CTX), dtype=np.int64)
    dev_ctx[1, 0] = 4  # the boot #8 shape: a stale cell where the host has a token
    _window_case(emb, host_ctx=host_ctx, dev_ctx=torch.from_numpy(dev_ctx))

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_skips == 1, "the window leg did not refuse"
    assert emb._host_id_mismatches == 0, "a refused comparison became a verdict"
    assert emb._host_id_matches == 0
    assert emb._hostids_disabled is False


def test_the_window_refusal_names_the_request_cell_and_both_values(
    caplog,
) -> None:
    """A refusal a reader cannot check is a refusal that has to be re-proven.

    The window is per request, so the position has to be (request, cell) and not a
    flat index: a flat index would be read as a token row, and the two domains are
    already the most confused pair in this feature.
    """
    emb = StagingEmbedding()
    host_ctx = np.zeros((ROWS, CTX), dtype=np.int64)
    host_ctx[1, 0] = 1156
    dev_ctx = host_ctx.copy()
    dev_ctx[1, 0] = 4
    _window_case(emb, host_ctx=host_ctx, dev_ctx=torch.from_numpy(dev_ctx))

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "ngram_context differs at request 1 cell 0: host=1156 device=4" in text, text


def test_a_window_nobody_handed_over_still_compares_ids_and_says_so(
    caplog,
) -> None:
    """The third leg being dark is not the same as the third leg passing.

    An integration that does not hand over the buffer (a stub, or an older caller)
    must still get the ids and query verdict -- refusing everything would turn a
    missing diagnostic into a disabled feature. But the step is counted, and the
    first one is logged, so nobody reads a green gate as a full one.
    """
    emb = StagingEmbedding()
    _window_case(emb, host_ctx=np.zeros((ROWS, CTX), dtype=np.int64), dev_ctx=None)

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_checks == 1, "the ids verdict was withdrawn with the window"
    assert emb._host_id_mismatches == 0
    assert emb._host_ctx_unchecked == 1
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "the ngram context is NOT compared" in text, text


def test_a_window_with_fewer_requests_than_the_host_refuses(
    caplog,
) -> None:
    """A window for three requests cannot describe a step of six.

    Slicing it to the host's length would compare a prefix of something else and
    call the result agreement -- the same rule the ids buffer applies.
    """
    emb = StagingEmbedding()
    _window_case(
        emb,
        host_ctx=np.zeros((ROWS, CTX), dtype=np.int64),
        dev_ctx=torch.zeros((3, CTX), dtype=torch.long),
    )

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_skips == 1
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "device ngram_context describes 3 requests, the host step has 6" in text


def test_a_window_of_a_different_width_refuses() -> None:
    """A three-cell window is a different ngram_size, not a longer window."""
    emb = StagingEmbedding()
    _window_case(
        emb,
        host_ctx=np.zeros((ROWS, CTX), dtype=np.int64),
        dev_ctx=torch.zeros((CAPACITY, CTX + 1), dtype=torch.long),
    )

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_skips == 1


def test_all_three_legs_agree_and_no_leg_calls_itself_unchecked() -> None:
    """The green path has to be reachable with all three legs present.

    Without this, every refusal test above could be passing because the gate
    refuses everything -- the failure mode that makes a red control worthless.
    """
    emb = StagingEmbedding()
    _agree_inputs(emb)
    # The helper states what both sides read; this is the staged ids buffer the
    # compare runs against, and the row count that says a step is pending. Without
    # them compare_host_ids returns before the gate, which is how this test first
    # failed: a green path that never reached the gate.
    emb._host_check_ids = torch.from_numpy(
        np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    )
    emb._host_check_rows = ROWS

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_checks == 1
    assert emb._host_input_skips == 0
    assert emb._host_ctx_unchecked == 0, "the window leg reported itself unchecked"
    assert emb._host_id_mismatches == 0
    assert emb._host_id_matches == 1, "agreement on inputs did not become a match"


def test_the_dump_prints_the_window_the_device_read(caplog) -> None:  # noqa: ANN
    """ "A dump now shows the window the mismatching rows extended into.

    Boot #8's line printed ``ctx host=[248068, 198] device=n/a``, so the analysis
    had to invert the ids by hand to learn what the device had used for the
    left-context term. Note what this line can and cannot show: a dump only ever
    runs on a comparison the gate attributed, and after P1.11 the gate attributes
    one only when the windows already matched -- so the two halves print equal by
    construction. The point is not that they differ, it is that the reader can see
    the cells the device read instead of taking the host's word for them, and that
    a missing side still says which one is missing. The window is printed whole
    (every request, both cells) because it is per request and tiny; slicing it by
    token row would be a category error.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[2, 0] = -1
    host_ctx = np.zeros((2, CTX), dtype=np.int64)
    host_ctx[0] = [248068, 198]
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (ran_ids, np.array([0, 3, 6], dtype=np.int64), host_ctx)

    emb.compare_host_ids(
        torch.from_numpy(dev_ids),
        (
            torch.from_numpy(ran_ids),
            torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
            torch.from_numpy(host_ctx.copy()),
        ),
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "ctx host=[[248068, 198], [0, 0]]" in text, text
    assert "device=[[248068, 198], [0, 0]]" in text, text
    assert "no device window handed over" not in text, text
    assert "input gate: agreed" in text, text


def test_an_unattributed_window_mismatch_never_reaches_the_dump(caplog) -> None:  # noqa: ANN
    """The refuse-not-disable rule, seen from the dump's side.

    A window disagreement means the host mirrored a different prompt, so printing
    the ids diff next to it would put a formula-looking diagnosis beside a sentence
    that says the inputs were different. The comparison is refused and the ids are
    never printed at all.
    """
    emb = StagingEmbedding()
    host_ctx = np.zeros((2, CTX), dtype=np.int64)
    host_ctx[0] = [248068, 198]
    dev_ctx = host_ctx.copy()
    dev_ctx[0] = [248068, 4]
    _window_case(
        emb,
        host_ctx=host_ctx,
        dev_ctx=torch.from_numpy(dev_ctx),
        ids_differ=True,
    )

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "first mismatch" not in text, text
    assert "ctx host=" not in text, text
    assert emb._host_id_mismatches == 0


def test_the_worker_hands_the_device_window_over_at_staging_time(monkeypatch) -> None:  # noqa: ANN
    """P1.11 lives or dies in ``model_state``, not in the gate.

    Every other test in this file drives ``compare_host_ids`` with a device tuple it
    built itself, so a production caller that still passed ``None`` for the window
    would leave this file entirely green while the leg stayed dark on every boot --
    the phantom-knob failure mode, this time inside our own harness. So this calls
    the real staging function, with a stand-in ``self``, and asserts the object the
    layer is handed is the state's own buffer, by identity: a copy would be the
    previous step's window, and the comparison would be quietly wrong.
    """
    import types

    from vllm.models.qwen4_exp.amd import model_state as ms
    from vllm.models.qwen4_exp.amd import ple_hostids as ph

    class Recorder:
        def __init__(self) -> None:
            self.staged: list[Any] = []

        def check_host_gen(self) -> None:
            return None

        def note_draft_provenance(self, values, row_idx, gen) -> None:  # noqa: ANN
            return None

        def stage_host_step(  # noqa: ANN
            self, ids, qsl, ctx, *, gather, device_inputs, consumed_qsl=None
        ) -> None:
            # ``consumed_qsl`` is P1.14's addition to the production signature. A
            # stub narrower than production's does not catch a wiring change, it
            # just stops the staging loop with a TypeError -- which the producer
            # swallows by design -- so it has to be restated here when it moves.
            self.staged.append((gather, device_inputs))

    class History:
        pending: list[Any] = []
        window = np.zeros((4, 8), dtype=np.int64)
        length = 0

        def drain(self) -> None:
            return None

    class Drafts:
        def wait_host_draft_tokens(self):  # noqa: ANN
            return np.zeros((2, 3), dtype=np.int64), np.array([0, 1], dtype=np.int64), 1

    layer = Recorder()
    window_buf = torch.zeros((4, CTX), dtype=torch.long)
    self_ = types.SimpleNamespace(
        uses_ngram_embedding=True,
        ngram_context_len=CTX,
        ngram_eos_token_id=EOS,
        _ple_host_layers=[layer],
        _ple_host_gen_layers=[layer],
        _host_drafts=Drafts(),
        _ple_host_history=History(),
        _last_draft_gen=0,
        _ple_host_steps=0,
        # The state's own device buffer. Identity is the assertion below, so this
        # has to be one object, not something that merely equals it.
        ngram_context=window_buf,
        _note_host_refusal=lambda reason: None,
        disable_host_staging=lambda why: None,
    )
    monkeypatch.setattr(ph, "hostids_mode", lambda: 2)
    monkeypatch.setattr(
        ms,
        "build_decode_step_inputs",
        lambda *a, **k: (
            np.arange(6, dtype=np.int64),
            np.array([0, 3, 6], dtype=np.int64),
            np.zeros((2, CTX), dtype=np.int64),
        ),
    )

    ms.Qwen4ExpModelState._stage_host_ple_ids(
        self_,
        types.SimpleNamespace(
            num_reqs=2,
            input_ids=torch.zeros(CAPACITY, dtype=torch.long),
            query_start_loc=torch.zeros(3, dtype=torch.long),
            idx_mapping_np=np.array([0, 1], dtype=np.int64),
            num_scheduled_tokens=[3, 3],
        ),
        types.SimpleNamespace(num_computed_tokens_np=np.array([4, 4], dtype=np.int64)),
    )

    assert len(layer.staged) == 1, "the staging loop never ran"
    gather, device_inputs = layer.staged[0]
    assert gather is False, "mode 2 must not gather"
    assert device_inputs is not None, "no device inputs were handed over"
    assert device_inputs[2] is window_buf, (
        "the gate was not handed the state's own ngram_context buffer"
    )


# ---------------------------------------------------------------------------
# P1.12 -- the dump reads the compute's own packed workspace
# ---------------------------------------------------------------------------
#
# Why this leg exists: the q=3 boot (2026-10-02 11:53) is the first step where
# ids, query_start_loc AND the ngram window were all proven identical on both
# sides, and a faithful CPU replay of _compute_ngram_ids on those exact inputs
# reproduced the HOST ids exactly -- not the device's. So the arrays the caller
# can see do not explain the array the device produced, and what is left is
# padded_buffer: the compute's own scratch, written from input_ids inside the
# compiled piece and by nobody else. Its address space is the only remaining
# witness.


def _mismatch_row2_with_workspace(
    emb: StagingEmbedding, packed_cell: int, *, four: bool = True
) -> None:
    """Drive one attributed mismatch at row 2 with a workspace under our control.

    Row 2 of a ``[0, 3, 6]`` batch is request 0, column 2, so -- by the same
    ``context = cat([ngram_context, packed])`` and ``adjusted_columns`` arithmetic
    the compute uses -- the ngram=2 group reads its left context from
    ``packed[0, 1]`` and the ngram=3 group from ``packed[0, 0]``.
    """
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[2, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ids_arr = np.array([10, 11, 12, 14, 15, 16], dtype=np.int64)
    qsl_arr = np.array([0, 3, 6], dtype=np.int64)
    ctx_arr = np.arange(ROWS * CTX, dtype=np.int64).reshape(ROWS, CTX)
    packed = torch.full((4, 64), EOS, dtype=torch.long)
    packed[0, 0], packed[0, 1], packed[0, 2] = 10, packed_cell, 12
    packed[1, 0], packed[1, 1], packed[1, 2] = 14, 15, 16
    emb._host_check_inputs = (
        (ids_arr, qsl_arr, ctx_arr, packed) if four else (ids_arr, qsl_arr, ctx_arr)
    )
    emb.compare_host_ids(
        torch.from_numpy(dev_ids),
        (
            torch.from_numpy(ids_arr),
            torch.from_numpy(qsl_arr),
            torch.from_numpy(ctx_arr),
        ),
    )


def test_the_dump_names_the_workspace_cell_that_fed_the_differing_row(
    caplog,
) -> None:
    """The dump must say which cell of the compute's scratch fed the bad row.

    "The inputs agreed and the ids still differ" is only actionable if the line
    also reports what the hash read out of its own workspace at that position:
    the token means the arithmetic is the suspect, anything else means the
    compute consumed an ids tensor that is not the one the op was handed.
    """
    emb = StagingEmbedding()
    with caplog.at_level("WARNING"):
        _mismatch_row2_with_workspace(emb, 11)
    assert emb._hostids_disabled is True, "the comparison never reached its dump"
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "packed device=" in text, text
    assert "row2ngram2=packed[0,1]=11" in text, text
    assert "row2ngram3=packed[0,0]=10" in text, text


def test_the_packed_leg_reports_the_workspace_and_not_the_hope(caplog) -> None:
    """Red control for the leg above: mutate the cell, the line must change.

    A leg that prints a value derived from the caller's arrays would pass the
    test above and tell us nothing at 3 a.m. This one only earns its keep by
    tracking what the scratch actually holds.
    """
    emb = StagingEmbedding()
    with caplog.at_level("WARNING"):
        _mismatch_row2_with_workspace(emb, 3)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "row2ngram2=packed[0,1]=3" in text, text
    assert "row2ngram2=packed[0,1]=11" not in text, text


def test_the_packed_leg_names_its_own_absence(caplog) -> None:
    """A leg that could not run must not read as a leg that agreed."""
    emb = StagingEmbedding()
    with caplog.at_level("WARNING"):
        _mismatch_row2_with_workspace(emb, 11, four=False)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "packed=n/a(stager handed no workspace)" in text, text


def test_staging_hands_the_dump_the_layer_s_own_workspace(hostids_env) -> None:
    """Production wiring, not a hand-built tuple.

    Every other test here assembles ``_host_check_inputs`` itself, so none of
    them can notice the stager forgetting to pass the workspace -- and a dump
    whose new leg is always ``n/a`` is the phantom-metric failure in its purest
    form: a reassuring absence that looks like a working instrument.
    """
    hostids_env["mode"] = 2
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)

    assert emb._host_check_inputs is not None, "staging kept no host side"
    assert (
        len(emb._host_check_inputs) > 3
        and emb._host_check_inputs[3] is emb.padded_buffer
    ), "the dump was not handed the workspace the compute writes into"
