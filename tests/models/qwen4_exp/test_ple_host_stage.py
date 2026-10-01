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
    stage_host_step = Qwen4ExpNGramEmbedding.stage_host_step
    take_host_stage = Qwen4ExpNGramEmbedding.take_host_stage
    compare_host_ids = Qwen4ExpNGramEmbedding.compare_host_ids
    note_draft_provenance = Qwen4ExpNGramEmbedding.note_draft_provenance
    _id_dump = Qwen4ExpNGramEmbedding._id_dump
    disable_hostids = Qwen4ExpNGramEmbedding.disable_hostids
    _alloc_gen_scratch = Qwen4ExpNGramEmbedding._alloc_gen_scratch
    _gen_scratch = Qwen4ExpNGramEmbedding._gen_scratch
    check_host_gen = Qwen4ExpNGramEmbedding.check_host_gen
    _fire_gen = Qwen4ExpNGramEmbedding._fire_gen

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
        self._host_check_inputs: tuple[Any, Any, Any] | None = None
        self._host_check_draft: tuple[Any, Any, int] | None = None
        self._host_check_dev_inputs: Any = None
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

    emb.compare_host_ids(device)

    assert emb._hostids_disabled is True
    assert emb._host_id_mismatches == 1
    assert emb._host_check_rows == -1, "the check was not consumed"


def test_compare_passes_and_is_consumed_on_agreement() -> None:
    emb = StagingEmbedding()
    ids = np.full((ROWS, HEADS), 42, dtype=np.int64)
    emb._host_check_ids = torch.from_numpy(ids.copy())
    emb._host_check_rows = ROWS
    emb.compare_host_ids(torch.from_numpy(ids.copy()))
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

    with _op_env(emb, monkeypatch) as emb:
        _run_op()

    assert emb.lookup_calls == 1, "check mode must not change what runs"
    assert emb._hostids_disabled is False, "a matching check disabled the knob"
    assert emb._host_id_mismatches == 0


def test_op_disables_after_a_check_mismatch(monkeypatch):
    emb = StagingEmbedding()
    emb._host_check_ids = torch.full((ROWS, HEADS), -7, dtype=torch.long)
    emb._host_check_rows = ROWS

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
        self._last_draft_gen = -1
        self._ple_host_tokens = None
        # The real class builds this in __init__ and seeds it at add time, so the
        # stub has to as well: an unseeded window is a refused step, and these
        # tests are about which steps get through.
        self._ple_host_history = HostTokenHistory(slots, CTX + 1)
        for slot in range(slots):
            self._ple_host_history.seed(slot, _slot_tokens(slot)[: _POSITION + 1])
        self.slots = slots

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


def _slot_tokens(slot: int, width: int = 64) -> np.ndarray:
    """The token history a slot carries, shared by the batch and the stub."""
    return np.arange(width, dtype=np.int64) + 1000 * (slot + 1)


_POSITION = 5  # every request in _batch_and_states sits here


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
    all_token_ids = (
        SimpleNamespace(_uva_buf=SimpleNamespace(np=tokens))
        if uva
        else SimpleNamespace(_uva_buf=None)
    )
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping_np=idx_mapping,
        num_scheduled_tokens=scheduled,
        # The runner's own device arrays, padded to capacity. The differential
        # dump reads these (not anything forward() stashed) because torch.compile
        # runs that body at trace time only.
        input_ids=torch.zeros(16, dtype=torch.int32),
        query_start_loc=torch.zeros(4, dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens_np=computed, all_token_ids=all_token_ids
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
    return emb.stage_host_step(inp, qsl, ctx, gather=gather)


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
    input_batch, req_states, _ = _batch_and_states()

    assert st.stage_host_tokens(*_commit(done=True)) is True
    # The commit carries one accepted token per request; the device's post-update
    # kernel advances num_computed by the same count, and so does the scheduler's
    # host copy that the batch carries.
    req_states.num_computed_tokens_np[[0, 3]] += 1
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
    input_batch, req_states, _ = _batch_and_states()

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
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1
    assert st._ple_host_history.pending == 0


def test_dropped_commit_costs_coverage_never_correctness(hostids_env) -> None:
    """A commit that fails validation is dropped, and the exact-length check shows it.

    The device advanced its counter whether or not the host mirror got the tokens,
    so the mirror is one token short afterwards. The check must refuse the steps it
    can no longer vouch for rather than build ids from a history with a hole in it
    -- which is the same guarantee a full queue gives (``MAX_PENDING`` drops the new
    commit), paid for here by one refused step instead of a wrong embedding.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    slots, rows, counts, event = _commit(done=True)
    # A commit claiming more accepted tokens than the row holds.
    assert (
        st.stage_host_tokens(slots, rows, np.array([9, 1], dtype=np.int32), event)
        is False
    )
    assert "stage-tokens-rejected" in st._ple_host_refusals

    # The device advanced anyway, so the mirror now lags by exactly one token.
    req_states.num_computed_tokens_np[[0, 3]] += 1
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals.get("history-length") == 1
    assert st._ple_host_steps == 0, "staged ids built from a history with a hole"

    # The next commit closes the gap: two accepted tokens per request, and the
    # counter advances by the same amount the device advanced.
    assert (
        st.stage_host_tokens(slots, rows, np.array([2, 2], dtype=np.int32), event)
        is True
    )
    req_states.num_computed_tokens_np[[0, 3]] += 1
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1


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

    assert emb._host_id_mismatches == 1
    assert emb._hostids_disabled is True
    text = "\n".join(r.getMessage() for r in caplog.records)
    for marker in (
        "first mismatch at 6 rows",
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
    """The dump has to survive torch.compile, which forward() does not.

    The 14:54 boot printed the draft half of its mismatch and silently dropped
    the input half. The compare read ``_host_check_dev_inputs`` and the only
    writer was ``forward()``, whose Python body dynamo runs at trace time and
    never again on a compiled path -- so on the boot that mattered the attribute
    was whatever tracing left behind. Staging is plain Python in the worker, and
    it is the one place both sides of the comparison are in scope at once, so it
    captures the runner's device arrays and the compare falls back to them when
    the op has nothing of its own to hand over.
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
    assert captured[0] is input_batch.input_ids

    # The compiled path: the op hands over nothing, and the dump still names
    # both sides of the inputs.
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[0, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb.compare_host_ids(torch.from_numpy(dev_ids))

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "inputs host=" in text, f"the input half is missing again:\n{text}"
    assert "device=" in text, text
    assert "inputs missing on the device side" not in text, text
    assert emb._host_check_dev_inputs is None, "captured inputs were not consumed"
