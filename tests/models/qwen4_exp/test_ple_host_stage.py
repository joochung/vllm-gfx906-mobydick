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
    disable_hostids = Qwen4ExpNGramEmbedding.disable_hostids

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
        self._rows = rows

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
    disable_host_staging = model_state_module.Qwen4ExpModelState.disable_host_staging
    _note_host_refusal = model_state_module.Qwen4ExpModelState._note_host_refusal

    def __init__(self, slots: int = 8) -> None:
        self.uses_ngram_embedding = True
        self.ngram_context_len = CTX
        self.ngram_eos_token_id = EOS
        self._host_drafts = None
        self._ple_host_layers: list | None = None
        self._ple_host_refusals: dict[str, int] = {}
        self._ple_host_refusal_logged: set[str] = set()
        self._ple_host_steps = 0
        self._ple_host_tokens = None
        self.slots = slots

    def staged_rows(self) -> list[int]:
        return [layer._host_stage_rows for layer in self._ple_host_layers or []]


class FakeDrafts:
    def __init__(self, values: np.ndarray, row_idx: np.ndarray) -> None:
        self._result = (values, row_idx, 1)

    def wait_host_draft_tokens(self):
        return self._result


def _batch_and_states(num_reqs: int = 2, q: int = 3, *, uva: bool = True):
    """A pure-decode batch over slots 0 and 3, with a UVA-shaped host mirror."""
    slots = [0, 3][:num_reqs]
    idx_mapping = np.array(slots, dtype=np.int64)
    scheduled = np.array([q] * num_reqs, dtype=np.int32)
    computed = np.zeros(8, dtype=np.int64)
    computed[slots] = 5  # context is available for every row
    tokens = np.zeros((8, 64), dtype=np.int64)
    for slot in slots:
        tokens[slot, :] = np.arange(64) + 1000 * (slot + 1)
    all_token_ids = (
        SimpleNamespace(_uva_buf=SimpleNamespace(np=tokens))
        if uva
        else SimpleNamespace(_uva_buf=None)
    )
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping_np=idx_mapping,
        num_scheduled_tokens=scheduled,
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


def test_no_host_view_of_all_token_ids_disables(hostids_env) -> None:
    hostids_env["layers"] = [StagingEmbedding()]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states(uva=False)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_layers == [], "kept trying on a platform with no host view"
    assert st._ple_host_steps == 0


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
    # slots, positions and drafts imply. Row 0 of each request is the last sampled
    # token from the UVA mirror at the slot's computed position; the rest are the
    # staged drafts.
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
