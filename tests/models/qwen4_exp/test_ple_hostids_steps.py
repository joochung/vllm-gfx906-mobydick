# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the host rebuild of a decode step's PLE inputs.

:func:`build_decode_step_inputs` claims that three host-visible channels -- the
UVA mirror of ``all_token_ids``, ``num_computed_tokens_np``, and the staged draft
mirror -- are enough to produce the same ``input_ids`` / ``query_start_loc`` /
``ngram_context`` the device would have used. Two groups of tests hold that up,
and they are testing different things:

* ``test_position_convention_*`` transcribes ``_post_update_kernel`` (the Triton
  kernel that advances ``total_len``, ``num_computed_tokens``,
  ``last_sampled_tokens`` and the ``all_token_ids`` stores) and checks the two
  load-bearing invariants the builder relies on: ``total_len ==
  num_computed + 1``, and ``all_token_ids[num_computed] == last_sampled``. This
  is a real differential test against the device's own arithmetic.

* ``test_matches_device_formula_*`` transcribes ``_prepare_ngram_context`` and
  the row assembly, and checks the *builder* reproduces them over a randomised
  multi-step run. It cannot catch a wrong assumption (both sides would be wrong
  together); it catches indexing, dtype, ordering and per-request interleaving
  mistakes, which is what a loop like this is actually likely to get wrong.

Refusals are tested as a group because "fall back to the device path" is the
correct answer to every one of them, and a builder that guesses instead would
produce a plausible ngram id for the wrong context -- the failure mode this whole
effort has been built around.
"""

import numpy as np
import pytest

from vllm.models.qwen4_exp.amd.ple_hostids import (
    HostIdsRefused,
    build_decode_step_inputs,
)

CTX = 2  # ngram_context_len == ngram_size - 1 == 3 - 1
K = 3  # num_speculative_tokens: a decode step is 1 bonus + K drafts
EOS = 151645


# --------------------------------------------------------------------------
# Transcriptions of the device side (keep in sync with the sources named)
# --------------------------------------------------------------------------


class DeviceReqState:
    """``_post_update_kernel`` for one request, transcribed.

    vllm/v1/worker/gpu/input_batch.py:544-602. Deliberately literal: the point
    of this class is that the counter arithmetic under test is *this* arithmetic,
    not what I believe it to be.
    """

    def __init__(self, max_model_len: int):
        self.uva = np.zeros(max_model_len, dtype=np.int32)
        self.total_len = 0
        self.num_computed = 0
        self.last_sampled = -1

    def post_update(
        self,
        query_len: int,
        sampled_tokens: np.ndarray,
        num_rejected: int,
    ) -> None:
        num_sampled = int(sampled_tokens.size)
        total_len = self.total_len
        if num_sampled > 0:
            self.last_sampled = int(sampled_tokens[num_sampled - 1])
            self.total_len = total_len + num_sampled
        for i in range(num_sampled):
            self.uva[total_len + i] = int(sampled_tokens[i])
        computed_delta = query_len - num_rejected
        if computed_delta != 0:
            self.num_computed += computed_delta


def device_prepare_ngram_context(
    state: DeviceReqState, context_len: int = CTX
) -> np.ndarray:
    """``Qwen4ExpModelState._prepare_ngram_context`` (model_state.py:201-228)."""
    end = state.num_computed
    offsets = np.arange(-context_len, 0, dtype=np.int64)
    indices = end + offsets
    valid = indices >= 0
    indices = np.maximum(indices, 0)  # token_indices.clamp_min_(0)
    out = np.where(valid, state.uva[indices], EOS).astype(np.int64)
    return out


def device_input_ids(state: DeviceReqState, drafts: np.ndarray) -> np.ndarray:
    """The rows ``combine_sampled_and_draft_tokens`` places on the device."""
    return np.concatenate(([state.last_sampled], drafts)).astype(np.int64)


# --------------------------------------------------------------------------
# The derivation: the two invariants the builder is built on
# --------------------------------------------------------------------------


def test_position_convention_holds_across_speculative_steps():
    rng = np.random.default_rng(20261001)
    # Long enough for the whole run: 60 speculative steps commit up to K+1
    # tokens each.
    state = DeviceReqState(512)
    # After a prefill of P tokens: P stored, P-1 computed. The token at index
    # P-1 is the last prompt token, which is what the first decode step feeds.
    prompt = rng.integers(1000, 2000, size=5, dtype=np.int32)
    state.total_len = prompt.size
    state.num_computed = prompt.size - 1
    state.uva[: prompt.size] = prompt
    state.last_sampled = int(prompt[-1])

    for _ in range(60):
        assert state.total_len == state.num_computed + 1, (
            "a request carries exactly one more stored token than computed "
            "token; the builder reads this step's first row at num_computed"
        )
        assert state.uva[state.num_computed] == state.last_sampled, (
            "the token stored at index num_computed is the one the sampler "
            "produced for this step's first row"
        )

        drafts = rng.integers(3000, 4000, size=K, dtype=np.int32)
        accepted = int(rng.integers(1, K + 2))  # 1..K+1 committed rows
        # sampled_tokens = the accepted sequence: the accepted draft inputs, then
        # the newly sampled token that the next step feeds first.
        sampled = np.concatenate(
            (drafts[: accepted - 1], [int(rng.integers(5000, 6000))])
        )
        assert sampled.size == accepted
        state.post_update(
            query_len=K + 1,
            sampled_tokens=sampled,
            num_rejected=(K + 1) - accepted,
        )


def test_position_convention_after_a_full_rejection():
    """One row accepted, all drafts rejected: the common worst case."""
    state = DeviceReqState(32)
    state.total_len, state.num_computed = 4, 3
    state.uva[:4] = [11, 12, 13, 14]
    state.last_sampled = 14
    state.post_update(query_len=K + 1, sampled_tokens=np.array([99]), num_rejected=K)
    assert state.num_computed == 4
    assert state.total_len == 5
    assert state.uva[state.num_computed] == 99 == state.last_sampled
    # The rejected drafts must not have been stored anywhere.
    assert state.uva[5:].tolist() == [0] * 27


# --------------------------------------------------------------------------
# Fixtures for the builder
# --------------------------------------------------------------------------


class Harness:
    """A batch of ``DeviceReqState`` laid out as the runner would present them."""

    def __init__(self, slots: list[int], max_model_len: int = 64):
        self.max_model_len = max_model_len
        self.states: dict[int, DeviceReqState] = {
            s: DeviceReqState(max_model_len) for s in slots
        }
        self.all_token_ids = np.zeros((len(slots) + 8, max_model_len), dtype=np.int32)
        self.num_computed = np.zeros(len(slots) + 8, dtype=np.int32)

    def sync(self, slot: int) -> None:
        """Publish a device-side request state into the host-visible arrays."""
        state = self.states[slot]
        self.all_token_ids[slot, : state.total_len] = state.uva[: state.total_len]
        self.num_computed[slot] = state.num_computed

    def stage(self, batch: list[int], rng: np.random.default_rng) -> np.ndarray:
        """Fresh drafts for the next step, laid out in ``batch`` order.

        These are the values the epilogue mirror holds: tokens that are nowhere
        in ``all_token_ids``, because they are speculative.
        """
        staged = np.zeros((len(batch), K), dtype=np.int32)
        for row in range(len(batch)):
            staged[row] = rng.integers(3000, 4000, size=K, dtype=np.int32)
        return staged

    def step(
        self,
        batch: list[int],
        staged: np.ndarray,
        row_idx: np.ndarray,
        rng: np.random.default_rng,
    ) -> None:
        """Run on the device side the step that ``staged`` describes.

        Causality matters: ``staged`` holds this step's *draft rows*, which
        post_update consumes (an accepted prefix becomes committed tokens) and
        which must never be confused with the next step's drafts.
        """
        row_of_slot = {int(s): r for r, s in enumerate(row_idx)}
        for slot in batch:
            state = self.states[slot]
            drafts_in = staged[row_of_slot[slot]]
            accepted = int(rng.integers(1, K + 2))
            sampled = np.concatenate(
                (drafts_in[: accepted - 1], [int(rng.integers(5000, 6000))])
            )
            state.post_update(
                query_len=K + 1,
                sampled_tokens=sampled,
                num_rejected=(K + 1) - accepted,
            )
            self.sync(slot)


def _build(harness: Harness, batch: list[int], staged, row_idx):
    return build_decode_step_inputs(
        np.array(batch, dtype=np.int32),
        np.full(len(batch), K + 1, dtype=np.int32),
        harness.num_computed,
        harness.all_token_ids,
        staged,
        row_idx,
        ngram_context_len=CTX,
        eos_token_id=EOS,
    )


# --------------------------------------------------------------------------
# The builder vs the transcribed device formulas
# --------------------------------------------------------------------------


def test_matches_device_formula_over_a_long_run():
    rng = np.random.default_rng(7)
    batch = [0, 3, 5]
    harness = Harness(batch, max_model_len=512)
    for slot in batch:  # a short prompt each
        state = harness.states[slot]
        prompt = rng.integers(1000, 2000, size=6, dtype=np.int32)
        state.total_len, state.num_computed = prompt.size, prompt.size - 1
        state.uva[: prompt.size] = prompt
        state.last_sampled = int(prompt[-1])
        harness.sync(slot)

    row_idx = np.array(batch, dtype=np.int32)
    staged = harness.stage(batch, rng)
    for _ in range(50):
        input_ids, qsl, ctx = _build(harness, batch, staged, row_idx)
        for i, slot in enumerate(batch):
            state = harness.states[slot]
            row = int(np.where(row_idx == slot)[0][0])
            expect_ids = device_input_ids(state, staged[row])
            got = input_ids[int(qsl[i]) : int(qsl[i + 1])]
            assert got.tolist() == expect_ids.tolist(), f"step ids for slot {slot}"
            assert ctx[i].tolist() == device_prepare_ngram_context(state).tolist(), (
                f"context for slot {slot}"
            )
        assert qsl[0] == 0
        assert qsl[-1] == (K + 1) * len(batch)
        # The step the builder just described runs on the device; the drafter then
        # stages a *different* block for the step after it.
        harness.step(batch, staged, row_idx, rng)
        staged = harness.stage(batch, rng)


def test_row_order_follows_the_staged_index_not_batch_position():
    """The dangerous confusion: batch order changed between the two steps.

    The staged rows were published in the *previous* step's order. Here the
    previous batch was ``[5, 0, 3]`` and this step's is ``[0, 3, 5]``, so reading
    the mirror by position would hand every request another request's drafts.
    """
    rng = np.random.default_rng(11)
    prev, now = [5, 0, 3], [0, 3, 5]
    harness = Harness(now)
    for slot in now:
        state = harness.states[slot]
        prompt = np.array([100 + slot, 200 + slot, 300 + slot, 400 + slot], np.int32)
        state.total_len, state.num_computed = prompt.size, prompt.size - 1
        state.uva[: prompt.size] = prompt
        state.last_sampled = int(prompt[-1])
        harness.sync(slot)

    drafts = {s: np.full(K, 1000 + s, dtype=np.int32) for s in now}
    staged = np.array([drafts[s] for s in prev], dtype=np.int32)
    row_idx = np.array(prev, dtype=np.int32)
    # Advance the device state so the mirror's bonus tokens move, without
    # re-staging (the staged block is the previous step's).
    for slot in now:
        state = harness.states[slot]
        state.post_update(
            query_len=K + 1,
            sampled_tokens=np.concatenate((drafts[slot][:1], [9000 + slot])),
            num_rejected=K - 1,
        )
        harness.sync(slot)

    input_ids, qsl, _ = _build(harness, now, staged, row_idx)
    for i, slot in enumerate(now):
        got = input_ids[int(qsl[i]) : int(qsl[i + 1])]
        assert got[1:].tolist() == [1000 + slot] * K, (
            f"slot {slot} got another request's drafts: {got.tolist()}"
        )


def test_first_decode_step_left_fills_eos_like_the_device():
    """pos < context_len: the device EOS-fills the left of the window."""
    harness = Harness([1])
    state = harness.states[1]
    state.uva[0] = 42
    state.total_len, state.num_computed = 1, 0
    state.last_sampled = 42
    harness.sync(1)
    staged = np.full((1, K), 77, dtype=np.int32)
    input_ids, qsl, ctx = _build(harness, [1], staged, np.array([1], np.int32))
    assert ctx[0].tolist() == [EOS, EOS]
    assert input_ids.tolist() == [42, 77, 77, 77]

    state.uva[1] = 43
    state.total_len, state.num_computed = 2, 1
    state.last_sampled = 43
    harness.sync(1)
    _, _, ctx = _build(harness, [1], staged, np.array([1], np.int32))
    assert ctx[0].tolist() == [EOS, 42]


def test_rows_beyond_the_bonus_come_from_the_mirror_not_uva():
    """Rejected drafts leave stale bytes in ``all_token_ids`` past ``total_len``.

    Reading the row's draft slots from the UVA mirror instead of the staged
    mirror would silently use whatever the previous speculation left there.
    """
    harness = Harness([2])
    state = harness.states[2]
    prompt = np.array([5, 6, 7, 8], dtype=np.int32)
    state.total_len, state.num_computed = 4, 3
    state.uva[:4] = prompt
    state.last_sampled = 8
    # Garbage past the committed prefix, exactly what a rejected step leaves.
    state.uva[4:8] = [999, 998, 997, 996]
    harness.sync(2)
    harness.all_token_ids[2, 4:8] = [999, 998, 997, 996]

    staged = np.array([[70, 71, 72]], dtype=np.int32)
    input_ids, _, _ = _build(harness, [2], staged, np.array([2], np.int32))
    assert input_ids.tolist() == [8, 70, 71, 72]


# --------------------------------------------------------------------------
# Refusals: every one of these means "run the device path"
# --------------------------------------------------------------------------


def _ready_harness() -> tuple[Harness, list[int], np.ndarray]:
    batch = [0, 1]
    harness = Harness(batch)
    for slot in batch:
        state = harness.states[slot]
        prompt = np.arange(10 * slot + 1, 10 * slot + 6, dtype=np.int32)
        state.total_len, state.num_computed = prompt.size, prompt.size - 1
        state.uva[: prompt.size] = prompt
        state.last_sampled = int(prompt[-1])
        harness.sync(slot)
    staged = np.array([[30, 31, 32], [40, 41, 42]], dtype=np.int32)
    return harness, batch, staged


def test_refuses_batch_that_is_not_pure_decode():
    harness, batch, staged = _ready_harness()
    queries = np.array([K + 1, 37], dtype=np.int32)  # a prefill joined the batch
    with pytest.raises(HostIdsRefused) as excinfo:
        build_decode_step_inputs(
            np.array(batch, np.int32),
            queries,
            harness.num_computed,
            harness.all_token_ids,
            staged,
            np.array(batch, np.int32),
            ngram_context_len=CTX,
            eos_token_id=EOS,
        )
    assert excinfo.value.reason == "not-pure-decode"

    # Control: the identical call with the prefill removed must succeed, or the
    # test above would pass for an unrelated reason.
    ids, _, _ = build_decode_step_inputs(
        np.array([batch[0]], np.int32),
        np.array([K + 1], np.int32),
        harness.num_computed,
        harness.all_token_ids,
        staged[:1],
        np.array([batch[0]], np.int32),
        ngram_context_len=CTX,
        eos_token_id=EOS,
    )
    assert ids.size == K + 1


def test_refuses_request_with_no_staged_row():
    harness, batch, staged = _ready_harness()
    with pytest.raises(HostIdsRefused) as excinfo:
        _build(harness, batch, staged, np.array([batch[0]], np.int32))
    assert excinfo.value.reason == "row-missing"


def test_refuses_missing_or_empty_staging():
    harness, batch, staged = _ready_harness()
    row_idx = np.array(batch, np.int32)
    with pytest.raises(HostIdsRefused) as excinfo:
        build_decode_step_inputs(
            np.array(batch, np.int32),
            np.full(len(batch), K + 1, np.int32),
            harness.num_computed,
            harness.all_token_ids,
            None,
            row_idx,
            ngram_context_len=CTX,
            eos_token_id=EOS,
        )
    assert excinfo.value.reason == "no-drafts"

    with pytest.raises(HostIdsRefused) as excinfo:
        build_decode_step_inputs(
            np.array([], np.int32),
            np.array([], np.int32),
            harness.num_computed,
            harness.all_token_ids,
            staged,
            row_idx,
            ngram_context_len=CTX,
            eos_token_id=EOS,
        )
    assert excinfo.value.reason == "empty-batch"


def test_refuses_out_of_range_position():
    """The host mirror is optimistic; a wild value must not index the table."""
    harness, batch, staged = _ready_harness()
    harness.num_computed[batch[1]] = harness.max_model_len + 5
    with pytest.raises(HostIdsRefused) as excinfo:
        _build(harness, batch, staged, np.array(batch, np.int32))
    assert excinfo.value.reason == "position-out-of-range"


def test_refuses_padded_negative_slot():
    harness, batch, staged = _ready_harness()
    with pytest.raises(HostIdsRefused) as excinfo:
        _build(harness, [batch[0], -1], staged, np.array(batch, np.int32))
    assert excinfo.value.reason == "negative-slot"


def test_refusal_is_a_valueerror_for_existing_callers():
    """``compute_ngram_ids_host`` documents ValueError; the subclass keeps that."""
    harness, batch, staged = _ready_harness()
    with pytest.raises(ValueError):
        _build(harness, batch, staged, np.array([99], np.int32))


def test_reason_tokens_are_stable():
    """The consumer counts these to prove coverage; renaming one is a contract
    change, so the set is asserted."""
    assert HostIdsRefused("row-missing", "x").reason == "row-missing"
    reasons = {
        "no-drafts",
        "empty-batch",
        "not-pure-decode",
        "row-missing",
        "position-out-of-range",
        "negative-slot",
        "context-width",
        "layout",
    }
    harness, batch, staged = _ready_harness()
    raised: set[str] = set()
    for call in (
        lambda: _build(harness, [], staged, np.array(batch, np.int32)),
        lambda: _build(harness, [batch[0]], None, np.array(batch, np.int32)),
        lambda: _build(harness, [batch[0], -1], staged, np.array(batch, np.int32)),
        lambda: _build(harness, batch, staged, np.array([batch[0]], np.int32)),
    ):
        try:
            call()
        except HostIdsRefused as exc:
            raised.add(exc.reason)
    assert raised <= reasons
