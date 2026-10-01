# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for :class:`HostTokenHistory`, the host mirror of committed tokens.

This class exists because ``req_states.all_token_ids`` is host-*visible* but not
host-*valid*: its prompt cells are staged by the host at add time, while its
generated cells are written by ``_post_update_kernel`` on the GPU, unordered
before a host read in ``prepare_inputs``. The first differential boot caught that
on 2026-10-01. So the tests here are biased toward the two properties that make
the mirror trustworthy rather than merely addressable:

* it advances by exactly the tokens the device appended (``num_sampled``, the
  device's own trim rule), and
* it never claims a token whose copy has not landed -- ``drain`` is non-blocking
  and stops at the first un-landed commit, so ordering survives.
"""

import numpy as np
import pytest

from vllm.models.qwen4_exp.amd.ple_hostids import HostTokenHistory

W = 3  # width = ngram_context_len + 1 = 2 + 1
SLOTS = 4


class FakeEvent:
    """The only method ``HostTokenHistory`` calls on a copy event."""

    def __init__(self, done: bool = False) -> None:
        self.done = done
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.done

    def complete(self) -> None:
        self.done = True


def _stage(history, slots, rows, counts, done=None):
    width = max(len(r) for r in rows)
    matrix = np.zeros((len(rows), width), dtype=np.int32)
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = row
    return history.stage_step(
        np.array(slots, np.int32),
        matrix,
        np.array(counts, np.int32),
        done,
    )


def test_construction_rejects_degenerate_shapes():
    with pytest.raises(ValueError):
        HostTokenHistory(0, W)
    with pytest.raises(ValueError):
        HostTokenHistory(SLOTS, 0)


def test_seed_keeps_the_chronological_tail():
    history = HostTokenHistory(SLOTS, W)
    history.seed(1, [10, 11, 12, 13, 14])
    assert history.length[1] == 5
    assert history.window[1].tolist() == [12, 13, 14]
    # Newest at the right: that cell is the token the next step feeds first.
    assert history.window[1, -1] == 14


def test_seed_of_a_short_request_zeroes_the_unused_left():
    history = HostTokenHistory(SLOTS, W)
    history.seed(2, [42])
    assert history.length[2] == 1
    assert history.window[2].tolist() == [0, 0, 42]
    history.seed(3, [])
    assert history.length[3] == 0


def test_seed_of_a_resumed_request_includes_generated_tokens():
    """Preemption/resume: the scheduler sends the whole list again.

    Re-seeding is what makes slot reuse safe -- a window left over from a
    previous request cannot survive an add.
    """
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1, 2, 3])
    _stage(history, [0], [[9]], [1])
    history.drain()
    assert history.window[0].tolist() == [2, 3, 9]
    history.seed(0, [1, 2, 3, 9, 77])  # resumed with the generated token
    assert history.window[0].tolist() == [3, 9, 77]
    assert history.length[0] == 5


def test_commit_appends_in_order_for_every_batch_size():
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1, 2])
    history.commit(0, [3])
    assert history.window[0].tolist() == [1, 2, 3] and history.length[0] == 3
    history.commit(0, [4, 5])  # accepted bonus + one draft
    assert history.window[0].tolist() == [3, 4, 5] and history.length[0] == 5
    history.commit(0, [6, 7, 8, 9])  # more than the window holds
    assert history.window[0].tolist() == [7, 8, 9] and history.length[0] == 9
    history.commit(0, [])  # a step that committed nothing is a no-op
    assert history.window[0].tolist() == [7, 8, 9] and history.length[0] == 9


def test_stage_never_reads_the_sampled_array():
    """Staging must be safe while the D2H is still in flight.

    Copying at stage time is the bug this class exists to avoid, so the staged
    values are held by reference and only read once the event reports done.
    """

    class Exploding:
        ndim = 2
        shape = (1, 4)

        def __array__(self, dtype=None):
            raise AssertionError("HostTokenHistory read the sampled array")

    history = HostTokenHistory(SLOTS, W)
    assert _stage_history_untouched(history, Exploding()) is True


def _stage_history_untouched(history, sampled) -> bool:
    return history.stage_step(
        np.array([0], np.int32),
        sampled,
        np.array([1], np.int32),
        FakeEvent(False),
    )


def test_drain_waits_for_the_copy_and_keeps_commits_in_order():
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1])
    first, second = FakeEvent(False), FakeEvent(True)
    assert _stage(history, [0], [[2, 3]], [2], first)
    assert _stage(history, [0], [[4]], [1], second)

    # The second commit's data is host-visible, but applying it before the first
    # would put token 4 before tokens 2 and 3.
    assert history.drain() == 0
    assert history.pending == 2
    assert history.window[0].tolist() == [0, 0, 1]

    first.complete()
    assert history.drain() == 2  # both, in order
    assert history.pending == 0
    assert history.window[0].tolist() == [2, 3, 4]
    assert history.length[0] == 4


def test_drain_only_polls_never_blocks():
    history = HostTokenHistory(SLOTS, W)
    event = FakeEvent(False)
    _stage(history, [0], [[5]], [1], event)
    history.drain()
    assert event.queries == 1  # query(), never synchronize()


def test_num_sampled_is_the_device_trim_rule():
    """Rejected drafts sit in the sampled row and must not be appended.

    ``_post_update_kernel`` takes the first ``num_sampled[i]`` entries of row i;
    the mirror uses the same counts, so agreement with the device is structural.
    """
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1, 2])
    matrix = np.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=np.int32)
    assert history.stage_step(
        np.array([0, 1], np.int32),
        matrix,
        np.array([1, 3], np.int32),
        None,
    )
    assert history.drain() == 2
    assert history.window[0].tolist() == [1, 2, 7]  # one accepted, drafts rejected
    assert history.window[1].tolist() == [11, 12, 13]
    assert history.length[0] == 3
    assert history.length[1] == 3


def test_invalidate_drops_an_in_flight_commit_for_that_slot():
    """A request removed while its copy is in flight must not poison its slot.

    The event is the point of the test: the copy lands *after* the slot has been
    handed to a new request, so the drop has to happen at ``invalidate`` time. A
    queue that still carried the dead request's row would append its token to the
    new request, and the builder's exact-length check would then refuse that slot
    for the rest of its life.
    """
    history = HostTokenHistory(SLOTS, W)
    history.seed(2, [1, 2, 3])
    event = FakeEvent(False)
    _stage(history, [2], [[99]], [1], event)
    history.invalidate(2)
    # The slot is handed to a new request while the old copy is still in flight...
    history.seed(2, [7, 8])
    assert history.drain() == 0
    # ...and only afterwards does that copy land.
    event.done = True
    assert history.drain() == 0, "the dead request's row was applied"
    assert history.window[2].tolist() == [0, 7, 8]
    assert history.length[2] == 2


def test_queue_is_bounded_and_counts_what_it_drops():
    history = HostTokenHistory(SLOTS, W)
    for _ in range(HostTokenHistory.MAX_PENDING):
        assert _stage(history, [0], [[1]], [1], FakeEvent(False))
    assert history.pending == HostTokenHistory.MAX_PENDING
    assert _stage(history, [0], [[2]], [1], FakeEvent(False)) is False
    assert history.dropped == 1
    # What a drop costs is a length that lags the device, which the builder then
    # refuses on -- see test_dropped_commit_costs_coverage_never_correctness in
    # test_ple_host_stage.py. It can never show up as a wrong id.
    assert history.length[0] == 0


def test_stage_rejects_inconsistent_shapes():
    history = HostTokenHistory(SLOTS, W)
    cases = [
        # slots, rows, counts, why
        ([9], [[1]], [1], "slot outside the table"),
        ([-1], [[1]], [1], "padding slot"),
        ([0, 1], [[1]], [1], "counts disagree with slots"),
        ([0], [[1, 2, 3]], [5], "more accepted than the row holds"),
        ([0], [[1, 2, 3]], [-1], "negative count"),
    ]
    for slots, rows, counts, _why in cases:
        before = history.dropped
        assert _stage(history, slots, rows, counts) is False, _why
        assert history.dropped == before + 1, _why

    # A sampled array that is not [requests, tokens] cannot be trimmed at all.
    assert (
        history.stage_step(
            np.array([0], np.int32),
            np.zeros((0, 2), np.int32),
            np.array([1], np.int32),
            None,
        )
        is False
    )
    assert history.dropped == len(cases) + 1
    assert history.staged == 0
