# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side ngram id computation for the PLE decode path.

Why this exists
---------------
The ngram table lives in host RAM (``MmapShardedNGramEmbedding``), so the
gather has to happen on the host. But the *ids* it gathers with are produced on
the GPU by ``Qwen4ExpNGramEmbedding._compute_ngram_ids``, which forces a
blocking device->host copy every step:

    pinned_ids.copy_(ngram_ids, non_blocking=False)   # ple_layer.py

``non_blocking=False`` is not a tuning choice. The host reads that buffer
immediately afterwards, so the copy has to have landed -- which means the host
waits for everything queued on the compute stream ahead of it. On this box that
wait (``drain`` in ``ple_prof.py``) is the term that the profiler measures, and
D2' proved the gather itself is no longer the cost: after the sorted-gather
rewrite the residual lookup is page-fault-bound, and ``drain`` is host *idle*
while the GPU catches up.

Removing the round trip means computing the ids where the gather already is.
The host can do that, but not as cheaply as it looks: this runner keeps **no host
array of the step's input ids** (there is no ``input_ids_np`` / ``token_ids_cpu``
here -- ``input_batch.input_ids`` is device-only, and decode rows are placed on
the device by ``combine_sampled_and_draft_tokens``). The host has to *rebuild*
them per request, from three pieces it does have:

* the tokens accepted last step, as a host list-of-lists in ``ModelRunnerOutput``
  (``sampled_token_ids``), which is the row's first token;
* this step's draft tokens, mirrored to pinned host memory by
  ``DraftTokensHandler`` when ``VLLM_GFX906_PLE_HOSTIDS`` is set -- without that
  mirror the host knows 1 of the 4 rows a speculative decode step feeds this
  layer, because ``spec_decode/utils.py`` deliberately never copies drafts to
  the host (``docs/d1-host-ids-design.md`` has the trace);
* the per-request token history in ``req_states.all_token_ids._uva_buf.np``,
  which is pinned host memory that the GPU also writes, and is where the
  two-token ngram context comes from.

Bit-identity is the whole contract
----------------------------------
This module is a *reimplementation* of ``_compute_ngram_ids`` in numpy, which
makes it the one kind of optimization that can be silently wrong: an id that is
off by one is a different row of a 20-million-row table, so the model receives a
plausible embedding of the wrong ngram and nothing raises. Two things hold it in
place.

1. It reproduces the device function's semantics exactly, including the parts
   that are easy to guess wrong:

   * **int64 wraparound.** ``shifted * multiplier`` overflows int64 for ordinary
     token ids. Torch wraps two's complement; numpy wraps too but warns, so the
     arithmetic runs under ``np.errstate(over="ignore")``. ``bitwise_xor`` on
     signed int64 and ``torch.bitwise_xor`` agree because both act on the
     two's-complement bits.
   * **floor remainder.** ``torch.remainder`` signs its result like Python's
     ``%`` (non-negative for a positive divisor, even for a negative dividend).
     ``np.remainder`` is the same floor definition; ``np.mod`` would agree too
     but ``%`` on a Python int would not, so nothing here uses Python ints.
   * **eos segment boundaries.** ``_shift_precompute``/``_shift_apply`` make a
     token blind to anything at or before the nearest preceding eos: a shift is
     only honoured when ``position_in_segment >= shift``. Reproduced below with
     a running maximum of eos positions, which -- like ``torch.cummax`` followed
     by the one-element right shift in the device code -- looks strictly
     backwards, so padding cells after a request's last token cannot influence
     it either way.
   * **the column identity.** The device code indexes the concatenated
     ``[context | packed]`` row at ``columns + ngram_size - 1``, which is the
     token's own row index *only because* ``ngram_context`` is exactly
     ``ngram_size - 1`` wide. That is asserted, not assumed: a context of any
     other width means the device path is reading a different window than this
     one, and the correct response is to refuse, not to compute.

2. ``tests/models/qwen4_exp/test_ple_hostids.py`` compares this function against
   ``_compute_ngram_ids`` itself (on CPU tensors, so it runs without a GPU) and
   requires exact equality on every case -- real config-derived hash buffers
   *and* adversarial ones. A differential test against a duplicate of the math
   would prove nothing; this one calls the device function.

Refusal, not failure
--------------------
Nothing here raises into the engine. ``hostids_enabled()`` gates on the
environment, and ``compute_ngram_ids_host`` raises ``ValueError`` on any layout
it has not been shown to match; the caller catches that and takes the device
path. An optimization that cannot prove its preconditions steps aside -- the
same contract ``ple_prefetch.py`` uses (``disable(reason)``), and for the same
reason: a wrong PLE id is not a crash, it is a wrong answer.
"""

from __future__ import annotations

import os

import numpy as np

PADDING_SENTINEL = -1
"""Mirrors ``MmapShardedNGramEmbedding.PADDING_SENTINEL``.

Kept here as documentation of what the *id producer* may emit. Out-of-range ids
(including this sentinel) are not handled by this module: they are handled by
``MmapShardedNGramEmbedding.forward``, which folds them onto row 0. The host
path deliberately does not pre-fold, so that the fold -- and its warning, and
its ``VLLM_GFX906_PLE_STRICT`` raise -- happens in exactly one place for both
producers.
"""


def hostids_enabled() -> bool:
    """``VLLM_GFX906_PLE_HOSTIDS=1`` turns host-side ids on. Off by default.

    Deliberately read with ``os.environ.get`` rather than registered in
    ``vllm/envs.py``: like the other gfx906 experiment knobs it is local, and
    the cost of that is a benign "Unknown vLLM environment variable" warning at
    boot. Engagement is proven by the activation line the caller logs, never by
    the absence of that warning.
    """
    return os.environ.get("VLLM_GFX906_PLE_HOSTIDS", "0") == "1"


def _as_int64(values: np.ndarray | None, name: str) -> np.ndarray:
    """Return ``values`` as a contiguous int64 array, without a copy if it can be.

    A copy is required (and allowed) if the caller handed over a different
    dtype: this is a per-step host function, so the cost is bounded by a few
    thousand int conversions, and taking ``np.asarray`` seriously (rather than
    asserting the dtype) is what keeps the pinned-buffer view case zero-copy.
    """
    if values is None:
        raise ValueError(f"{name} is required")
    out = np.asarray(values, dtype=np.int64)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


def compute_ngram_ids_host(
    input_ids: np.ndarray,
    query_start_loc: np.ndarray,
    ngram_context: np.ndarray,
    *,
    multipliers: np.ndarray,
    vocab_sizes: np.ndarray,
    head_offsets: np.ndarray,
    eos_token_id: int,
    ngram_size: int,
    heads_per_ngram: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ngram embedding ids on the host.

    Numerically identical to ``Qwen4ExpNGramEmbedding._compute_ngram_ids``; see
    the module docstring for which parts of that identity are load-bearing.

    Args:
        input_ids: this step's token ids in token order, length ``T``. Values
            may be negative (the drafter's ``PADDING_SENTINEL``); range checking
            and the fold onto row 0 belong to the embedding module.
        query_start_loc: ``[num_reqs + 1]`` cumulative token offsets, exactly
            the tensor the device path is handed.
        ngram_context: ``[num_reqs, ngram_size - 1]`` -- the tokens immediately
            preceding each request's first token this step, left-padded with
            ``eos_token_id``.
        multipliers / vocab_sizes / head_offsets: the layer's hash buffers.
            Host (numpy or CPU-torch) mirrors of ``layer_multipliers``,
            ``ngram_heads_vocab_sizes`` and ``ngram_heads_offsets``.
        eos_token_id, ngram_size, heads_per_ngram: taken from the layer.
        out: optional pre-allocated ``[T, (ngram_size - 1) * heads_per_ngram]``
            int64 destination, which in production is
            ``ple_embedding._pinned_ngram_ids[:T].numpy()`` -- a view of pinned
            memory, so the ids land in the buffer the gather already reads and
            no copy happens at all.

    Raises:
        ValueError: any layout this function has not been shown to reproduce.
            The caller must fall back to the device path, not proceed.
    """
    input_ids = _as_int64(input_ids, "input_ids")
    query_start_loc = _as_int64(query_start_loc, "query_start_loc")
    ngram_context = _as_int64(ngram_context, "ngram_context")
    multipliers = _as_int64(multipliers, "multipliers").reshape(-1)
    vocab_sizes = _as_int64(vocab_sizes, "vocab_sizes").reshape(-1)
    head_offsets = _as_int64(head_offsets, "head_offsets").reshape(-1)

    ngram_size = int(ngram_size)
    heads_per_ngram = int(heads_per_ngram)
    eos_token_id = int(eos_token_id)
    num_heads = (ngram_size - 1) * heads_per_ngram

    # --- preconditions: refuse anything the device path has not been shown to
    # agree with. Every one of these is a shape the engine can legitimately
    # present (drafting, padding, a config we do not run), and the correct
    # response to all of them is the device path.
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be >= 2, got {ngram_size}")
    if heads_per_ngram <= 0:
        raise ValueError(f"heads_per_ngram must be > 0, got {heads_per_ngram}")
    if multipliers.size < ngram_size:
        raise ValueError(f"need {ngram_size} multipliers, got {multipliers.size}")
    if vocab_sizes.size < num_heads or head_offsets.size < num_heads:
        raise ValueError(
            f"need {num_heads} vocab_sizes/head_offsets, got "
            f"{vocab_sizes.size}/{head_offsets.size}"
        )
    if ngram_context.ndim != 2:
        raise ValueError(f"ngram_context must be 2D, got {ngram_context.ndim}D")
    if ngram_context.shape[1] != ngram_size - 1:
        # See "the column identity" in the module docstring: the device code
        # indexes at columns + ngram_size - 1, which only names the token
        # itself when the context is exactly ngram_size - 1 wide.
        raise ValueError(
            f"ngram_context width {ngram_context.shape[1]} != ngram_size - 1 "
            f"= {ngram_size - 1}; the device path would read a different window"
        )
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got {input_ids.shape}")

    num_reqs = int(query_start_loc.size) - 1
    if num_reqs < 0:
        raise ValueError("query_start_loc must hold at least one boundary")
    if ngram_context.shape[0] < num_reqs:
        raise ValueError(
            f"ngram_context has {ngram_context.shape[0]} rows for {num_reqs} requests"
        )
    num_tokens = int(input_ids.size)
    if num_reqs > 0 and int(query_start_loc[0]) != 0:
        raise ValueError(
            f"query_start_loc must start at 0, got {int(query_start_loc[0])}"
        )
    if num_reqs > 0 and int(query_start_loc[-1]) != num_tokens:
        raise ValueError(
            f"query_start_loc ends at {int(query_start_loc[-1])} but there are "
            f"{num_tokens} tokens"
        )
    if np.any(np.diff(query_start_loc) < 0):
        raise ValueError("query_start_loc must be non-decreasing")

    if out is None:
        out = np.empty((num_tokens, num_heads), dtype=np.int64)
    else:
        if out.dtype != np.int64:
            raise ValueError(f"out must be int64, got {out.dtype}")
        if out.shape != (num_tokens, num_heads):
            raise ValueError(
                f"out has shape {out.shape}, need {(num_tokens, num_heads)}"
            )
    if num_tokens == 0 or num_reqs == 0:
        return out

    ctx_len = ngram_size - 1
    starts = query_start_loc[:-1]
    ends = query_start_loc[1:]

    for req in range(num_reqs):
        start, end = int(starts[req]), int(ends[req])
        qlen = end - start
        if qlen == 0:
            # The device path produces a row for every token it is given, and a
            # zero-length request contributes none, so there is nothing to do.
            continue
        # The device path's `context` row: [ngram_context | packed], where the
        # request's tokens occupy packed columns 0..qlen-1 and every later cell
        # is eos. Cells after this request's last token cannot affect any of its
        # ids (both the segment scan and the shifted gathers look backwards
        # only), so the row is cut at the last token this request owns.
        row = np.concatenate((ngram_context[req], input_ids[start:end]))
        # Row index of each of this request's tokens. `columns + ngram_size - 1`
        # in the device code; identical to the token's index because the width
        # precondition above holds.
        idx = np.arange(ctx_len, ctx_len + qlen, dtype=np.int64)

        # position_in_segment, for those indices only.
        #   previous_eos[i] = max{j < i : row[j] == eos}, or -1
        # which is torch.cummax over (j if eos else -1), shifted right by one
        # with a -1 pad -- i.e. `_shift_precompute`.
        eos_pos = np.where(row == eos_token_id, np.arange(row.size, dtype=np.int64), -1)
        previous_eos = np.concatenate((np.array([-1], dtype=np.int64), eos_pos[:-1]))
        previous_eos = np.maximum.accumulate(previous_eos)[idx]
        position_in_segment = idx - previous_eos - 1

        shifted = [row[idx]]
        for shift in range(1, ngram_size):
            source = idx - shift
            # `_shift_apply`: gather from clamp_min(source, 0) but keep the value
            # only when the source exists and the segment is at least `shift`
            # tokens deep; otherwise eos. The clamp is load-bearing for the
            # gather (a negative index would wrap the row) and harmless for the
            # result, because those lanes are replaced.
            values = np.where(
                (source >= 0) & (position_in_segment >= shift),
                row[np.clip(source, 0, None)],
                eos_token_id,
            )
            shifted.append(values)

        # One int64 multiply per (token, shift) is allowed to overflow. See the
        # module docstring: torch wraps, numpy wraps, only the warning differs.
        with np.errstate(over="ignore"):
            for ngram in range(2, ngram_size + 1):
                head_start = (ngram - 2) * heads_per_ngram
                head_end = head_start + heads_per_ngram
                mixed = shifted[0] * multipliers[0]
                for index in range(1, ngram):
                    mixed = np.bitwise_xor(
                        mixed,
                        shifted[index] * multipliers[index],
                    )
                out[start:end, head_start:head_end] = (
                    np.remainder(
                        mixed[:, None],
                        vocab_sizes[None, head_start:head_end],
                    )
                    + head_offsets[None, head_start:head_end]
                )
    return out


class HostIdsRefused(ValueError):
    """The host cannot produce this step's ids, so the device path must run.

    Subclasses :class:`ValueError` because that is the contract
    ``compute_ngram_ids_host`` already documents: a refusal is not a failure to
    propagate, it is the signal to fall back. ``reason`` is a short stable token
    so a caller can *count* which precondition fires in production instead of
    guessing from a log line -- the difference between "the host path covers
    94 % of decode steps" and a knob that quietly never engages.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def build_decode_step_inputs(
    idx_mapping_np: np.ndarray,
    num_scheduled_tokens_np: np.ndarray,
    num_computed_tokens_np: np.ndarray,
    all_token_ids_np: np.ndarray,
    draft_values_np: np.ndarray | None,
    draft_row_idx: np.ndarray | None,
    *,
    ngram_context_len: int,
    eos_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild one pure-decode step's PLE inputs on the host.

    Returns ``(input_ids, query_start_loc, ngram_context)`` -- exactly the three
    arrays ``Qwen4ExpNGramEmbedding._compute_ngram_ids`` is handed on the device,
    so the result goes straight into :func:`compute_ngram_ids_host`.

    Why each piece lives where it lives (the derivation, not a guess):

    * **The first row's token is at sequence position ``num_computed_tokens``.**
      ``_post_update_kernel`` advances ``total_len`` by ``num_sampled`` and
      ``num_computed_tokens`` by ``query_len - num_rejected``, and those two
      deltas are the same number (the accepted rows). So a request carries
      exactly one more stored token than it has computed tokens, and the stored
      token the next step feeds first is the one at index
      ``num_computed_tokens`` of ``all_token_ids`` -- which the GPU wrote into
      the UVA mirror at the end of the previous step.
    * **The context window is the ``ngram_context_len`` tokens strictly before
      that index**, EOS-filled on the left. That is ``_prepare_ngram_context``
      reproduced: ``context_end = num_computed_tokens``, offsets
      ``arange(-context_len, 0)``, ``where(idx >= 0, token, eos)``. Reading it
      from ``all_token_ids_np`` rather than ``.gpu`` is the whole point -- same
      pages, no stream-ordered read.
    * **Rows 1..q-1 are this step's drafts** and are deliberately *not* in
      ``all_token_ids`` (they are speculative, only the accepted prefix is
      committed). They come from the pinned mirror staged at the previous step's
      epilogue, addressed by the row index that mirror published -- never by
      batch position.

    The one thing this function cannot check is that ``num_computed_tokens_np``
    equals the device's ``num_computed_tokens``: the two are updated by different
    code (the scheduler here, ``_post_update_kernel`` there) and
    ``states.py`` calls the host one an *optimistic upper bound*. A disagreement
    would shift every id this function produces and nothing here would notice,
    so the consumer must run in differential mode (compare against the device's
    own ids) before this path is trusted for serving.

    Raises:
        HostIdsRefused: any batch shape this function has not been shown to
            reproduce. The caller must take the device path.
    """
    if draft_values_np is None or draft_row_idx is None:
        raise HostIdsRefused("no-drafts", "staging is off or nothing was staged")
    ngram_context_len = int(ngram_context_len)
    if ngram_context_len < 1:
        raise HostIdsRefused("context-width", str(ngram_context_len))

    num_reqs = int(idx_mapping_np.size)
    if num_reqs == 0:
        raise HostIdsRefused("empty-batch")
    if int(num_scheduled_tokens_np.size) < num_reqs:
        raise HostIdsRefused(
            "layout",
            f"num_scheduled_tokens has {num_scheduled_tokens_np.size} "
            f"entries for {num_reqs} requests",
        )
    if all_token_ids_np.ndim != 2:
        raise HostIdsRefused("layout", "all_token_ids_np must be 2D")

    slots = np.asarray(idx_mapping_np, dtype=np.int64)
    if (slots < 0).any():
        raise HostIdsRefused("negative-slot", slots.tolist())
    queries = np.asarray(num_scheduled_tokens_np, dtype=np.int64)[:num_reqs]
    num_draft = int(draft_values_np.shape[1])
    if not bool((queries == num_draft + 1).all()):
        # A request with a different row count is a prefill, a chunked prefill, a
        # short draft, or a padded slot. Any of them means this step is not the
        # shape the host path was written for, and the batch is not separable:
        # the ids array is one flat buffer, so a partial host answer would be a
        # partial D2H -- which is the whole reason narrowed D1 was killed.
        raise HostIdsRefused(
            "not-pure-decode",
            f"queries {queries.tolist()} != draft+1 = {num_draft + 1}",
        )

    # Staged rows are addressed by the req_state slot they belonged to. The
    # previous step's batch may be larger, smaller, or shifted relative to this
    # one, and a slot appears at most once per batch, so a scatter into a
    # max_num_reqs-sized table is exact and costs one fill per step.
    row_of_slot = np.full(int(all_token_ids_np.shape[0]), -1, dtype=np.int64)
    staged_slots = np.asarray(draft_row_idx, dtype=np.int64)
    if staged_slots.size and int(staged_slots.max()) >= row_of_slot.size:
        raise HostIdsRefused("layout", "staged row index out of range")
    row_of_slot[staged_slots] = np.arange(staged_slots.size, dtype=np.int64)

    positions = np.asarray(num_computed_tokens_np, dtype=np.int64)[slots]
    if (positions < 0).any() or (positions >= all_token_ids_np.shape[1]).any():
        # Also catches a wildly optimistic host mirror of the device counter.
        raise HostIdsRefused("position-out-of-range", f"positions {positions.tolist()}")

    total_tokens = int(queries.sum())
    input_ids = np.empty(total_tokens, dtype=np.int64)
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(queries, out=query_start_loc[1:])
    ngram_context = np.full(
        (num_reqs, ngram_context_len), int(eos_token_id), dtype=np.int64
    )

    for i in range(num_reqs):
        slot = int(slots[i])
        pos = int(positions[i])
        row = int(row_of_slot[slot])
        if row < 0:
            # This request was not in the step that staged the drafts, so its
            # rows 1..q-1 are unknown here. Refusing the whole batch is the only
            # safe answer: the device would use this step's real drafts.
            raise HostIdsRefused("row-missing", f"slot {slot} had no staged draft")

        start_of_context = pos - ngram_context_len
        if start_of_context >= 0:
            ngram_context[i] = all_token_ids_np[slot, start_of_context:pos]
        elif pos > 0:
            # Left-EOS-fill, exactly the device's where(idx >= 0, token, eos).
            ngram_context[i, ngram_context_len - pos :] = all_token_ids_np[slot, 0:pos]

        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        input_ids[start] = all_token_ids_np[slot, pos]
        input_ids[start + 1 : end] = draft_values_np[row, : end - start - 1]

    return input_ids, query_start_loc, ngram_context


__all__ = [
    "PADDING_SENTINEL",
    "HostIdsRefused",
    "build_decode_step_inputs",
    "compute_ngram_ids_host",
    "hostids_enabled",
]
