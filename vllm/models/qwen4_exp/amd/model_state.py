# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd import ple_hostids
from vllm.models.qwen4_exp.amd.ple_hostids import (
    HostIdsRefused,
    HostTokenHistory,
    build_decode_step_inputs,
)
from vllm.models.qwen4_exp.amd.ple_prefetch import (
    PleRowPrefetcher,
    plan_lookahead_chunks,
    prefetch_enabled,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        # Host-side id staging (VLLM_GFX906_PLE_HOSTIDS). The draft mirror is set
        # once by the model runner, never per step; the layer list and refusal
        # counters exist so "the host path quietly never engaged" is a log line
        # rather than a guess.
        self._host_drafts: object | None = None
        self._ple_host_layers: list | None = None
        # Layers watched by the generation tripwire. Kept separate from
        # ``_ple_host_layers`` because disabling the host path must not disarm the
        # instrument: once a graph has captured the staged branch, a disabled host
        # path is the frozen-embedding state, and the tripwire is the only thing
        # that can see it.
        self._ple_host_gen_layers: list = []
        self._ple_host_refusals: dict[str, int] = {}
        self._ple_host_refusal_logged: set[str] = set()
        self._ple_host_steps = 0
        # The draft-block generation this model state last consumed. Freshness is
        # the one property of the mirror that no shape check can infer.
        self._last_draft_gen = -1
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        # Opt-in read-ahead for the PLE ngram table (see ple_prefetch.py).
        # Built on first use, not here: the model is not loaded yet at this
        # point, and the prefetcher needs the embedding's shard pointers.
        self._ple_prefetcher: PleRowPrefetcher | None = None
        self._ple_prefetch_wanted = prefetch_enabled()
        self._ple_host_tokens = None
        # P1.6: the host mirror of committed tokens, plus the request -> slot map
        # that lets ``remove_request`` clean the mirror's queue. Built here rather
        # than lazily because seeding happens at add time, before any step has
        # consulted the knob -- and an unseeded window is a step that cannot
        # stage. The size comes from the ngram config, so the mirror is correct
        # whether or not ids are ever staged on the host.
        self._ple_host_history = HostTokenHistory(
            self.max_num_reqs, self.ngram_context_len + 1
        )
        self._ple_host_slots: dict[str, int] = {}

    # ------------------------------------------------------------- lifecycle
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Seed with exactly the list ``req_states.add_request`` staged into its
        # UVA array -- the runner passes the same ``prefill_token_ids`` to both.
        # That list is host-written, so it is the one part of a request's history
        # the host can trust without asking the device; everything after it
        # arrives through ``HostTokenHistory.stage_step``.
        self._ple_host_slots[new_req_data.req_id] = req_index
        self._ple_host_history.seed(req_index, new_req_data.prefill_token_ids or ())

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        # The runner calls this *before* ``req_states.remove_request``, so the
        # slot is still resolvable here. Cleaning the mirror is not tidiness: an
        # in-flight commit belonging to a finished request would otherwise land on
        # whoever reuses the slot next and corrupt its token count.
        req_index = self._ple_host_slots.pop(req_id, None)
        if req_index is not None:
            self._ple_host_history.invalidate(req_index)

    # ------------------------------------------------------------- read-ahead
    def _ple_prefetch_modules(self):
        """Locate the ngram embedding module without hardcoding layer indices.

        PLE lives at ``ple_layer_ids`` (layer 2 for this checkpoint), and the
        attribute path is a model-implementation detail. Searching by type keeps
        this correct if the model is rearranged, and returns None -- disabling
        the prefetch -- instead of raising if it moves.
        """
        from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpNGramEmbedding

        for module in self.model.modules():
            if isinstance(module, Qwen4ExpNGramEmbedding):
                return module
        return None

    def _build_ple_prefetcher(self) -> PleRowPrefetcher | None:
        ngram = self._ple_prefetch_modules()
        if ngram is None:
            logger.info_once(
                "PLE read-ahead requested but no ngram embedding module was "
                "found; running without it."
            )
            self._ple_prefetch_wanted = False
            return None
        # A lookahead larger than one scheduled chunk is pure waste: the next
        # chunk cannot be bigger than the batch token budget.
        max_tokens = int(
            getattr(
                self.vllm_config.scheduler_config,
                "max_num_batched_tokens",
                8192,
            )
        )
        return PleRowPrefetcher(
            ngram.ngram_embedding,
            (
                ngram.layer_multipliers,
                ngram.ngram_heads_vocab_sizes,
                ngram.ngram_heads_offsets,
            ),
            type(ngram)._compute_ngram_ids,
            ngram_size=int(ngram.ngram_size),
            heads_per_ngram=int(ngram.heads_per_ngram),
            eos_token_id=int(ngram.eos_token_id),
            max_tokens=max_tokens,
        )

    def _prefetch_next_ple_chunk(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        """Ask the kernel to prefetch the next prefill chunk's embedding rows.

        Every input is already on the host as numpy, so this costs no device
        synchronisation: ``num_computed_tokens_np`` is the optimistic CPU mirror
        and ``prefill_len_np`` the scheduled prompt length. A stale mirror only
        warms pages the chunk does not need, which is wasted bandwidth, never a
        wrong number -- that asymmetry is why an upper-bound mirror is safe here.
        """
        pf = self._ple_prefetcher
        if pf is not None and pf.disabled_reason is not None:
            # Permanently off: stop planning too, this runs on every step and
            # the whole point is to spend no host CPU on the critical path.
            self._ple_prefetcher = None
            self._ple_prefetch_wanted = False
            return
        if pf is None and not self._ple_prefetch_wanted:
            return
        # CUDA graph capture replays this hook with dummy batches, once per
        # capture size. Hinting those pages would stream garbage off NVMe
        # during boot, which is when the box can least afford it. Checked every
        # call: capture happens long after the first real step.
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        if pf is None:
            try:
                self._ple_prefetcher = self._build_ple_prefetcher()
            except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
                logger.warning(
                    "PLE read-ahead could not be set up (%s: %s); running "
                    "without it. Outputs are unaffected.",
                    type(exc).__name__,
                    exc,
                )
                self._ple_prefetch_wanted = False
                return
        pf = self._ple_prefetcher
        if pf is None:
            # The builder declined (no ngram module): nothing to plan against.
            return
        try:
            if self._ple_host_tokens is None:
                buf = getattr(
                    getattr(req_states, "all_token_ids", None), "_uva_buf", None
                )
                host = getattr(buf, "np", None)
                if host is None or host.ndim != 2:
                    # all_token_ids is normally a UVA-backed host array; on a
                    # platform where it is real GPU memory there is no cheap
                    # host view, and a D2H copy per step would defeat the point.
                    pf.disable("req_states.all_token_ids has no host view")
                    return
                self._ple_host_tokens = host
            n = input_batch.num_reqs
            for req, start, end, ctx in plan_lookahead_chunks(
                host_tokens=self._ple_host_tokens,
                idx_mapping=input_batch.idx_mapping_np[:n],
                scheduled=input_batch.num_scheduled_tokens[:n],
                computed=input_batch.num_computed_tokens_np[:n],
                prefill_len=input_batch.prefill_len_np[:n],
                context_len=self.ngram_context_len,
                eos_token_id=self.ngram_eos_token_id,
            ):
                pf.prefetch_token_chunk(self._ple_host_tokens, req, start, end, ctx)
        except Exception as exc:  # noqa: BLE001 - never reach the model path
            pf.disable(f"lookahead planning failed ({type(exc).__name__}: {exc})")

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        context = self.ngram_context[:num_reqs_padded]
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def _stage_host_ple_ids(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        """Compute this step's PLE ids on the host, if the step is one we can.

        Runs every step, including steps that replay a captured CUDA graph —
        which is the whole reason the host has to do this work itself: a captured
        FULL decode graph replays the H2D out of ``_pinned_output`` and contains
        neither the D2H of the ids nor the gather that produced them.

        Every failure mode returns without touching the model inputs, so the
        device path stays the default answer to anything unexpected. Refusals are
        counted per reason and the first of each is logged: coverage is a claim
        that needs evidence, and "the knob was set" is not that evidence.
        """
        mode = ple_hostids.hostids_mode()
        if mode == 0 or not self.uses_ngram_embedding:
            return
        if self._ple_host_layers is None:
            from vllm.models.qwen4_exp.amd.ple_layer import host_stageable_layers

            layers = [
                layer
                for layer in host_stageable_layers()
                if layer.ngram_size - 1 == self.ngram_context_len
            ]
            if not layers:
                # The env var is set but no PLE module registered (import order,
                # or the knob was set for a model without PLE). Say so once.
                self._note_host_refusal("no-registered-layers")
                return
            self._ple_host_layers = layers
            self._ple_host_gen_layers = list(layers)
        # Tripwire first, and above the refusal paths it exists to observe: a step
        # the host could not stage is exactly the step whose replay is worth
        # looking at. It raises when a graph captured the staged branch and the
        # handoff went stale, which is the one case where "fall back to the device
        # path" is not something a replay can do.
        for layer in self._ple_host_gen_layers:
            layer.check_host_gen()
        if self._host_drafts is None:
            self._note_host_refusal("no-draft-source")
            return
        # Apply every queued commit whose device->host copy has landed. This is
        # the only place the queue is drained, and it never blocks: a commit still
        # in flight makes this step refuse below rather than wait, because waiting
        # here is the per-step host sync the whole design exists to remove.
        self._ple_host_history.drain()
        if self._ple_host_history.pending:
            self._note_host_refusal("history-pending")
            return

        num_reqs = input_batch.num_reqs
        try:
            values, row_idx, gen = self._host_drafts.wait_host_draft_tokens()  # type: ignore[attr-defined]  # noqa: E501
            # Freshness, not presence. Two consecutive decode steps over an
            # unchanged batch produce an IDENTICAL slot->row map with different
            # values, so a map that lines up proves nothing about which step the
            # numbers came from. A generation that did not advance since the last
            # consumption means the epilogue staged nothing in between, and these
            # are the previous step's drafts -- the signature the 14:10 boot
            # showed: the rows whose tokens come from history and draft column 0
            # agreed, the row that needs column 1 did not.
            if values is not None and gen == self._last_draft_gen:
                raise HostIdsRefused("draft-gen-stale", f"gen={gen}")
            input_ids, query_start_loc, ngram_context = build_decode_step_inputs(
                input_batch.idx_mapping_np[:num_reqs],
                input_batch.num_scheduled_tokens[:num_reqs],
                req_states.num_computed_tokens_np,
                self._ple_host_history.window,
                self._ple_host_history.length,
                values,
                row_idx,
                ngram_context_len=self.ngram_context_len,
                eos_token_id=self.ngram_eos_token_id,
            )
        except HostIdsRefused as exc:
            self._note_host_refusal(exc.reason)
            return
        except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
            self.disable_host_staging(
                f"host id build failed ({type(exc).__name__}: {exc})"
            )
            return

        gather = mode == 1
        device_inputs: tuple[Any, ...] | None = None
        if not gather:
            # Differential mode: keep the provenance so a mismatch can name the
            # block its draft columns came from instead of describing its output.
            for layer in self._ple_host_layers:
                layer.note_draft_provenance(values, row_idx, gen)
            # The device half of this step's inputs, captured in the worker where
            # both sides are in scope. forward() cannot do it: torch.compile runs
            # that body at trace time, so anything stashed there is a trace-time
            # value. The runner's own arrays are padded, so the dump slices them
            # to the host's lengths.
            #
            # The window is handed over as a reference to the state's own buffer,
            # not as a copy, and that is the whole reason it can be compared at all:
            # prepare_inputs() fills ``ngram_context`` via ``_prepare_ngram_context``
            # and *then* calls this staging function, so by the time the op reads the
            # buffer back it holds this step's window -- the same cells the forward
            # was handed a view of. An earlier version of this file claimed the
            # worker had no device-side counterpart for the window and passed None;
            # boot #8's inversion put the one wrong term exactly in the cell that
            # window feeds, so the claim went looking for its evidence and failed.
            d_ids = getattr(input_batch, "input_ids", None)
            d_qsl = getattr(input_batch, "query_start_loc", None)
            d_ctx = getattr(self, "ngram_context", None)
            if d_ids is not None and d_qsl is not None:
                device_inputs = (d_ids, d_qsl, d_ctx)
        for layer in self._ple_host_layers:
            try:
                layer.stage_host_step(
                    input_ids,
                    query_start_loc,
                    ngram_context,
                    gather=gather,
                    device_inputs=device_inputs,
                )
            except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
                self.disable_host_staging(
                    f"host id staging failed ({type(exc).__name__}: {exc})"
                )
                return
        self._last_draft_gen = gen
        self._ple_host_steps += 1
        if self._ple_host_steps == 1:
            logger.info(
                "PLE HOSTIDS: host step inputs built for %d request(s), %d row(s) "
                "(mode %s: %s)",
                num_reqs,
                int(input_ids.size),
                mode,
                "host ids authoritative, device D2H skipped"
                if gather
                else "differential: device path in charge, ids compared every step",
            )

    def set_host_draft_source(self, handler: object) -> None:
        """Wire in the worker's staged draft-token mirror (P1.1).

        Called once at init by the model runner, and only when staging is enabled,
        so the per-step path costs one attribute read. Without it the host has
        rows 1..q-1 of no step and refuses every batch.
        """
        self._host_drafts = handler

    def stage_host_tokens(
        self,
        slots: object,
        sampled: object,
        num_sampled: object,
        done: object,
    ) -> bool:
        """Queue one step's accepted tokens for the host PLE mirror (P1.6).

        Called by the model runner's epilogue with the arrays the async output
        copy produced, plus its copy event: nothing here reads ``sampled``, so
        the data may still be in flight. The consumer applies it in
        ``_stage_host_ple_ids`` once ``done.query()`` says so, and refuses the
        step until then -- which is what keeps this knob free of the per-step
        host sync it exists to remove.

        Returns whether the commit was queued. Failures are counted, never
        raised: a knob that can kill a serving step is worse than one that
        reports no coverage.
        """
        if ple_hostids.hostids_mode() == 0:
            # The epilogue runs on every step of every boot; with the knob off
            # this must cost nothing and, above all, must not log.
            return False
        try:
            staged = self._ple_host_history.stage_step(
                slots,
                sampled,
                num_sampled,
                done,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
            self._note_host_refusal(f"stage-tokens-{type(exc).__name__}")
            return False
        if not staged:
            self._note_host_refusal("stage-tokens-rejected")
        return staged

    def disable_host_staging(self, why: str) -> None:
        if not self._ple_host_layers:
            return
        captured = any(
            getattr(layer, "_staged_branch_captured", False)
            for layer in self._ple_host_gen_layers
        )
        if captured:
            # Do not claim a fallback that a replay cannot perform. The op body
            # stopped running for these steps, so the graph keeps replaying
            # whatever is in the pinned buffer and the generation tripwire is what
            # says so out loud.
            logger.warning(
                "PLE HOSTIDS: host staging disabled after %d step(s) -- %s. A graph "
                "has captured the staged branch, so the device path does NOT handle "
                "these steps: the pinned rows will keep replaying. The generation "
                "tripwire stays armed. Refusals so far: %s",
                self._ple_host_steps,
                why,
                dict(self._ple_host_refusals),
            )
            self._ple_host_layers = []
            return
        logger.warning(
            "PLE HOSTIDS: host staging disabled after %d step(s) -- %s. The "
            "device path handles every step from here on. Refusals so far: %s",
            self._ple_host_steps,
            why,
            dict(self._ple_host_refusals),
        )
        self._ple_host_layers = []

    def _note_host_refusal(self, reason: str) -> None:
        self._ple_host_refusals[reason] = self._ple_host_refusals.get(reason, 0) + 1
        if reason not in self._ple_host_refusal_logged:
            self._ple_host_refusal_logged.add(reason)
            logger.info(
                "PLE HOSTIDS: first refusal (%s); device path used for that step. "
                "Counts so far: %s",
                reason,
                dict(self._ple_host_refusals),
            )

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        # Host ids first: the pinned rows they fill are what this step's H2D (or,
        # under a captured graph, the replayed H2D) delivers to the device, so
        # they have to be in place before the forward is enqueued.
        self._stage_host_ple_ids(input_batch, req_states)
        # Issued last: the current chunk's inputs are already prepared, so the
        # hints overlap the GPU work that follows instead of delaying it.
        self._prefetch_next_ple_chunk(input_batch, req_states)
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc[: num_reqs + 1]
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])

        ngram_context = self.ngram_context[:num_reqs]
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
