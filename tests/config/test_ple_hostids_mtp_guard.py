# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the VLLM_GFX906_PLE_HOSTIDS / speculative-decoding guard.

The guard exists because of a measurement on this box, not a theory. On
2026-10-01 22:20:51 (boot ``vllm.log.20261001-215123``, MTP with
``num_speculative_tokens=3``, differential mode) all four TP workers reported the
same mismatch, and inverting the n-gram hash exactly
(``scripts/invert-mismatch-2220b.py`` -- four heads are residues of one 64-bit
value under four ~2e7 coprimes, so CRT recovers it) showed:

* the host staged a step whose query was ``[760, 1156, 2640, 343]``;
* rows 2-3 of the device's ids buffer hash to a query whose row-1 token is ``4``,
  not ``1156`` -- another MTP pass's draft;
* row 1 of that *same* buffer agrees with the host, which requires row-1 token
  ``1156``.

No single compute produces that array, so the buffer is a patchwork written by the
main pass and the three drafter passes. The staged path chooses itself on row-count
agreement, so a pass with the same row count and different tokens consumes
embeddings gathered for a different query: a plausible answer built from the wrong
tokens, which is the class of failure this engine has been burned by before
(``docs/gfx906/mtp-cudagraph-profile-crash.md`` 13.2).

The behaviour that matters is that the combination is refused at config time
rather than running for 23 minutes and then serving subtly wrong PLE rows:

* refused with MTP and the knob on at any level, including differential;
* untouched with no speculative decoding -- that is the boot where a differential
  answer is valid at all -- and untouched with the knob off, because a guard that
  fires on ordinary boots is a guard that gets removed;
* overridable without editing code;
* and actually called from ``VllmConfig.__post_init__``, because a correct
  predicate nobody calls protects nothing.
"""

import ast
import inspect

import pytest

vllm_config_module = pytest.importorskip(
    "vllm.config.vllm",
    reason="vllm.config.vllm is not importable in this build",
)


@pytest.fixture(autouse=True)
def _knob_off_by_default(monkeypatch):
    monkeypatch.delenv(vllm_config_module.ENV_PLE_HOSTIDS, raising=False)
    monkeypatch.delenv(vllm_config_module.ENV_ALLOW_HOSTIDS_MTP, raising=False)


def _conflict(spec_tokens, *, knob="1", environ=None):
    env = {} if environ is None else dict(environ)
    if knob is not None:
        env[vllm_config_module.ENV_PLE_HOSTIDS] = knob
    return vllm_config_module.ple_hostids_mtp_conflict(spec_tokens, environ=env)


def test_mtp_with_the_knob_on_is_refused():
    reason = _conflict(3)
    assert reason is not None
    assert vllm_config_module.ENV_PLE_HOSTIDS in reason


def test_differential_mode_is_refused_too():
    # Mode 2 reads the same per-pass buffer, so it cannot be evaluated either.
    assert _conflict(3, knob="2") is not None


def test_one_speculative_token_is_still_two_passes():
    assert _conflict(1) is not None


def test_the_reason_says_why_and_names_the_way_out():
    reason = _conflict(3)
    assert reason is not None
    # An operator reading only this line must learn the mechanism (passes per
    # step) and the fix (a drafter-off boot, or the override).
    assert "4 times per step" in reason
    assert "--speculative-config" in reason
    assert vllm_config_module.ENV_ALLOW_HOSTIDS_MTP in reason


def test_no_speculative_decoding_is_untouched():
    # This is the configuration where a differential answer means something.
    assert _conflict(0) is None


def test_the_guard_is_silent_when_the_knob_is_off():
    # The overwhelmingly common case: an MTP boot with no experimental knob.
    assert _conflict(3, knob=None) is None
    assert _conflict(3, knob="0") is None
    assert _conflict(3, knob="") is None


def test_escape_hatch_disables_the_guard():
    assert _conflict(3, environ={vllm_config_module.ENV_ALLOW_HOSTIDS_MTP: "1"}) is None


def test_escape_hatch_must_be_exactly_one():
    assert (
        _conflict(3, environ={vllm_config_module.ENV_ALLOW_HOSTIDS_MTP: "0"})
        is not None
    )
    # Anything but "1" keeps the protection, including a typo of it.
    assert (
        _conflict(3, environ={vllm_config_module.ENV_ALLOW_HOSTIDS_MTP: "true"})
        is not None
    )


def test_helper_is_wired_into_config_validation():
    """The predicate has to be *called* from config validation.

    A guard whose predicate is correct and whose call site was dropped in a rebase
    is invisible until the wrong-answer boot it was written for happens again.
    """
    tree = ast.parse(inspect.getsource(vllm_config_module))
    calls_by_function: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "ple_hostids_mtp_conflict"
            ):
                calls_by_function[node.name] = calls_by_function.get(node.name, 0) + 1
    assert calls_by_function.get("__post_init__") == 1, calls_by_function
