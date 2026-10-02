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

* refused with MTP and mode 1 (the mode that serves host rows), with no override
  available for it;
* refused with MTP and mode 2 (differential) until the override is set, because
  mode 2 is the one configuration where a comparison is possible at all; see
  ``test_override_opens_the_differential_and_only_the_differential``;
* untouched with no speculative decoding, which is a no-op case for this guard --
  and *not* the boot to run the differential on: measured 2026-10-02 a drafter-off
  boot compares nothing (the host mirror's token source is the drafter, so every
  step refuses with ``no-draft-source``) and does not compile on this engine
  (``ConstraintViolationError: input_ids specialized to 4096``,
  ``logs/vllm.log.20261001-234242`` / ``logs/vllm.log.20261002-065905``). The
  guard's message says so, and ``test_message_does_not_advertise_the_dead_route``
  keeps it saying so;
* untouched with the knob off, because a guard that fires on ordinary boots is a
  guard that gets removed;
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


def _override():
    return {vllm_config_module.ENV_ALLOW_HOSTIDS_MTP: "1"}


def test_mtp_with_the_knob_on_is_refused():
    reason = _conflict(3)
    assert reason is not None
    assert vllm_config_module.ENV_PLE_HOSTIDS in reason


def test_differential_mode_is_refused_without_the_override():
    # Mode 2 reads the same per-pass buffer, so it is not automatically safe; it
    # earns its override by never serving what it compared.
    assert _conflict(3, knob="2") is not None


def test_one_speculative_token_is_still_two_passes():
    assert _conflict(1) is not None


def test_the_reason_says_why_and_names_the_way_out():
    reason = _conflict(3)
    assert reason is not None
    # An operator reading only this line must learn the mechanism (passes per
    # step), which mode is dangerous, and the way out that actually works.
    assert "4 times per step" in reason
    assert "--speculative-config" in reason
    assert vllm_config_module.ENV_ALLOW_HOSTIDS_MTP in reason


def test_mode_one_and_mode_two_get_different_advice():
    # The two modes fail differently: mode 1 serves the patchwork, mode 2 only
    # looks at it. One message for both would have to be wrong about one of them.
    serving, diff = _conflict(3), _conflict(3, knob="2")
    assert "no override" in serving
    assert "only compares" in diff
    assert serving != diff


def test_message_does_not_advertise_the_dead_route():
    # The first version of this guard told the operator to drop
    # --speculative-config. That route was measured closed on both counts, and a
    # remedy is part of the guard: asserting the correction is what keeps the
    # regression out of the next boot.
    for reason in (_conflict(3), _conflict(3, knob="2")):
        assert reason is not None
        assert "does not work" in reason
        assert "no-draft-source" in reason
        assert "Run the differential on a boot without" not in reason


def test_override_opens_the_differential_and_only_the_differential():
    # Red control for the narrowing: a guard that lets the override open mode 1
    # offers a permission slip onto the silent-wrong-answer path.
    assert _conflict(3, knob="2", environ=_override()) is None
    assert _conflict(3, knob="1", environ=_override()) is not None
    # Mode 0 needs no override, and an unknown mode is not a mode 2 in disguise.
    assert _conflict(3, knob=None) is None
    assert _conflict(3, knob="02", environ=_override()) is not None


def test_no_speculative_decoding_is_untouched():
    # One pass per step, so the patchwork argument does not apply. This is NOT the
    # configuration to run the differential in -- see the module docstring.
    assert _conflict(0) is None


def test_the_guard_is_silent_when_the_knob_is_off():
    # The overwhelmingly common case: an MTP boot with no experimental knob.
    assert _conflict(3, knob=None) is None
    assert _conflict(3, knob="0") is None
    assert _conflict(3, knob="") is None


def test_escape_hatch_disables_the_guard_for_the_differential():
    assert _conflict(3, knob="2", environ=_override()) is None


def test_escape_hatch_must_be_exactly_one():
    # Anything but "1" keeps the protection, including a typo of it.
    assert (
        _conflict(3, knob="2", environ={vllm_config_module.ENV_ALLOW_HOSTIDS_MTP: "0"})
        is not None
    )  # noqa: E501
    assert (
        _conflict(
            3, knob="2", environ={vllm_config_module.ENV_ALLOW_HOSTIDS_MTP: "true"}
        )
        is not None
    )  # noqa: E501


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
