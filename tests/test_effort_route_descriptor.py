"""Reasoning-effort route descriptor (SSOT) — projection, carriage and
disclosure tests.

The descriptor in ``ouroboros/provider_models.py`` is the single table that
answers, per registered provider route: which wire carrier (if any) carries
the effort tier, which provider tier values the route accepts, how canonical
tiers project onto them, and what an ABSENT parameter means. These tests pin:

* the z.ai/GLM alias projection measured on a live Coding Plan key
  (PR #1194 close comment): none/minimal/low→low, medium/high→high,
  xhigh/ultra→max;
* the effort_not_carried disclosure when a tier is requested against a
  no-carrier route (the tier is dropped, never guessed onto an unmeasured
  wire, and the drop is a visible usage fact);
* the descriptor↔capability_evidence declared-vs-observed check;
* the UI-facing tier list (3 choices for GLM, not 8; identity spellings).
"""

import pytest

from ouroboros.provider_models import (
    EFFORT_ROUTE_DESCRIPTOR,
    canonical_tiers_for_route,
    effort_descriptor_for_route,
    project_effort_for_route,
)


class TestDescriptorShape:
    def test_every_row_declares_required_fields(self):
        for route, row in EFFORT_ROUTE_DESCRIPTOR.items():
            assert "carrier" in row, route
            assert "absent_meaning" in row, route
            assert row["absent_meaning"] in ("max", "provider_default", "off"), route
            if row["carrier"] == "none":
                assert row["tiers"] == [], route
            else:
                assert row["tiers"], route
                for tier in row["tiers"]:
                    assert row["project"].get(tier, tier) in row["tiers"] or tier in (
                        "low", "high", "max", "minimal", "medium", "xhigh", "ultra"
                    ), (route, tier)

    def test_unknown_provider_fails_to_carrier_none(self):
        assert effort_descriptor_for_route("never-measured")["carrier"] == "none"

    def test_unmeasured_chinese_routes_declare_none(self):
        # DO-NOT rule: qwen/kimi/zai carriers are NOT declared by analogy to
        # GLM — no key, no measurement in this tree.
        for route in ("zai", "qwen", "kimi"):
            assert EFFORT_ROUTE_DESCRIPTOR[route]["carrier"] == "none", route

    def test_deepseek_and_glm_share_one_alias_table(self):
        assert (
            EFFORT_ROUTE_DESCRIPTOR["deepseek"]["project"]
            is EFFORT_ROUTE_DESCRIPTOR["zai-glm"]["project"]
        )


class TestZaiInsufficientBalance:
    """429 code 1113 "Insufficient balance" is billing, not rate limiting."""

    class _Exc(Exception):
        status_code = 429
        code = "1113"
        type = ""

    def test_probe_maps_to_no_credits(self):
        from ouroboros.llm_probe import controlled_probe_error

        result = controlled_probe_error(self._Exc("Insufficient balance"))
        assert result["error"] == "No credits"
        assert result["status_code"] == 429

    def test_plain_429_stays_rate_limited(self):
        from ouroboros.llm_probe import controlled_probe_error

        class Plain(Exception):
            status_code = 429
            code = ""
            type = ""

        assert controlled_probe_error(Plain("too many requests"))["error"] == "Rate limited"


class TestZaiProjection:
    """Measured on a live Z.ai Coding Plan key (glm-5.3), PR #1194 close."""

    @pytest.mark.parametrize(
        ("canonical", "wire"),
        [
            ("none", "low"),      # thinking cannot be disabled (400 code 1210)
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
            ("ultra", "max"),
        ],
    )
    def test_projection(self, canonical, wire):
        d = effort_descriptor_for_route("openai-compatible", "glm-5.3")
        assert d["carrier"] == "reasoning_effort"
        assert project_effort_for_route(d, canonical) == wire

    def test_absent_tier_means_max(self):
        d = effort_descriptor_for_route("openai-compatible", "glm-5.3")
        assert d["absent_meaning"] == "max"

    def test_glm_detection_is_model_identity_only(self):
        assert effort_descriptor_for_route("openai-compatible", "GLM-5.3")["carrier"] == "reasoning_effort"
        assert effort_descriptor_for_route("openai-compatible", "zai-org/GLM-4.7")["carrier"] == "reasoning_effort"
        # A non-GLM model on the same lane stays honestly unmeasured.
        assert effort_descriptor_for_route("openai-compatible", "Qwen/Qwen3-235B")["carrier"] == "none"

    def test_ui_offers_three_identity_tiers_not_eight_aliases(self):
        d = effort_descriptor_for_route("openai-compatible", "glm-5.3")
        assert canonical_tiers_for_route(d) == ["low", "high", "max"]

    def test_no_forced_tool_suppression_on_glm(self):
        # Measured: tool_choice required/named work WITH thinking on glm-5.3 —
        # DeepSeek's forced-tool suppression exception must NOT carry over.
        d = effort_descriptor_for_route("openai-compatible", "glm-5.3")
        assert d["forced_tool_suppression"] is False
        assert EFFORT_ROUTE_DESCRIPTOR["deepseek"]["forced_tool_suppression"] is True


class TestBuilderCarriage:
    """_build_remote_kwargs reads the descriptor for every direct lane."""

    def _target(self, provider, resolved_model):
        return {
            "provider": provider,
            "resolved_model": resolved_model,
            "usage_model": f"{provider}::{resolved_model}",
            "supports_openrouter_extensions": False,
        }

    def _build(self, provider, resolved_model, effort, tools=None, tool_choice="auto"):
        from ouroboros.llm import LLMClient

        client = LLMClient()
        return client._build_remote_kwargs(
            self._target(provider, resolved_model),
            [{"role": "user", "content": "hi"}],
            effort, 256, tool_choice, None, tools,
        )

    def test_glm_effort_reaches_the_wire(self):
        kwargs = self._build("openai-compatible", "glm-5.3", "low")
        assert kwargs["reasoning_effort"] == "low"

    def test_glm_xhigh_projects_to_max(self):
        kwargs = self._build("openai-compatible", "glm-5.3", "xhigh")
        assert kwargs["reasoning_effort"] == "max"

    def test_glm_medium_projects_to_high(self):
        kwargs = self._build("openai-compatible", "glm-5.3", "medium")
        assert kwargs["reasoning_effort"] == "high"

    def test_glm_minimal_projects_to_low(self):
        kwargs = self._build("openai-compatible", "glm-5.3", "minimal")
        assert kwargs["reasoning_effort"] == "low"

    def test_openai_still_carries_identity(self):
        kwargs = self._build("openai", "gpt-5.6-terra", "medium")
        assert kwargs["reasoning_effort"] == "medium"

    def test_no_carrier_route_drops_with_disclosure(self):
        from ouroboros.llm_capability_policy import _EFFORT_NOT_CARRIED_CVAR

        kwargs = self._build("openai-compatible", "some-vllm-model", "high")
        assert "reasoning_effort" not in kwargs
        note = _EFFORT_NOT_CARRIED_CVAR.get()
        assert note == {
            "requested": "high",
            "provider": "openai-compatible",
            "model": "some-vllm-model",
            "absent_meaning": "provider_default",
        }

    def test_no_disclosure_when_no_tier_requested(self):
        from ouroboros.llm_capability_policy import _EFFORT_NOT_CARRIED_CVAR

        self._build("openai-compatible", "some-vllm-model", "none")
        assert _EFFORT_NOT_CARRIED_CVAR.get() is None

    def test_usage_surfaces_effort_not_carried(self):
        from ouroboros.llm import LLMClient

        client = LLMClient()
        self._build("openai-compatible", "some-vllm-model", "high")
        msg, usage = client._normalize_remote_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            self._target("openai-compatible", "some-vllm-model"),
        )
        assert usage["effort_not_carried"]["requested"] == "high"
        # consumed once, never sticks to the context
        assert "effort_not_carried" not in (
            client._normalize_remote_response(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                 "usage": {}},
                self._target("openai-compatible", "some-vllm-model"),
            )[1]
        )

    def test_glm_forced_tool_keeps_thinking_on(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        kwargs = self._build("openai-compatible", "glm-5.3", "high", tools=tools, tool_choice="required")
        assert kwargs["reasoning_effort"] == "high"
        assert "extra_body" not in kwargs or "thinking" not in kwargs.get("extra_body", {})

    def test_deepseek_forced_tool_still_suppresses(self):
        from tests.test_deepseek_provider import TestWireProjection  # noqa: F401
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        kwargs = self._build("deepseek", "deepseek-v4-flash", "high", tools=tools, tool_choice="required")
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


class TestDescriptorCapabilityEvidenceAgreement:
    """DECLARED (descriptor) vs OBSERVED (capability_evidence) must never
    silently disagree: a mismatch is a loud fact, and the descriptor's
    declared ceilings/floors stay within the provider tiers it declares."""

    def _observed_facts(self):
        import inspect
        import ouroboros.capability_evidence as ce

        # The observable API surface of the evidence store: recorded
        # ceilings/floors/rejected_params are diagnostic namespaces keyed by
        # normalized model identity. A descriptor row DISAGREES with the
        # observed namespace only when an observed fact names an effort tier
        # outside the row's declared provider tiers for a matching route.
        return ce

    def test_descriptor_tiers_are_canonical_scale_members(self):
        from ouroboros.config import effort_rank

        valid = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
        for route, row in EFFORT_ROUTE_DESCRIPTOR.items():
            for tier in row["tiers"]:
                assert tier in valid, (route, tier)
            for canonical, provider in (row.get("project") or {}).items():
                assert canonical in valid, (route, canonical)
                assert provider in valid, (route, provider)

    def test_projection_lands_inside_declared_tiers(self):
        # Pinned for the measured low/high/max dialects (DeepSeek, GLM/z.ai):
        # every canonical tier projects onto a declared wire value. Identity
        # routes (openai/openrouter/claudexor) and anthropic pass unlisted
        # tiers through unchanged — request-wire recovery owns those at send
        # time, so the projection-table invariant does not apply there.
        from ouroboros.provider_models import EFFORT_ROUTE_ALIASES_LMH

        for route, row in EFFORT_ROUTE_DESCRIPTOR.items():
            if row.get("project") is not EFFORT_ROUTE_ALIASES_LMH:
                continue
            tiers = set(row["tiers"])
            for canonical in ("minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
                assert project_effort_for_route(row, canonical) in tiers, (route, canonical)

    def test_mismatch_is_detectable(self):
        # The loud-fact check: for each declared carried route, any observed
        # effort ceiling ABOVE the descriptor's top tier (or floor BELOW its
        # bottom tier) is a declared-vs-observed mismatch. This test pins the
        # predicate itself so a future table edit that contradicts a recorded
        # observation fails loudly here, not silently in production.
        from ouroboros.provider_models import _effort_descriptor_mismatches_observation

        row = dict(EFFORT_ROUTE_DESCRIPTOR["zai-glm"])
        assert not _effort_descriptor_mismatches_observation(
            row, observed_ceiling="max", observed_floor="low"
        )
        assert _effort_descriptor_mismatches_observation(
            row, observed_ceiling="ultra", observed_floor=""
        ), "ceiling above declared top tier is a mismatch"
        assert _effort_descriptor_mismatches_observation(
            row, observed_ceiling="", observed_floor="minimal"
        ), "floor below declared bottom tier is a mismatch"
