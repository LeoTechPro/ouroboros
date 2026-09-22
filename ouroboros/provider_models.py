"""Provider-specific model ID helpers, direct-provider defaults, and the
provider registry (SSOT for prefix→provider→credentials knowledge that was
previously duplicated across llm.py, pricing.py, agent_task_pipeline.py and
deep_self_review.py)."""

from __future__ import annotations


from ouroboros.model_slots import ResolvedModelTarget, parse_fallback_chain
from ouroboros.settings_defaults import OPENROUTER_DEFAULTS, OPENROUTER_REVIEW_DEFAULTS, SETTINGS_DEFAULTS  # noqa: F401
from ouroboros.settings_integrity import runtime_setting

# MiniMax exposes the same OpenAI-compatible API on two regional hosts. Keep the
# mapping centralized so transport, capability evidence, and settings diagnostics
# fingerprint the exact endpoint selected by the owner.
MINIMAX_REGION_ENDPOINTS: dict[str, str] = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}
MINIMAX_DEFAULT_REGION = "global_en"


def resolve_minimax_base_url(region: str = "") -> str:
    """Return the configured MiniMax OpenAI-compatible endpoint."""
    selected = str(region or "").strip().lower() or MINIMAX_DEFAULT_REGION
    return MINIMAX_REGION_ENDPOINTS.get(selected, MINIMAX_REGION_ENDPOINTS[MINIMAX_DEFAULT_REGION])


# DeepSeek serves one official OpenAI-compatible endpoint (no regions, no
# owner-configurable base URL — a proxy/mirror setup belongs to the generic
# openai-compatible slot). Kept as a module constant so transport and the
# provider test resolve the same host; the reviewer-window/base-url fingerprint
# maps deliberately have NO deepseek branch (both default to "" consistently —
# the fingerprint is already unique per provider+model).
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# DeepSeek's Chat Completions ``reasoning_effort`` enum is low/high/max
# (medium/xhigh are documented aliases of high; ultra aliases max) and thinking
# is switched off by ``thinking.type=disabled``, not by an effort value. This
# is the wire dialect of one provider, projected at the physical-send boundary;
# the canonical Ouroboros effort scale stays the SSOT everywhere else. The
# mapping lives in EFFORT_ROUTE_ALIASES_LMH below (the ONE shared
# low/high/max-dialect table: DeepSeek's row and the GLM-family z.ai row of
# EFFORT_ROUTE_DESCRIPTOR both read it); this name is the historical import.


def normalize_deepseek_reasoning_effort(value: str) -> str:
    """Project one canonical effort tier onto DeepSeek's Chat wire enum."""
    normalized = str(value or "").strip().lower()
    return DEEPSEEK_REASONING_EFFORT_ALIASES.get(normalized, normalized)


# --- Reasoning-effort route descriptor (SSOT for effort carriage) ---------------
# One structural table answers, per registered provider route: WHERE the effort
# tier rides on the wire (the carrier), WHICH provider tier values the route
# accepts, HOW a canonical Ouroboros tier projects onto those values, and WHAT
# an ABSENT parameter means on that route. Before this table each builder
# hard-coded its own branch (openai carries, deepseek projects, everything else
# silently drops), so a tier requested against a no-carrier route vanished with
# no record — the failure class this descriptor makes structurally impossible.
#
# Measurement discipline (deliberately conservative):
#   * deepseek / GLM-family low-high-max dialects: measured enums.
#   * openai / anthropic / openrouter / claudexor: measured wire contracts.
#   * cloudru / minimax / gigachat / generic openai-compatible / local: NOT
#     measured for effort carriage → carrier "none" with an honest absent_meaning.
#     "zai", "qwen" (DashScope) and "kimi" (Moonshot) are NOT registered routes
#     in this baseline (PR #1194 closed unmerged); per the DO-NOT rule their
#     carriers are declared "none" — no analogy to GLM without a key/measurement.
#     When a measured route lands, it becomes one table row, not a builder edit.
#
# Fields per route:
#   carrier: "reasoning_effort" | "extra_body.reasoning" | "anthropic.adaptive"
#            | "reasoningEffort" | "none"
#   tiers:   the provider-side enum values the route accepts (labels for UI);
#            empty when carrier is "none".
#   project: canonical tier -> provider tier. Identity rows are implicit; only
#            re-mapping rows are stored. Absent canonical tier → identity.
#   absent_meaning (REQUIRED even for carrier "none"): what the route does when
#            the parameter is not sent — "max" (provider silently reasons at its
#            top tier: z.ai's behavior, a real cost trap), "provider_default"
#            (the provider's own default applies), or "off" (no reasoning
#            parameter exists; sending one is an error).
#   forced_tool_suppression: whether the route requires thinking disabled when a
#            forced tool_choice is used (DeepSeek: thinking accepts only
#            auto/none tool_choice, so a forced call ships thinking disabled —
#            reason "provider_forced_tool_choice" on the clamp disclosure).
#            Do NOT copy this exception to GLM-family routes: measured
#            tool_choice required/named work WITH thinking on glm-5.3.
EFFORT_ROUTE_ALIASES_LMH = {
    # Shared alias projection for the low/high/max wire dialects (DeepSeek and
    # the measured GLM-family z.ai route — Egor's measurement 2026-09-21):
    # none/minimal/low→low, medium/high→high, xhigh/ultra→max. ONE mapping;
    # DeepSeek's normalize helper reads the same table. (``none`` on DeepSeek
    # is intercepted by the builder as the thinking-disable toggle BEFORE this
    # projection — see forced_tool_suppression.)
    "none": "low",
    "minimal": "low",
    "medium": "high",
    "xhigh": "max",
    "ultra": "max",
}
DEEPSEEK_REASONING_EFFORT_ALIASES = EFFORT_ROUTE_ALIASES_LMH

EFFORT_ROUTE_DESCRIPTOR: dict[str, dict] = {
    "openai": {
        "carrier": "reasoning_effort",
        "tiers": ["minimal", "low", "medium", "high"],
        "project": {},
        "absent_meaning": "provider_default",
        "forced_tool_suppression": False,
    },
    "deepseek": {
        "carrier": "reasoning_effort",
        "tiers": ["low", "high", "max"],
        "project": EFFORT_ROUTE_ALIASES_LMH,
        "absent_meaning": "provider_default",
        "forced_tool_suppression": True,
    },
    # GLM through the generic openai-compatible lane (z.ai PAYG and Coding Plan
    # endpoints both speak the OpenAI-compatible shape). The enum and alias
    # projection are byte-identical to DeepSeek's dialect (measured on a live
    # Coding Plan key, PR #1194 close comment), so this row shares ONE table.
    # Absent tier means MAX billing — the loudest absent_meaning in the set.
    "zai-glm": {
        "carrier": "reasoning_effort",
        "tiers": ["low", "high", "max"],
        "project": EFFORT_ROUTE_ALIASES_LMH,
        "absent_meaning": "max",
        "forced_tool_suppression": False,
    },
    "anthropic": {
        "carrier": "anthropic.adaptive",
        "tiers": ["low", "medium", "high"],
        "project": {"minimal": "low"},
        "absent_meaning": "provider_default",
        "forced_tool_suppression": False,
    },
    "openrouter": {
        "carrier": "extra_body.reasoning",
        "tiers": ["minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
        "project": {},
        "absent_meaning": "provider_default",
        "forced_tool_suppression": False,
    },
    "claudexor": {
        "carrier": "reasoningEffort",
        "tiers": ["minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
        "project": {},
        "absent_meaning": "provider_default",
        "forced_tool_suppression": False,
    },
    # Unmeasured routes: honest "none". The tier is dropped (and disclosed as
    # effort_not_carried); absent_meaning records what the route does without it.
    "openai-compatible": {
        "carrier": "none", "tiers": [], "project": {},
        "absent_meaning": "provider_default", "forced_tool_suppression": False,
    },
    "cloudru": {
        "carrier": "none", "tiers": [], "project": {},
        "absent_meaning": "provider_default", "forced_tool_suppression": False,
    },
    "minimax": {
        "carrier": "none", "tiers": [], "project": {},
        "absent_meaning": "provider_default", "forced_tool_suppression": False,
    },
    "gigachat": {
        "carrier": "none", "tiers": [], "project": {},
        "absent_meaning": "off", "forced_tool_suppression": False,
    },
    "local": {
        "carrier": "none", "tiers": [], "project": {},
        "absent_meaning": "off", "forced_tool_suppression": False,
    },
    "zai": {
        # Placeholder row for the unmerged PR #1194 route: no key, no
        # measurement in THIS tree → carrier none by the DO-NOT rule.
        "carrier": "none", "tiers": [], "project": {},
        "absent_meaning": "provider_default", "forced_tool_suppression": False,
    },
    "qwen": {
        "carrier": "none", "tiers": [], "project": {},
        "absent_meaning": "provider_default", "forced_tool_suppression": False,
    },
    "kimi": {
        "carrier": "none", "tiers": [], "project": {},
        "absent_meaning": "provider_default", "forced_tool_suppression": False,
    },
}


# Which canonical tiers each UI surface may OFFER for a route: the descriptor's
# provider tiers with their canonical pre-images, computed — never hand-listed.
def canonical_tiers_for_route(descriptor: dict) -> list[str]:
    """Canonical effort choices a route's descriptor supports (empty = none).

    Each provider tier is offered under its IDENTITY canonical name when one
    exists (low/high/max are identity rows), and otherwise under a canonical
    pre-image — so the GLM low/high/max dialect offers exactly ``low, high,
    max``, never ``minimal/medium/xhigh`` aliases of the same wire values."""
    tiers = list(descriptor.get("tiers") or [])
    if not tiers:
        return []
    project = descriptor.get("project") or {}
    inverse: dict[str, list[str]] = {}
    for canonical, provider in project.items():
        inverse.setdefault(provider, []).append(canonical)
    out: list[str] = []
    for tier in tiers:
        if project.get(tier, tier) == tier:
            # Identity row (implicit tier→tier): the provider value IS the
            # canonical spelling — offer it directly.
            out.append(tier)
        else:
            candidates = inverse.get(tier) or []
            out.append(candidates[0] if candidates else tier)
    return out


# GLM-family detection for the generic openai-compatible lane. The z.ai hosts
# serve GLM models behind an OpenAI-compatible API whose effort dialect is the
# measured low/high/max enum; a model id naming glm flags the route. Detection
# is by MODEL IDENTITY ONLY — never by a generic base_url heuristic: any vLLM
# server could host anything.
_GLM_MODEL_TOKEN = "glm"


def _is_glm_model(resolved_model: str) -> bool:
    return _GLM_MODEL_TOKEN in str(resolved_model or "").strip().lower()


def effort_descriptor_for_route(provider: str, resolved_model: str = "") -> dict:
    """Resolve the effort descriptor for one physical route.

    The z.ai GLM dialect rides the generic ``openai-compatible`` provider lane,
    so a GLM model id on that lane resolves to the measured "zai-glm" row; every
    other openai-compatible target honestly resolves to carrier "none" (no
    measurement, no guessed carriage). Unknown providers fail to the "none"
    row — never to a guessed carrier.
    """
    provider_id = str(provider or "").strip()
    if provider_id == "openai-compatible" and _is_glm_model(resolved_model):
        return EFFORT_ROUTE_DESCRIPTOR["zai-glm"]
    return EFFORT_ROUTE_DESCRIPTOR.get(
        provider_id,
        {"carrier": "none", "tiers": [], "project": {},
         "absent_meaning": "provider_default", "forced_tool_suppression": False},
    )


def project_effort_for_route(descriptor: dict, canonical_effort: str) -> str:
    """Project a canonical tier onto a route's provider enum via its table."""
    value = str(canonical_effort or "").strip().lower()
    if not value:
        return value
    return str((descriptor.get("project") or {}).get(value, value))


def _effort_descriptor_mismatches_observation(
    descriptor: dict,
    *,
    observed_ceiling: str = "",
    observed_floor: str = "",
) -> bool:
    """DECLARED vs OBSERVED loud-fact predicate (spec point 3).

    An observation recorded by capability_evidence (an effort ceiling a
    provider actually rejected above, or a floor it required below) disagrees
    with the descriptor when the observation names a canonical tier that the
    descriptor's declared provider tiers cannot express — e.g. an observed
    ``ultra`` ceiling on a route whose declared enum tops out at ``max``.
    Callers surface a true verdict as a loud fact (test + record), never as a
    silent table override: the descriptor changes only through a measured
    table edit.
    """
    from ouroboros.config import effort_rank

    if descriptor.get("carrier") == "none":
        return False
    tiers = sorted(
        set(descriptor.get("tiers") or []),
        key=lambda t: effort_rank(t),
    )
    if not tiers:
        return False
    bottom, top = tiers[0], tiers[-1]
    ceiling = str(observed_ceiling or "").strip().lower()
    floor = str(observed_floor or "").strip().lower()
    if ceiling and effort_rank(ceiling) > effort_rank(top):
        return True
    if floor and effort_rank(floor) < effort_rank(bottom):
        return True
    return False


# Direct-provider prefix → canonical provider name. Un-prefixed models route
# through OpenRouter. Order matters only for readability; prefixes are disjoint.
PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claudexor::", "claudexor"),
    ("openai::", "openai"),
    ("anthropic::", "anthropic"),
    ("minimax::", "minimax"),
    ("cloudru::", "cloudru"),
    ("gigachat::", "gigachat"),
    ("deepseek::", "deepseek"),
    ("openai-compatible::", "openai-compatible"),
    ("openrouter::", "openrouter"),
)

# Primary credential env var per provider (single-key providers).
PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "cloudru": "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Settings whose content-bound grant lets an external skill bypass the core LLM
# transport and incur model spend.  This belongs with the provider/credential
# registry, not in the frozen PluginAPI contract.
MODEL_PROVIDER_CREDENTIAL_KEYS: frozenset[str] = frozenset({
    *PROVIDER_ENV_KEYS.values(),
    "OPENAI_COMPATIBLE_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_PASSWORD",
})

# EVERY env/settings key ``llm.LLM._resolve_remote_target`` reads for a provider, GROUPED so
# a credential and the fields it is useless without travel together or not at all (GigaChat
# needs CREDENTIALS *or* USER+PASSWORD plus its endpoint/scope; Cloud.ru's key is meaningless
# against the wrong base_url; the openai-compatible lane legitimately falls back to the legacy
# OPENAI_* pair).  Deriving a per-run credential set from anything but this table is guessing:
# `anthropic/claude-sonnet-4.6` is an OPENROUTER model id, only `anthropic::…` is direct.
PROVIDER_CREDENTIAL_GROUPS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "minimax": ("MINIMAX_API_KEY", "MINIMAX_REGION"),
    "cloudru": ("CLOUDRU_FOUNDATION_MODELS_API_KEY", "CLOUDRU_FOUNDATION_MODELS_BASE_URL"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "gigachat": (
        "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD", "GIGACHAT_USER",
        "GIGACHAT_BASE_URL", "GIGACHAT_SCOPE", "GIGACHAT_VERIFY_SSL_CERTS",
    ),
    "openai-compatible": (
        "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
    ),
    "local": (),
    # The owned engine holds credentials; no control token reaches model settings/env.
    "claudexor": (),
}

# Active settings keys that hold a ROUTED model identity (prefix -> provider via
# provider_for_model). Heavy is a bounded migration/history input, not a live
# route selector; keeping the split here prevents new consumers (including
# Provider Test) from accidentally resurrecting it.
# Superset of the live slots; a key absent from settings still declares whatever
# ``config.SETTINGS_DEFAULTS`` will hand the runtime, which is why declared_model_settings()
# fills the defaults in rather than treating "unset" as "unused".
ACTIVE_MODEL_SETTING_KEYS: tuple[str, ...] = (
    "OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS",
    "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_MODEL_FALLBACK",
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW", "OUROBOROS_WEBSEARCH_MODEL",
    "OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
)
LEGACY_MODEL_SETTING_KEYS: tuple[str, ...] = ("OUROBOROS_MODEL_HEAVY",)
# Compatibility import name. Its meaning is now explicitly the active set.
MODEL_SETTING_KEYS = ACTIVE_MODEL_SETTING_KEYS

# Settings keys whose value is a Claude Agent SDK / Claude Code model NAME (``opus[1m]``),
# NOT a routed model identity: they carry no provider prefix, so provider_for_model would
# mis-route them to OpenRouter.  Their transport is the Anthropic SDK subprocess, which
# authenticates with ANTHROPIC_API_KEY (the Claude runtime gateways:
# tools/claude_advisory_review.py), so a non-empty value DECLARES the anthropic provider.


def provider_for_model(model: str) -> str:
    """Return the execution provider for a model id (``local`` for local lanes)."""
    name = str(model or "").strip()
    if name.endswith(" (local)"):
        return "local"
    for prefix, provider in PROVIDER_PREFIXES:
        if name.startswith(prefix):
            return provider
    return "openrouter"


def parse_claudexor_model(model: str) -> tuple[str, str]:
    """Split the model transport's opaque source and model, never an account pin."""
    if not str(model).startswith("claudexor::"):
        raise ValueError("Not a Claudexor model identity")
    source, separator, native_model = str(model)[len("claudexor::"):].partition("=")
    if not separator or not source.strip() or not native_model.strip():
        raise ValueError("Claudexor models use claudexor::<source>=<model>")
    return source.strip(), native_model.strip()


def resolve_model_target(
    model: str,
    *,
    effort: str = "",
    credential_ref: str = "",
    context_window: int = 0,
) -> ResolvedModelTarget:
    """Construct the ABI-4 typed target at an EXISTING resolution seam.

    Wraps what the resolution already computed (reuse-first): the transport
    lane comes from ``provider_for_model``, and the optional facts keep their
    typed sentinels unless the calling seam genuinely resolved them. No
    parallel resolver, no pricing, no window probing — a context window is
    Capability Evidence's fact (0 = unknown, fail-open).
    """
    model_id = str(model or "").strip()
    return ResolvedModelTarget(
        model_id=model_id,
        provider_route=provider_for_model(model_id),
        credential_ref=str(credential_ref or "").strip(),
        effort=str(effort or "").strip(),
        context_window=max(0, int(context_window or 0)),
    )


def fallback_candidate_targets(active_model: str = "") -> tuple[ResolvedModelTarget, ...]:
    """The cross-model fallback candidate ladder as typed targets (ABI-4).

    Same membership and order as ``model_slots.get_fallback_models`` — a typed
    view over the ONE chain SSOT, not a second resolver. Effort stays the ""
    sentinel: the ladder resolves destinations, the dispatching round owns the
    active effort. ``provider_route`` stays the ``""`` sentinel DELIBERATELY:
    the chain's local-vs-remote dispatch lane is the loop's single global
    ``USE_LOCAL_FALLBACK`` flag (the pre-existing contract, byte-identical
    through the ABI-4 sweep), so a per-candidate route here would be a
    fabricated fact no dispatcher consumes.
    """
    from ouroboros.model_slots import get_fallback_models

    return tuple(
        ResolvedModelTarget(model_id=model, provider_route="")
        for model in get_fallback_models(active_model)
    )


def provider_has_credentials(provider: str) -> bool:
    """Whether a route is configured; a managed engine's live readiness is separate."""
    if provider == "claudexor":
        return True  # A selected engine route needs no API key in Ouroboros.
    if provider == "local":
        return True
    if provider == "openai-compatible":
        compat = str(runtime_setting("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
        legacy_key = str(runtime_setting("OPENAI_API_KEY", "") or "").strip()
        legacy_base = str(runtime_setting("OPENAI_BASE_URL", "") or "").strip()
        return bool(compat or (legacy_key and legacy_base))
    if provider == "gigachat":
        creds = str(runtime_setting("GIGACHAT_CREDENTIALS", "") or "").strip()
        user = str(runtime_setting("GIGACHAT_USER", "") or "").strip()
        password = str(runtime_setting("GIGACHAT_PASSWORD", "") or "").strip()
        return bool(creds or (user and password))
    env_key = PROVIDER_ENV_KEYS.get(provider, "OPENROUTER_API_KEY")
    return bool(str(runtime_setting(env_key, "") or "").strip())


def provider_has_credentials_in_settings(provider: str, settings: dict) -> bool:
    """Mapping-based twin used by pure config/default compilers (no ambient env)."""
    def get(key: str) -> str:
        return str((settings or {}).get(key, "") or "").strip()

    if provider == "claudexor":
        return True  # Selection declares the route; never persist a synthetic healthy bit.

    if provider == "local":
        return bool(get("LOCAL_MODEL_SOURCE"))
    if provider == "openai-compatible":
        return bool(
            get("OPENAI_COMPATIBLE_BASE_URL")
            or (get("OPENAI_API_KEY") and get("OPENAI_BASE_URL"))
        )
    if provider == "gigachat":
        return bool(
            get("GIGACHAT_CREDENTIALS")
            or (get("GIGACHAT_USER") and get("GIGACHAT_PASSWORD"))
        )
    env_key = PROVIDER_ENV_KEYS.get(provider, "OPENROUTER_API_KEY")
    return bool(get(env_key))


def model_has_credentials_in_settings(model: str, settings: dict) -> bool:
    return provider_has_credentials_in_settings(provider_for_model(model), settings)


def model_has_credentials(model: str) -> bool:
    """Return True when the model's provider has usable credentials configured."""
    return provider_has_credentials(provider_for_model(model))


def local_only_review_route_env() -> bool:
    """Whether review slots must inherit the configured local Main route."""
    local_main = str(runtime_setting("USE_LOCAL_MAIN", "") or "").strip().lower()
    if local_main not in {"1", "true", "yes", "on"}:
        return False
    return not any(
        provider_has_credentials(provider)
        for provider in (
            "openrouter", "openai", "anthropic", "minimax", "cloudru", "gigachat",
            "deepseek", "openai-compatible",
        )
    )


def review_model_uses_local(model: str) -> bool:
    """Return the transport route for a resolved review slot."""
    return local_only_review_route_env() or provider_for_model(str(model or "").strip()) == "local"


def resolve_credentialed_model(default_model: str) -> str:
    """Return ``default_model`` if its provider is credentialed, else the first
    configured model slot whose provider has credentials (light → fallback →
    main). Falls back to ``default_model`` when nothing is credentialed
    so callers surface the original provider error rather than a silent swap."""
    if model_has_credentials(default_model):
        return default_model
    # LIGHT/MAIN are single-model slots; FALLBACKS is a comma chain expanded via the
    # shared SSOT parser (which also honors the legacy singular OUROBOROS_MODEL_FALLBACK)
    # instead of testing the whole comma-string as one broken model id. Empty Light
    # (default -> Main) simply contributes nothing here.
    candidates: list[str] = []
    light = str(runtime_setting("OUROBOROS_MODEL_LIGHT", "") or "").strip()
    if light:
        candidates.append(light)
    candidates.extend(parse_fallback_chain())
    for env_name in ("OUROBOROS_MODEL",):
        raw = str(runtime_setting(env_name, "") or "").strip()
        if raw:
            candidates.append(raw)
    for candidate in candidates:
        if model_has_credentials(candidate):
            return candidate
    return default_model


def declared_model_settings(
    settings: dict,
    *,
    include_claude_sdk_defaults: bool = True,
) -> dict[str, str]:
    """Return the model slots a settings mapping DECLARES, with runtime defaults filled in.

    An absent or empty slot is not "unused": the server falls back to
    ``config.SETTINGS_DEFAULTS`` for it, so the default's provider is genuinely reachable and
    must be declared.  ``include_claude_sdk_defaults`` is a RETIRED no-op kept for caller
    compatibility: the Claude-SDK transport and its dedicated model slots are gone, so both
    values declare the same set."""
    declared: dict[str, str] = {}
    for key in MODEL_SETTING_KEYS:
        value = str((settings or {}).get(key) or "").strip()
        if not value:
            value = str(SETTINGS_DEFAULTS.get(key) or "").strip()
        if value:
            declared[key] = value
    return declared


def providers_for_declared_models(declared: dict) -> dict[str, list[str]]:
    """Map ``{settings key: model string}`` to ``{provider: sorted model strings}``.

    Comma chains (fallbacks, review triads) are expanded; the Claude-SDK slots resolve to
    ``anthropic`` by transport rather than by prefix."""
    found: dict[str, set] = {}
    for key, raw in (declared or {}).items():
        text = str(raw or "").strip()
        if not text:
            continue
        for part in text.split(","):
            model = part.strip()
            if model:
                found.setdefault(provider_for_model(model), set()).add(model)
    return {provider: sorted(models) for provider, models in sorted(found.items())}


def credential_keys_for_providers(providers) -> tuple[str, ...]:
    """Return the ordered, de-duplicated credential keys a provider set needs."""
    keys: list[str] = []
    for provider in providers:
        group = PROVIDER_CREDENTIAL_GROUPS.get(str(provider))
        if group is None:
            # Unknown provider: fail OPEN with its primary key rather than silently
            # handing a run no credential at all.
            group = tuple(filter(None, (PROVIDER_ENV_KEYS.get(str(provider), ""),)))
        for key in group:
            if key not in keys:
                keys.append(key)
    return tuple(keys)


ALL_PROVIDER_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    key for group in PROVIDER_CREDENTIAL_GROUPS.values() for key in group
)


def provider_credential_plan(
    settings: dict,
    *,
    include_claude_sdk_defaults: bool = True,
) -> dict:
    """Derive WHICH provider credentials a settings mapping's declared models actually need.

    Returns ``{declared_model_slots, providers, planned_keys, fail_open}``.  ``fail_open`` is
    the disclosed escape hatch: when nothing resolves (a settings mapping with no model slot
    at all) the plan is the FULL credential universe, because a benchmark that dies on a
    missing key at hour six is worse than one that carries a spare.  The optional
    ``include_claude_sdk_defaults`` switch is RETIRED (no-op): the Claude-SDK transport and
    its dedicated model slots are gone, so both values produce the same plan."""
    declared = declared_model_settings(settings)
    providers = providers_for_declared_models(declared)
    planned = credential_keys_for_providers(providers)
    fail_open = not planned
    if fail_open:
        planned = tuple(sorted(ALL_PROVIDER_CREDENTIAL_KEYS))
    return {
        "declared_model_slots": declared,
        "providers": providers,
        "planned_keys": sorted(planned),
        "fail_open": fail_open,
    }


OPENAI_DIRECT_DEFAULTS = {
    "main": "openai::gpt-5.6-terra",
    "heavy": "",
    "light": "openai::gpt-5.6-luna",
    "fallback": "openai::gpt-5.6-sol",
    # Deep self-review is a real slot with a SHIPPED default; without a
    # per-provider value a direct-only install keeps an unreachable
    # OpenRouter-form id it has no credential for (v6.82.0). Only providers whose
    # model genuinely carries the >=1M window this review sizes against get one —
    # Cloud.ru and GigaChat are documented BELOW that floor, so filling their slot
    # would advertise a deep review that is doomed to overflow its real route.
    #
    # Plain Sol, the same model the OpenRouter default names. A `-pro` suffix is an
    # OpenRouter routing slug, not an OpenAI model id: live-probed 2026-07-29 against
    # api.openai.com, `gpt-5.6-sol-pro` on /v1/chat/completions -> 404; the pro
    # reasoning mode exists only on /v1/responses as `reasoning.mode="pro"` (200),
    # and passing `reasoning` to /v1/chat/completions -> 400 "Unknown parameter".
    # Every LLM call in llm.py is a chat.completions call, so an owner's pinned
    # `-pro` slug lands here too (deep_self_review.deep_review_route).
    "deep_self_review": "openai::gpt-5.6-sol",
}

CLOUDRU_DIRECT_DEFAULTS = {
    "main": "cloudru::zai-org/GLM-4.7",
    "heavy": "cloudru::zai-org/GLM-4.7",
    "light": "cloudru::zai-org/GLM-4.7",
    "fallback": "cloudru::zai-org/GLM-4.7",
}

GIGACHAT_DIRECT_DEFAULTS = {
    # Ultra is available only to individuals in Freemium. GigaChat 2 Max is
    # available to personal and legal-entity paid plans as well, so it is the
    # strongest current default that does not strand B2B/CORP installs.
    "main": "gigachat::GigaChat-2-Max",
    "heavy": "gigachat::GigaChat-2-Max",
    "light": "gigachat::GigaChat-2-Max",
    "fallback": "gigachat::GigaChat-2-Max",
}

MINIMAX_DIRECT_DEFAULTS = {
    "main": "minimax::MiniMax-M3",
    "heavy": "minimax::MiniMax-M3",
    "light": "minimax::MiniMax-M2.7",
    "fallback": "minimax::MiniMax-M2.7",
    # NO deep_self_review default: MiniMax documents M3 as "up to 1M tokens" with a
    # GUARANTEED minimum of 512K (platform.minimax.io, 2026-08). Deep review sizes
    # against a firm 1M window, so the guaranteed floor decides — the slot follows
    # the Cloud.ru/GigaChat clear-instead-of-fill path; owners can opt in manually.
}

DEEPSEEK_DIRECT_DEFAULTS = {
    "main": "deepseek::deepseek-v4-pro",
    "heavy": "",
    "light": "deepseek::deepseek-v4-flash",
    "fallback": "deepseek::deepseek-v4-flash",
    # DeepSeek documents a FIRM 1M context on every v4 model (api-docs
    # 2026-08: "1M context, 384K max output" with no guaranteed-minimum
    # caveat), so the slot follows the OpenAI/Anthropic fill pattern rather
    # than the MiniMax clear-instead-of-fill path (whose guaranteed floor was
    # 512K). The route's /models endpoint publishes NO window metadata, so the
    # sizing remains evidence-driven; a missing window measurement does not
    # remove review authority (see ARCHITECTURE §7).
    "deep_self_review": "deepseek::deepseek-v4-pro",
    # No vision default: deepseek-v4-flash-vision-exp is experimental; it is
    # recognized by supports_vision() for explicit owner selection only.
}

ANTHROPIC_DIRECT_DEFAULTS = {
    "main": "anthropic::claude-opus-5",
    "heavy": "",
    "light": "anthropic::claude-sonnet-5",
    "fallback": "anthropic::claude-sonnet-5",
    # Deep self-review is a real slot with a SHIPPED default; without a
    # per-provider value a direct-only install keeps an unreachable
    # OpenRouter-form id it has no credential for (v6.82.0).
    "deep_self_review": "anthropic::claude-opus-5",
}

DIRECT_PROVIDER_DEFAULTS = {
    "openai": OPENAI_DIRECT_DEFAULTS,
    "anthropic": ANTHROPIC_DIRECT_DEFAULTS,
    "cloudru": CLOUDRU_DIRECT_DEFAULTS,
    "gigachat": GIGACHAT_DIRECT_DEFAULTS,
    "minimax": MINIMAX_DIRECT_DEFAULTS,
    "deepseek": DEEPSEEK_DIRECT_DEFAULTS,
}

# Review panels are declared as provider ROLE sequences, then compiled against
# the owner's current provider-prefixed Main/Light values. This keeps explicit
# custom models useful while making the shipped single-provider policy obvious:
# OpenAI and Anthropic run their strongest Main model three independent times.
# The older MiniMax mixed panel is preserved because that profile was not part of
# this policy change.
DIRECT_PROVIDER_REVIEW_ROLES = {
    "openai": ("main", "main", "main"),
    "anthropic": ("main", "main", "main"),
    "cloudru": ("main", "main", "main"),
    "gigachat": ("main", "main", "main"),
    "minimax": ("main", "light", "light"),
    # Strongest-main ×3 policy (same as OpenAI/Anthropic): an exclusive
    # DeepSeek install reviews with three independent thinking v4-pro calls.
    "deepseek": ("main", "main", "main"),
}

DIRECT_PROVIDER_SCOPE_DEFAULTS = {
    provider: defaults["main"]
    for provider, defaults in DIRECT_PROVIDER_DEFAULTS.items()
}

_ANTHROPIC_MODEL_ALIASES = {
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-opus-4.7": "claude-opus-4-7",
    "claude-opus-4.8": "claude-opus-4-8",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
}


def normalize_anthropic_model_id(model_id: str) -> str:
    text = str(model_id or "").strip()
    return _ANTHROPIC_MODEL_ALIASES.get(text, text)


def migrate_model_value(provider: str, value: str) -> str:
    text = str(value or "").strip()
    if provider == "openai":
        if text.startswith("openai/"):
            return f"openai::{text[len('openai/'):]}"
        return text
    if provider == "anthropic":
        if text.startswith("anthropic::"):
            return f"anthropic::{normalize_anthropic_model_id(text[len('anthropic::'):])}"
        if text.startswith("anthropic/"):
            return f"anthropic::{normalize_anthropic_model_id(text[len('anthropic/'):])}"
        return text
    if provider == "cloudru":
        if text.startswith("cloudru::"):
            return text
        if text.startswith("cloudru/"):
            return f"cloudru::{text[len('cloudru/'):]}"
        return text
    if provider == "minimax":
        if text.startswith("minimax::"):
            return text
        if text.startswith("minimax/"):
            return f"minimax::{text[len('minimax/'):]}"
        return text
    if provider == "deepseek":
        if text.startswith("deepseek::"):
            return text
        if text.startswith("deepseek/"):
            return f"deepseek::{text[len('deepseek/'):]}"
        return text
    return text


def compute_direct_review_models_fallback(
    provider: str,
    main_model: str,
    light_model: str = "",
    *,
    review_runs: int = 3,
) -> list[str]:
    """Compile a direct-provider review panel from declarative role names."""
    if provider not in DIRECT_PROVIDER_DEFAULTS:
        return []
    provider_prefix = f"{provider}::"
    main = migrate_model_value(provider, main_model)
    if not main.startswith(provider_prefix):
        return []
    light = migrate_model_value(provider, light_model) if light_model else ""
    default_light = migrate_model_value(provider, DIRECT_PROVIDER_DEFAULTS[provider].get("light", ""))
    light_slot = light if light.startswith(provider_prefix) else default_light
    role_models = {"main": main, "light": light_slot or main}
    roles = DIRECT_PROVIDER_REVIEW_ROLES.get(provider, ("main", "light", "light"))
    compiled = [role_models.get(role, main) for role in roles]
    count = max(1, int(review_runs or 3))
    return [compiled[index % len(compiled)] for index in range(count)]


# Conservative static vision map by normalized id/prefix. The OpenRouter
# /models overlay (llm.py) refines this at runtime; static knowledge only
# covers families whose vision support is long-established.
_VISION_MODEL_PREFIXES: tuple[str, ...] = (
    "openai/gpt-5", "openai/gpt-4o", "openai/gpt-4.1", "openai/o3", "openai/o4",
    "google/gemini-", "anthropic/claude-",
    "x-ai/grok-4", "x-ai/grok-3",
    "qwen/qwen-vl", "qwen/qwen2.5-vl", "qwen/qwen3-vl",
    "mistralai/pixtral", "meta-llama/llama-4", "meta-llama/llama-3.2-90b-vision",
    "openai/gpt-5.5",
    # Narrow on purpose: only the dedicated vision variant. Plain deepseek
    # chat/v4 ids stay non-vision (pinned by tests), and this slash-form
    # normalized id also names a real OpenRouter vendor namespace — the
    # OpenRouter /models overlay may refine exact ids either way.
    "deepseek/deepseek-v4-flash-vision",
)

# Runtime overlay: model_id → bool, fed from OpenRouter /models
# architecture.input_modalities by llm.py (same lifecycle as its
# supported-parameters cache).
_VISION_OVERLAY: dict = {}


def update_vision_overlay(model_id: str, supports: bool) -> None:
    normalized = normalize_model_identity(model_id)
    if normalized:
        _VISION_OVERLAY[normalized] = bool(supports)


def supports_vision(model_id: str, *, model_role: str = "",
                    model_account_override: str | None = None) -> bool | None:
    """Image capability; None means unavailable subscription metadata, not blindness.

    Subscription metadata belongs to this call's role/account, never the global
    model-id overlay. Image senders preserve input when that fact is unknown;
    the actual call can start the engine and return its normal typed refusal.
    Metadata discovery itself must not start it or buy a model generation.
    """
    # Local lanes have no vision regardless of family name; check the RAW id —
    # normalize_model_identity strips the " (local)" suffix.
    if str(model_id or "").strip().endswith(" (local)"):
        return False
    if provider_for_model(model_id) == "claudexor":
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        from ouroboros.llm import LLMClient
        from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
        from ouroboros.model_wait import current_model_wait

        source, native_model = parse_claudexor_model(model_id)
        wait = current_model_wait()
        if model_account_override is None and wait is not None:
            model_account_override = wait.overrides.get(model_role, {}).get("model_account_override")
        account = (model_account_override if model_account_override is not None
                   else model_role_option(MODEL_ACCOUNTS_KEY, model_role))
        try:
            catalog = LLMClient.claudexor_model_catalog(
                source, account or None, requested_model=native_model)
        except ClaudexorUnavailable:
            return None
        if catalog.get("source") != source or (account and catalog.get("credentialProfileId") != account):
            return None
        item = next((row for row in catalog.get("models", []) if row.get("id") == native_model), {})
        modalities = item.get("inputModalities")
        return "image" in modalities if isinstance(modalities, list) and modalities else None
    normalized = normalize_model_identity(model_id)
    if not normalized:
        return False
    if normalized in _VISION_OVERLAY:
        return _VISION_OVERLAY[normalized]
    return normalized.startswith(_VISION_MODEL_PREFIXES)


# NOTE (v6.33.0): the static per-model context-window table was REMOVED. It
# perpetually went stale (1M-beta models hard-coded to 200K, [1m] ignored). The
# agent's OWN operating window is the owner low/max context MODE (the SSOT — see
# context_budget.py / loop.py), and external-model windows are resolved by
# Capability Evidence (ouroboros.capability_evidence: confirmed provider metadata
# / local health, or route-fingerprinted owner-ack), fail-closed when unknown.


def normalize_model_identity(model: str) -> str:
    text = str(model or "").strip()
    if text.endswith(" (local)"):
        text = text[:-8]
    if text.startswith("openai::"):
        return f"openai/{text[len('openai::'):]}"
    if text.startswith("openai-compatible::"):
        return f"openai-compatible/{text[len('openai-compatible::'):]}"
    if text.startswith("cloudru::"):
        return f"cloudru/{text[len('cloudru::'):]}"
    if text.startswith("gigachat::"):
        return f"gigachat/{text[len('gigachat::'):]}"
    if text.startswith("minimax::"):
        return f"minimax/{text[len('minimax::'):]}"
    if text.startswith("deepseek::"):
        return f"deepseek/{text[len('deepseek::'):]}"
    if text.startswith("anthropic::"):
        return f"anthropic/{normalize_anthropic_model_id(text[len('anthropic::'):])}"
    if text.startswith("anthropic/"):
        return f"anthropic/{normalize_anthropic_model_id(text[len('anthropic/'):])}"
    return text
