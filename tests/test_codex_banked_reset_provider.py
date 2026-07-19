"""Behavior coverage for Codex banked reset redemption in Providers."""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

import api.providers as providers
import api.routes as routes

ROOT = Path(__file__).resolve().parents[1]
PANELS_JS = (ROOT / "static" / "panels.js").read_text(encoding="utf-8")


class _FakeHandler:
    def __init__(self, body: dict | None = None, *, raw_body=None):
        raw = json.dumps(body if raw_body is None else raw_body).encode("utf-8")
        self.command = "POST"
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": str(len(raw))}
        self.client_address = ("127.0.0.1", 12345)
        self.status = None
        self.sent_headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.sent_headers[key] = value

    def end_headers(self):
        pass

    def payload(self):
        raw = self.wfile.getvalue().decode("utf-8")
        return json.loads(raw) if raw else {}


@contextmanager
def _noop_profile_env(*_args, **_kwargs):
    yield


def _snapshot(*, fetched_at: str = "2030-03-17T12:30:00Z", count: int = 1, pool=None, available=True):
    return SimpleNamespace(
        provider="openai-codex",
        source="usage_api" if pool is None else "usage_api_pool",
        title="Account limits",
        plan="Pro",
        windows=(
            SimpleNamespace(label="Session", used_percent=100.0 if available else 25.0, reset_at=datetime(2030, 3, 17, 17, 30, tzinfo=timezone.utc), detail=None),
            SimpleNamespace(label="Weekly", used_percent=40.0, reset_at=datetime(2030, 3, 24, 12, 30, tzinfo=timezone.utc), detail=None),
        ),
        details=("Credits balance: $12.50",),
        banked_resets=SimpleNamespace(available_count=count),
        available=True,
        unavailable_reason=None,
        fetched_at=fetched_at,
        pool=pool,
    )


def _extract_function_source(name: str) -> str:
    marker = f"async function {name}("
    start = PANELS_JS.find(marker)
    if start == -1:
        marker = f"function {name}("
        start = PANELS_JS.find(marker)
    assert start != -1, f"{name} not found"
    brace = PANELS_JS.find("{", start)
    depth = 0
    for idx in range(brace, len(PANELS_JS)):
        ch = PANELS_JS[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return PANELS_JS[start : idx + 1]
    raise AssertionError(f"unterminated function {name}")


def test_redeem_codex_reset_fails_closed_for_ambiguous_pool(monkeypatch):
    pool = {
        "total_credentials": 2,
        "queried_credentials": 2,
        "available_credentials": 2,
        "exhausted_credentials": 0,
        "failed_credentials": 0,
        "credentials": [
            {"label": "Primary", "status": "available", "windows": [], "banked_resets": {"available_count": 1}},
            {"label": "Backup", "status": "available", "windows": [], "banked_resets": {"available_count": 2}},
        ],
    }
    calls = []

    monkeypatch.setattr(providers, "_active_provider_id", lambda: "openai-codex")
    refreshes = []
    monkeypatch.setattr(providers, "_fetch_account_usage_with_profile_context", lambda provider, refresh=False: (refreshes.append(refresh) or _snapshot(count=3, pool=pool)))
    monkeypatch.setattr(providers, "invalidate_account_usage_status_cache", lambda provider_id=None: None)

    import sys
    import types

    agent_mod = types.ModuleType("agent")
    agent_mod.__path__ = []
    account_usage_mod = types.ModuleType("agent.account_usage")

    def fake_redeem_codex_reset_credit(*, force=False):
        calls.append(force)
        return {"ok": True, "message": "should not run"}

    account_usage_mod.redeem_codex_reset_credit = fake_redeem_codex_reset_credit
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.account_usage", account_usage_mod)

    result = providers.redeem_codex_reset_credit_status(force=False)

    assert result["ok"] is False
    assert result["http_status"] == 409
    assert result["redemption"]["reason_code"] == "ambiguous_pool"
    assert result["quota_status"]["account_limits"]["banked_resets"]["available_count"] == 3
    assert calls == []
    assert refreshes == [True]


def test_redeem_codex_reset_rejects_exhausted_multi_pool_before_unavailable_gate(monkeypatch):
    helper_calls = []
    quota_status = {
        "ok": False,
        "provider": "openai-codex",
        "status": "unavailable",
        "account_limits": {
            "banked_resets": {"available_count": 2},
            "pool": {
                "total_credentials": 2,
                "available_credentials": 0,
                "exhausted_credentials": 2,
                "credentials": [{"status": "exhausted"}, {"status": "exhausted"}],
            },
        },
    }
    monkeypatch.setattr(providers, "_active_provider_id", lambda: "openai-codex")
    monkeypatch.setattr(providers, "get_provider_quota", lambda provider_id=None, refresh=False: quota_status)

    import sys
    import types

    agent_mod = types.ModuleType("agent")
    agent_mod.__path__ = []
    account_usage_mod = types.ModuleType("agent.account_usage")
    account_usage_mod.redeem_codex_reset_credit = lambda **kwargs: helper_calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.account_usage", account_usage_mod)

    result = providers.redeem_codex_reset_credit_status(force=False)

    assert result["http_status"] == 409
    assert result["redemption"]["reason_code"] == "ambiguous_pool"
    assert helper_calls == []


def test_redeem_codex_reset_refreshes_and_rejects_zero_count_without_agent(monkeypatch):
    refreshes = []
    helper_calls = []
    monkeypatch.setattr(providers, "_active_provider_id", lambda: "openai-codex")
    monkeypatch.setattr(providers, "get_provider_quota", lambda provider_id=None, refresh=False: (refreshes.append(refresh) or {
        "ok": True, "provider": "openai-codex", "status": "available", "account_limits": {
            "banked_resets": {"available_count": 0}, "pool": None,
        },
    }))
    import sys
    import types

    agent_mod = types.ModuleType("agent")
    agent_mod.__path__ = []
    account_usage_mod = types.ModuleType("agent.account_usage")
    account_usage_mod.redeem_codex_reset_credit = lambda **kwargs: helper_calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.account_usage", account_usage_mod)
    result = providers.redeem_codex_reset_credit_status(force=False)

    assert result["http_status"] == 409
    assert result["redemption"]["reason_code"] == "no_banked_resets"
    assert refreshes == [True]
    assert helper_calls == []


def test_redeem_codex_reset_allows_positive_count_for_single_exhausted_pool(monkeypatch):
    helper_calls = []
    unavailable_snapshot = SimpleNamespace(
        provider="openai-codex",
        source="usage_api_pool",
        title="Account limits",
        plan=None,
        windows=(),
        details=("0/1 credentials available", "1 exhausted"),
        banked_resets={"available_count": 2},
        available=False,
        unavailable_reason="No Codex pool credentials returned available account limits.",
        fetched_at=None,
        pool={
            "total_credentials": 1,
            "available_credentials": 0,
            "exhausted_credentials": 1,
            "credentials": [{"status": "exhausted"}],
        },
    )
    monkeypatch.setattr(providers, "_fetch_account_usage_with_profile_context", lambda provider, refresh=False: unavailable_snapshot)
    aggregate_status = providers._provider_account_usage_status("openai-codex", "Codex", refresh=True)
    assert aggregate_status["status"] == "unavailable"
    assert aggregate_status["account_limits"]["available"] is False
    quota_status = {
        "ok": False,
        "provider": "openai-codex",
        "status": "unavailable",
        "account_limits": {
            "banked_resets": {"available_count": 2},
            "pool": {
                "total_credentials": 1,
                "available_credentials": 0,
                "exhausted_credentials": 1,
                "credentials": [{"status": "exhausted"}],
            },
        },
    }
    monkeypatch.setattr(providers, "_active_provider_id", lambda: "openai-codex")
    monkeypatch.setattr(providers, "get_provider_quota", lambda provider_id=None, refresh=False: quota_status)

    import sys
    import types

    agent_mod = types.ModuleType("agent")
    agent_mod.__path__ = []
    account_usage_mod = types.ModuleType("agent.account_usage")

    def fake_redeem_codex_reset_credit(*, force=False):
        helper_calls.append(force)
        return {"status": "reset", "redeemed": True, "message": "Reset redeemed."}

    account_usage_mod.redeem_codex_reset_credit = fake_redeem_codex_reset_credit
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.account_usage", account_usage_mod)

    result = providers.redeem_codex_reset_credit_status(force=False)

    assert result["ok"] is True
    assert result["quota_status"]["status"] == "unavailable"
    assert helper_calls == [False]


def test_redeem_codex_reset_calls_shared_helper_invalidates_cache_and_refreshes_quota(monkeypatch):
    calls = []
    invalidated = []
    snapshots = [
        _snapshot(count=1, fetched_at="2030-03-17T12:30:00Z"),
        _snapshot(count=0, fetched_at="2030-03-17T12:31:00Z"),
    ]

    def fake_fetch(provider, refresh=False):
        assert provider == "openai-codex"
        return snapshots.pop(0)

    monkeypatch.setattr(providers, "_active_provider_id", lambda: "openai-codex")
    monkeypatch.setattr(providers, "_fetch_account_usage_with_profile_context", fake_fetch)
    monkeypatch.setattr(providers, "invalidate_account_usage_status_cache", lambda provider_id=None: invalidated.append(provider_id))

    import sys
    import types

    agent_mod = types.ModuleType("agent")
    agent_mod.__path__ = []
    account_usage_mod = types.ModuleType("agent.account_usage")

    def fake_redeem_codex_reset_credit(*, force=False):
        calls.append(force)
        return SimpleNamespace(
            status="reset",
            message="Reset redeemed.",
            available_count=0,
            windows_reset=2,
            redeemed=True,
        )

    account_usage_mod.redeem_codex_reset_credit = fake_redeem_codex_reset_credit
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.account_usage", account_usage_mod)

    result = providers.redeem_codex_reset_credit_status(force=False)

    assert result["ok"] is True
    assert result["http_status"] == 200
    assert result["redemption"] == {
        "ok": True,
        "state": "reset",
        "message": "Reset redeemed.",
        "reason_code": None,
        "available_count": 0,
        "windows_reset": 2,
    }
    assert result["quota_status"]["account_limits"]["banked_resets"]["available_count"] == 0
    assert result["quota_status"]["account_limits"]["fetched_at"] == "2030-03-17T12:31:00Z"
    assert calls == [False]
    assert invalidated == ["openai-codex"]


def test_normalize_codex_reset_redemption_requires_real_reset_contract():
    assert providers._normalize_codex_reset_redemption(
        {"ok": True, "status": "redeemed", "message": "invented"}
    ) == {
        "ok": False,
        "state": "redeemed",
        "message": "invented",
        "reason_code": None,
    }
    assert providers._normalize_codex_reset_redemption(
        SimpleNamespace(status="reset", message="not redeemed", available_count=1, windows_reset=0, redeemed=False)
    ) == {
        "ok": False,
        "state": "reset",
        "message": "not redeemed",
        "reason_code": None,
        "available_count": 1,
        "windows_reset": 0,
    }


def test_normalize_codex_reset_redemption_falls_through_none_dict_values():
    result = providers._normalize_codex_reset_redemption(
        {
            "state": None,
            "status": "reset",
            "redeemed": True,
            "message": None,
            "detail": "Reset redeemed.",
        }
    )

    assert result["ok"] is True
    assert result["state"] == "reset"
    assert result["message"] == "Reset redeemed."


def test_result_value_falls_through_none_dict_values_to_later_names():
    assert providers._result_value({"message": None, "detail": "details"}, "message", "detail") == "details"


@pytest.mark.parametrize("value", [False, 0, ""])
def test_result_value_preserves_falsey_dict_values(value):
    assert providers._result_value({"value": value}, "value", "fallback") is value


def test_normalize_codex_reset_redemption_preserves_long_not_exhausted_guidance_and_redacts_sensitive_output():
    guidance = (
        "Current Codex usage is not exhausted. Review the account-limits output in the Codex app or web, "
        "then retry only if you intend to spend a banked reset on the current window. If you have confirmed "
        "the active window is the one you want to clear, rerun /usage reset --force to continue."
    )

    result = providers._normalize_codex_reset_redemption(
        {
            "status": "not_exhausted",
            "message": guidance,
            "available_count": 4,
            "windows_reset": 0,
            "redeemed": False,
        }
    )

    assert result == {
        "ok": False,
        "state": "not_exhausted",
        "message": guidance,
        "reason_code": None,
        "available_count": 4,
        "windows_reset": 0,
    }
    assert result["message"].endswith("rerun /usage reset --force to continue.")

    redacted = providers._normalize_codex_reset_redemption(
        {
            "status": "not_exhausted",
            "message": "Bearer secret access_token leaked from provider output",
            "redeemed": False,
        }
    )

    assert redacted == {
        "ok": False,
        "state": "not_exhausted",
        "message": "Codex reset redemption failed.",
        "reason_code": None,
    }


def test_redeem_codex_reset_sanitizes_helper_failure(monkeypatch):
    monkeypatch.setattr(providers, "_active_provider_id", lambda: "openai-codex")
    monkeypatch.setattr(providers, "_fetch_account_usage_with_profile_context", lambda provider, refresh=False: _snapshot(count=1))
    monkeypatch.setattr(providers, "invalidate_account_usage_status_cache", lambda provider_id=None: None)

    import sys
    import types

    agent_mod = types.ModuleType("agent")
    agent_mod.__path__ = []
    account_usage_mod = types.ModuleType("agent.account_usage")

    def fake_redeem_codex_reset_credit(*, force=False):
        raise RuntimeError("secret bearer access_token should not leak")

    account_usage_mod.redeem_codex_reset_credit = fake_redeem_codex_reset_credit
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.account_usage", account_usage_mod)

    result = providers.redeem_codex_reset_credit_status(force=True)

    assert result["ok"] is False
    assert result["http_status"] == 502
    assert result["redemption"]["ok"] is False
    assert "secret" not in json.dumps(result).lower()
    assert "access_token" not in json.dumps(result).lower()


def test_redeem_codex_reset_rejects_non_boolean_force_before_helper(monkeypatch):
    helper_called = []

    monkeypatch.setattr(providers, "_active_provider_id", lambda: "openai-codex")
    monkeypatch.setattr(providers, "_fetch_account_usage_with_profile_context", lambda provider, refresh=False: _snapshot(count=1))

    import sys
    import types

    agent_mod = types.ModuleType("agent")
    agent_mod.__path__ = []
    account_usage_mod = types.ModuleType("agent.account_usage")

    def fake_redeem_codex_reset_credit(*, force=False):
        helper_called.append(force)
        raise AssertionError("helper should not be called")

    account_usage_mod.redeem_codex_reset_credit = fake_redeem_codex_reset_credit
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.account_usage", account_usage_mod)

    result = providers.redeem_codex_reset_credit_status(force="yes")

    assert result["ok"] is False
    assert result["http_status"] == 400
    assert result["redemption"]["reason_code"] == "invalid_force"
    assert helper_called == []


def test_codex_reset_route_validates_body_and_uses_profile_scope(monkeypatch):
    seen = {"entered": 0}

    @contextmanager
    def fake_profile_env(path, logger_override=None):
        assert path == "/api/provider/openai-codex/reset"
        seen["entered"] += 1
        yield

    monkeypatch.setattr(routes, "_check_csrf", lambda handler: True)
    monkeypatch.setattr("api.profiles.profile_env_for_active_request", fake_profile_env)
    monkeypatch.setattr(routes, "redeem_codex_reset_credit_status", lambda force=False: {"ok": True, "http_status": 200, "quota_status": {"ok": True}, "redemption": {"ok": True}})

    wrong_provider = _FakeHandler({"provider": "openai", "force": False})
    assert routes.handle_post(wrong_provider, urlparse("/api/provider/openai-codex/reset")) is True
    assert wrong_provider.status == 400
    assert "only force" in wrong_provider.payload()["error"]

    invalid_force = _FakeHandler({"force": "yes"})
    assert routes.handle_post(invalid_force, urlparse("/api/provider/openai-codex/reset")) is True
    assert invalid_force.status == 400
    assert "force" in invalid_force.payload()["error"]

    for malformed in (None, [], "text"):
        malformed_handler = _FakeHandler(raw_body=malformed)
        assert routes.handle_post(malformed_handler, urlparse("/api/provider/openai-codex/reset")) is True
        assert malformed_handler.status == 400
        assert "JSON object" in malformed_handler.payload()["error"]

    ok = _FakeHandler({"force": False})
    assert routes.handle_post(ok, urlparse("/api/provider/openai-codex/reset")) is True
    assert ok.status == 200
    assert seen["entered"] == 1


def test_codex_reset_frontend_flow_covers_render_confirm_busy_and_rerender():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the frontend behavior harness")

    script = f"""
(async()=>{{
const assert = (cond, msg) => {{ if (!cond) throw new Error(msg); }};
const esc = (v) => String(v == null ? '' : v);
const t = (key, ...args) => {{
  const table = {{
    provider_quota_reset_busy: 'Redeeming…',
    provider_quota_reset_action: 'Redeem reset',
    provider_quota_banked_resets: 'Banked resets: {{0}}',
    provider_quota_reset_force_title: 'Redeem Codex reset?',
    provider_quota_reset_force_message: 'A full reset may be wasted because your current Codex window is not exhausted.',
    provider_quota_reset_confirm: 'Redeem',
  }};
  let text = table[key] || key;
  args.forEach((arg, idx) => {{ text = text.replace(`{{${{idx}}}}`, String(arg)); }});
  return text;
}};
{_extract_function_source('_providerQuotaResetRequestForce')}
{_extract_function_source('_providerQuotaBankedResetState')}
{_extract_function_source('_parseProviderQuotaApiError')}
{_extract_function_source('_redeemProviderQuotaReset')}

const announcer = {{ textContent: 'old announcement' }};
globalThis.$ = (id) => id === 'a11yAnnouncer' ? announcer : null;
globalThis.requestAnimationFrame = (callback) => callback();

const pooled = _providerQuotaBankedResetState({{
  provider: 'openai-codex',
  account_limits: {{
    windows: [{{remaining_percent: 0}}, {{remaining_percent: 55}}],
    banked_resets: {{available_count: 3, redeemable: false, reason_code: 'ambiguous_pool', complete: false}},
    pool: {{total_credentials: 2}},
  }},
}});
assert(pooled.canRedeem === false, 'ambiguous pool must not redeem');

const pooledUnknown = _providerQuotaBankedResetState({{
  provider: 'openai-codex',
  account_limits: {{
    windows: [{{remaining_percent: null}}, {{remaining_percent: undefined}}],
    banked_resets: null,
    pool: {{total_credentials: 2}},
  }},
}});
assert(pooledUnknown.canRedeem === false, 'unknown pool count must not redeem');

const status = {{
  provider: 'openai-codex',
  account_limits: {{
    windows: [{{remaining_percent: 25}}, {{remaining_percent: 60}}],
    banked_resets: {{available_count: 1, redeemable: true, reason_code: null}},
  }},
}};
assert(_providerQuotaResetRequestForce(status) === true, 'non-exhausted usage should require force');
assert(_providerQuotaResetRequestForce({{ provider: 'openai-codex', account_limits: {{ windows: [{{remaining_percent: null}}, {{remaining_percent: undefined}}, {{remaining_percent: 'abc'}}] }} }}) === true, 'unknown remaining percent must still require force');
assert(_providerQuotaResetRequestForce({{ provider: 'openai-codex', account_limits: {{ windows: [{{remaining_percent: 0}}, {{remaining_percent: null}}] }} }}) === false, 'finite exhausted window should skip force confirmation');
const exhaustedSingleton = {{
  provider: 'openai-codex',
  status: 'unavailable',
  account_limits: {{
    windows: [],
    pool: {{
      total_credentials: 1,
      exhausted_credentials: 1,
      credentials: [{{status: 'exhausted'}}],
    }},
  }},
}};
assert(_providerQuotaResetRequestForce(exhaustedSingleton) === false, 'explicitly exhausted singleton with empty windows should not require force');
assert(_providerQuotaResetRequestForce({{ provider: 'openai-codex', status: 'unavailable', account_limits: {{ windows: [], pool: {{ total_credentials: 1, failed_credentials: 1, credentials: [{{status: 'failed'}}] }} }} }}) === true, 'failed singleton with empty windows must require force');
assert(_providerQuotaResetRequestForce({{ provider: 'openai-codex', status: 'unavailable', account_limits: {{ windows: [], pool: {{ total_credentials: 1, credentials: [{{status: 'unknown'}}] }} }} }}) === true, 'unknown singleton with empty windows must require force');

let confirmCalls = 0;
globalThis.showConfirmDialog = async (opts) => {{
  confirmCalls += 1;
  assert(opts.title === 'Redeem Codex reset?', 'confirm title mismatch');
  assert(opts.message.includes('may be wasted'), 'confirm message mismatch');
  return true;
}};

let apiRequest = null;
globalThis.api = async (path, opts) => {{
  apiRequest = {{ path, opts }};
  return {{
    provider: 'openai-codex',
    account_limits: {{
      windows: [{{remaining_percent: 90}}, {{remaining_percent: 90}}],
      banked_resets: {{available_count: 0, redeemable: true, reason_code: null}},
    }},
    redemption: {{ ok: true, state: 'redeemed', message: 'Reset redeemed.' }},
  }};
}};

globalThis._buildProviderQuotaCard = (next) => {{
  return {{ next }};
}};

const button = {{
  disabled: false,
  textContent: 'Redeem reset',
  attrs: {{}},
  setAttribute(k, v) {{ this.attrs[k] = v; }},
  removeAttribute(k) {{ delete this.attrs[k]; }},
}};
const card = {{
  replaced: null,
  isConnected: true,
  replaceWith(node) {{ this.replaced = node; }},
}};

await _redeemProviderQuotaReset(card, button, status);
assert(confirmCalls === 1, 'expected force confirmation');
assert(apiRequest.path === '/api/provider/openai-codex/reset', 'reset endpoint mismatch');
assert(JSON.parse(apiRequest.opts.body).force === true, 'force payload mismatch');
assert(apiRequest.opts.retries === 0, 'reset request must not retry');
assert(apiRequest.opts.timeoutMs === 90000, 'reset request timeout mismatch');
assert(card.replaced && card.replaced.next.redemption.message === 'Reset redeemed.', 'fresh card should re-render with redemption state');
assert(announcer.textContent === 'Reset redeemed.', 'redemption should update the persistent announcer');

const makeButton = () => ({{
  disabled: false,
  textContent: 'Redeem reset',
  attrs: {{}},
  setAttribute(k, v) {{ this.attrs[k] = v; }},
  removeAttribute(k) {{ delete this.attrs[k]; }},
}});
const makeCard = () => ({{ isConnected: true, replaced: null, replaceWith(node) {{ this.replaced = node; }} }});
const singletonButton = makeButton();
const singletonCard = makeCard();
await _redeemProviderQuotaReset(singletonCard, singletonButton, exhaustedSingleton);
assert(confirmCalls === 1, 'explicitly exhausted singleton must not show confirmation');
assert(JSON.parse(apiRequest.opts.body).force === false, 'exhausted singleton must post force false');

const redeemRequiresForce = async (nextStatus, label) => {{
  const forceButton = makeButton();
  const forceCard = makeCard();
  const expectedConfirmCalls = confirmCalls + 1;
  await _redeemProviderQuotaReset(forceCard, forceButton, nextStatus);
  assert(confirmCalls === expectedConfirmCalls, label + ' must show confirmation');
  assert(JSON.parse(apiRequest.opts.body).force === true, label + ' must post force true');
}};
await redeemRequiresForce({{ provider: 'openai-codex', status: 'unavailable', account_limits: {{ windows: [], pool: {{ total_credentials: 1, failed_credentials: 1, credentials: [{{status: 'failed'}}] }} }} }}, 'failed singleton');
await redeemRequiresForce({{ provider: 'openai-codex', status: 'unavailable', account_limits: {{ windows: [], pool: {{ total_credentials: 1, credentials: [{{status: 'unknown'}}] }} }} }}, 'unknown singleton');

globalThis.showConfirmDialog = async () => false;
let cancelledRequest = false;
globalThis.api = async () => {{ cancelledRequest = true; throw new Error('should not call api'); }};
const cancelButton = {{
  disabled: false,
  textContent: 'Redeem reset',
  attrs: {{}},
  setAttribute(k, v) {{ this.attrs[k] = v; }},
  removeAttribute(k) {{ delete this.attrs[k]; }},
}};
const cancelCard = {{ isConnected: true, replaceWith() {{ throw new Error('should not replace'); }} }};
await _redeemProviderQuotaReset(cancelCard, cancelButton, status);
assert(cancelledRequest === false, 'cancelled confirmation must not post');
assert(cancelButton.disabled === false, 'cancelled flow must clean busy state');
assert(cancelButton.textContent === 'Redeem reset', 'cancelled flow must restore button label');
}})().catch((err)=>{{ console.error(err); process.exit(1); }});
"""

    subprocess.run([node, "-e", script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_codex_reset_frontend_markup_uses_header_action_and_keeps_pool_counts():
    header_start = PANELS_JS.index("<div class=\"provider-quota-header\">")
    header_end = PANELS_JS.index("<div class=\"provider-quota-body\">", header_start)
    header = PANELS_JS[header_start:header_end]
    assert "data-provider-quota-reset" in header
    assert 'class="provider-quota-refresh" type="button" data-provider-quota-reset' in header
    assert "provider_quota_reset_action')+' ('+bankedResetState.count+')" in header
    assert "bankedResetHtml" not in PANELS_JS
    assert "_buildProviderQuotaPoolBreakdown(accountLimits)" in PANELS_JS
    assert "provider-quota-pool-note" in PANELS_JS
    assert "$('a11yAnnouncer')" in PANELS_JS
    assert "setAttribute('aria-live'" not in PANELS_JS


def test_codex_reset_i18n_uses_english_fallback_keys_only():
    src = (ROOT / "static" / "i18n.js").read_text(encoding="utf-8")
    locale_pattern = re.compile(r"^\s{2}(?:[A-Za-z0-9_]+|'[^']+'):\s*\{$", re.MULTILINE)
    locale_keys = locale_pattern.findall(src)
    assert locale_keys

    from tests.test_provider_quota_locale_helpers import RESET_FALLBACK_KEYS

    reset_fallback_keys = RESET_FALLBACK_KEYS
    for key in reset_fallback_keys:
        assert src.count(f"{key}:") == 1, f"{key} should live only in LOCALES.en"
    assert src.count("provider_quota_resets_meta:") == len(locale_keys)
