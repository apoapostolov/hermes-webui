"""Regression coverage for Hermes Agent source changes during a WebUI process lifetime."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


REPO = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_loaded_agent_runtime_fails_closed_after_source_revision_changes(tmp_path: Path):
    agent_dir = tmp_path / "hermes-agent"
    agent_dir.mkdir()
    (agent_dir / "run_agent.py").write_text(
        "class AIAgent:\n    revision = 'before'\n",
        encoding="utf-8",
    )
    _git(agent_dir, "init", "-q")
    _git(agent_dir, "add", "run_agent.py")
    _git(agent_dir, "commit", "-qm", "before")

    probe = tmp_path / "probe.py"
    probe.write_text(
        """
from pathlib import Path
import subprocess

import api.streaming as streaming
from api import agent_runtime

agent_dir = Path(__file__).parent / "hermes-agent"
assert agent_runtime._AGENT_DIR == agent_dir.resolve()
assert streaming._get_ai_agent().revision == "before"

(agent_dir / "run_agent.py").write_text(
    "class AIAgent:\\n    revision = 'after'\\n",
    encoding="utf-8",
)
subprocess.run(["git", "add", "run_agent.py"], cwd=agent_dir, check=True)
subprocess.run(
    [
        "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "after",
    ],
    cwd=agent_dir,
    check=True,
)

try:
    streaming._get_ai_agent()
except RuntimeError as exc:
    message = str(exc)
    assert "Hermes Agent was updated" in message
    assert "Restart Hermes WebUI" in message
else:
    raise AssertionError("stale in-process AIAgent was reused after its source revision changed")

try:
    agent_runtime.require_ai_agent_class()
except agent_runtime.AgentRuntimeChangedError as exc:
    assert "Restart Hermes WebUI" in str(exc)
else:
    raise AssertionError("unguarded AIAgent import was allowed after its source revision changed")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "HERMES_WEBUI_AGENT_DIR": str(agent_dir),
            "HERMES_HOME": str(tmp_path / "hermes-home"),
            "HERMES_WEBUI_STATE_DIR": str(tmp_path / "webui-state"),
            "PYTHONPATH": os.pathsep.join((str(REPO), str(agent_dir))),
        }
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_initial_non_git_source_preserves_supported_runtime(monkeypatch):
    """Non-Git installs cannot be compared, so they preserve existing behavior."""
    from api import agent_runtime

    monkeypatch.setattr(agent_runtime, "_AGENT_REVISION", None)
    monkeypatch.setattr(agent_runtime, "_read_agent_revision", lambda _path: None)

    agent_runtime.ensure_agent_runtime_current()


def test_revision_identity_comes_from_loaded_agent_module(monkeypatch, tmp_path: Path):
    """The configured discovery path must not override the loaded module path."""
    from api import agent_runtime

    configured_dir = tmp_path / "configured-agent"
    loaded_dir = tmp_path / "loaded-agent"
    for agent_dir, revision in ((configured_dir, "configured"), (loaded_dir, "loaded")):
        agent_dir.mkdir()
        (agent_dir / "run_agent.py").write_text(
            f"class AIAgent:\n    revision = '{revision}'\n",
            encoding="utf-8",
        )
        _git(agent_dir, "init", "-q")
        _git(agent_dir, "add", "run_agent.py")
        _git(agent_dir, "commit", "-qm", revision)

    loaded_module = types.ModuleType("run_agent")
    loaded_module.__file__ = str(loaded_dir / "run_agent.py")
    monkeypatch.setitem(sys.modules, "run_agent", loaded_module)
    monkeypatch.setattr(agent_runtime, "_AGENT_DIR", configured_dir)
    monkeypatch.setattr(agent_runtime, "_AGENT_SOURCE_DIR", None)
    monkeypatch.setattr(agent_runtime, "_AGENT_REVISION", None)

    agent_runtime._capture_loaded_agent_revision()

    assert agent_runtime._AGENT_SOURCE_DIR == loaded_dir.resolve()
    assert agent_runtime._AGENT_REVISION == _git(loaded_dir, "rev-parse", "HEAD")


def test_known_revision_becoming_unreadable_fails_closed(monkeypatch):
    """Losing a previously-known revision is indistinguishable from source drift."""
    from api import agent_runtime

    monkeypatch.setattr(agent_runtime, "_AGENT_REVISION", "known-revision")
    monkeypatch.setattr(agent_runtime, "_read_agent_revision", lambda _path: None)

    with pytest.raises(agent_runtime.AgentRuntimeChangedError):
        agent_runtime.ensure_agent_runtime_current()


def test_import_recapture_cannot_downgrade_known_revision(monkeypatch, tmp_path: Path):
    """A second unreadable revision read must not erase a known identity."""
    from api import agent_runtime

    source_dir = tmp_path / "loaded-agent"
    source_dir.mkdir()
    loaded_module = types.ModuleType("run_agent")
    loaded_module.__file__ = str(source_dir / "run_agent.py")
    loaded_module.__dict__["AIAgent"] = type("AIAgent", (), {})
    monkeypatch.setitem(sys.modules, "run_agent", loaded_module)
    monkeypatch.setattr(agent_runtime, "_AGENT_SOURCE_DIR", source_dir.resolve())
    monkeypatch.setattr(agent_runtime, "_AGENT_REVISION", "known-revision")
    revisions = iter(("known-revision", None))
    monkeypatch.setattr(agent_runtime, "_read_agent_revision", lambda _path: next(revisions))

    with pytest.raises(agent_runtime.AgentRuntimeChangedError):
        agent_runtime.require_ai_agent_class()

    assert agent_runtime._AGENT_REVISION == "known-revision"


def test_runner_local_bypasses_webui_agent_barrier(monkeypatch):
    """Runner-local execution is owned by the runner, not this process."""
    from api import routes

    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda _cfg: False)
    monkeypatch.setattr(routes, "get_config", lambda: {})
    monkeypatch.setattr(
        routes,
        "ensure_agent_runtime_current",
        lambda: (_ for _ in ()).throw(
            AssertionError("runner-local must not inspect the WebUI Agent checkout")
        ),
    )
    monkeypatch.setattr(
        "api.runtime_adapter.runtime_adapter_runner_enabled",
        lambda: True,
    )

    assert routes._agent_runtime_barrier_response(runner_local_owned=True) is None


def test_runner_flag_does_not_bypass_webui_owned_hidden_turns(monkeypatch):
    """Runner/gateway modes do not transfer ownership of legacy hidden turns."""
    from api import agent_runtime, routes

    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda _cfg: True)
    monkeypatch.setattr(routes, "get_config", lambda: {})
    monkeypatch.setattr(
        "api.runtime_adapter.runtime_adapter_runner_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        routes,
        "ensure_agent_runtime_current",
        lambda: (_ for _ in ()).throw(
            agent_runtime.AgentRuntimeChangedError("restart required")
        ),
    )

    response = routes._agent_runtime_barrier_response(runner_local_owned=False)

    assert response == {
        "error": "restart required",
        "type": "agent_runtime_stale",
        "retryable": True,
    }


def test_chat_start_rejects_stale_runtime_before_session_materialization(monkeypatch):
    """A stale local runtime must not claim, create, or mutate session state."""
    from api import agent_runtime, routes

    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda _cfg: False)
    monkeypatch.setattr(routes, "get_config", lambda: {})

    def stale():
        raise agent_runtime.AgentRuntimeChangedError("restart required")

    monkeypatch.setattr(routes, "ensure_agent_runtime_current", stale)

    def must_not_materialize(*_args, **_kwargs):
        raise AssertionError("session materialized before stale-runtime barrier")

    monkeypatch.setattr(routes, "_get_or_materialize_session", must_not_materialize)
    monkeypatch.setattr(
        routes,
        "j",
        lambda _handler, payload, status=200: {"status": status, "payload": payload},
    )

    response = routes._handle_chat_start(object(), {"session_id": "session-1"})

    assert response == {
        "status": 409,
        "payload": {
            "error": "restart required",
            "type": "agent_runtime_stale",
            "retryable": True,
        },
    }


def test_runner_owned_start_run_does_not_enter_local_stream_barrier(monkeypatch):
    """The runner adapter must not invoke the WebUI in-process delegate."""
    from api import routes

    calls = []

    class RunnerClient:
        def start_run(self, request):
            calls.append(request)
            return {
                "run_id": "runner-run-1",
                "stream_id": "runner-stream-1",
                "session_id": request.session_id,
                "status": "started",
            }

    session = types.SimpleNamespace(session_id="session-1", profile=None)
    monkeypatch.setenv("HERMES_WEBUI_RUNTIME_ADAPTER", "runner-local")
    monkeypatch.setattr("api.runtime_adapter.runtime_adapter_enabled", lambda: False)
    monkeypatch.setattr("api.runtime_adapter.runtime_adapter_runner_enabled", lambda: True)
    monkeypatch.setattr(routes, "_runtime_runner_client_factory", lambda: RunnerClient())
    monkeypatch.setattr(
        routes,
        "_start_chat_stream_for_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("runner-owned start entered the WebUI local stream path")
        ),
    )
    monkeypatch.setattr(
        routes,
        "ensure_agent_runtime_current",
        lambda: (_ for _ in ()).throw(
            AssertionError("runner-owned start inspected the WebUI Agent revision")
        ),
    )

    response = routes._start_run(
        session,
        msg="hello",
        attachments=[],
        workspace="/tmp/workspace",
        model="test-model",
        model_provider="test-provider",
        normalized_model=False,
        source="webui",
        route="/api/chat/start",
    )

    assert response.get("stream_id") == "runner-stream-1", response
    assert len(calls) == 1


def test_gateway_owned_start_run_bypasses_local_runtime_barrier(monkeypatch):
    """Gateway chat must reach its worker without inspecting the local Agent."""
    from api import routes

    session = types.SimpleNamespace(session_id="session-1", profile=None)
    captured = {}
    monkeypatch.setattr("api.runtime_adapter.runtime_adapter_enabled", lambda: False)
    monkeypatch.setattr("api.runtime_adapter.runtime_adapter_runner_enabled", lambda: False)
    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda _cfg: True)
    monkeypatch.setattr(routes, "get_config", lambda: {})
    monkeypatch.setattr(
        routes,
        "ensure_agent_runtime_current",
        lambda: (_ for _ in ()).throw(
            AssertionError("gateway-owned start inspected the WebUI Agent revision")
        ),
    )

    def start_gateway(_session, **kwargs):
        captured.update(kwargs)
        return {"stream_id": "gateway-stream-1", "session_id": "session-1"}

    monkeypatch.setattr(routes, "_start_chat_stream_for_session", start_gateway)

    response = routes._start_run(
        session,
        msg="hello",
        attachments=[],
        workspace="/tmp/workspace",
        model="test-model",
        model_provider="test-provider",
        normalized_model=False,
        source="webui",
        route="/api/chat/start",
    )

    assert response["stream_id"] == "gateway-stream-1"
    assert captured["external_runtime_owned"] is True


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("_handle_btw", {"session_id": "session-1", "question": "question"}),
        ("_handle_background", {"session_id": "session-1", "prompt": "prompt"}),
        ("_handle_chat_sync", {"session_id": "session-1", "message": "message"}),
    ],
)
def test_hidden_turn_routes_reject_stale_runtime_before_session_creation(
    monkeypatch, route, body
):
    """Hidden-turn endpoints must reject before creation even in gateway mode."""
    from api import routes

    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda _cfg: True)
    monkeypatch.setattr(routes, "get_config", lambda: {})
    monkeypatch.setattr(
        "api.runtime_adapter.runtime_adapter_runner_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        routes,
        "ensure_agent_runtime_current",
        lambda: (_ for _ in ()).throw(
            routes.AgentRuntimeChangedError("restart required")
        ),
    )
    monkeypatch.setattr(
        routes,
        "j",
        lambda _handler, payload, status=200: {"status": status, "payload": payload},
    )
    monkeypatch.setattr(
        routes,
        "get_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("session loaded before stale-runtime barrier")
        ),
    )

    response = getattr(routes, route)(object(), body)

    assert response["status"] == 409
    assert response["payload"]["type"] == "agent_runtime_stale"


def test_server_side_turn_rejects_stale_runtime_before_session_acceptance(monkeypatch):
    """The direct process-wakeup entrypoint must fail before loading/mutating state."""
    from api import routes

    monkeypatch.setattr(
        routes,
        "_agent_runtime_barrier_response",
        lambda **_kwargs: {
            "error": "restart required",
            "type": "agent_runtime_stale",
            "retryable": True,
        },
    )
    monkeypatch.setattr(
        routes,
        "get_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("session loaded before stale-runtime barrier")
        ),
    )

    response = routes.start_session_turn("session-1", "wake up")

    assert response == {
        "error": "restart required",
        "type": "agent_runtime_stale",
        "retryable": True,
        "_status": 409,
    }
