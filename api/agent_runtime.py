"""Fail-closed guard for in-process Hermes Agent source revisions.

Hermes WebUI currently imports ``run_agent.AIAgent`` into its long-lived server
process. If the Agent checkout changes while that process is alive, Python may
combine already-cached modules with newly-read source. Refuse to reuse that
mixed runtime and require a clean WebUI restart instead.
"""

from __future__ import annotations

from pathlib import Path
import sys
import subprocess
import threading

# Retain the discovered path as a diagnostic/test-visible compatibility value;
# runtime identity is deliberately captured from the loaded module below.
from api.config import _AGENT_DIR  # noqa: F401

_RESTART_MESSAGE = (
    "Hermes Agent was updated while Hermes WebUI was running. "
    "Restart Hermes WebUI before retrying this action."
)


def _read_agent_revision(agent_dir: Path | None) -> str | None:
    """Return the checkout HEAD, or ``None`` for a non-Git/unavailable source."""
    if agent_dir is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(agent_dir), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


_AGENT_SOURCE_DIR: Path | None = None
_AGENT_REVISION: str | None = None
_AIAgent = None
_RUNTIME_LOCK = threading.Lock()


class AgentRuntimeChangedError(RuntimeError):
    """Raised when the loaded Agent runtime no longer matches its source tree."""


def _loaded_agent_source_dir() -> Path | None:
    """Return the source directory that actually supplied ``run_agent``."""
    module = sys.modules.get("run_agent")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    try:
        return Path(module_file).resolve().parent
    except (OSError, RuntimeError, TypeError):
        return None


def _capture_loaded_agent_revision() -> None:
    """Bind the guard to the checkout that supplied the loaded Agent module."""
    global _AGENT_SOURCE_DIR, _AGENT_REVISION

    source_dir = _loaded_agent_source_dir()
    if source_dir is None:
        return
    current_revision = _read_agent_revision(source_dir)
    if _AGENT_SOURCE_DIR is not None and source_dir != _AGENT_SOURCE_DIR:
        raise AgentRuntimeChangedError(_RESTART_MESSAGE)
    if _AGENT_REVISION is not None:
        if current_revision != _AGENT_REVISION:
            raise AgentRuntimeChangedError(_RESTART_MESSAGE)
        return
    _AGENT_SOURCE_DIR = source_dir
    _AGENT_REVISION = current_revision


def ensure_agent_runtime_current() -> None:
    """Reject a known Git checkout change instead of mixing Python modules."""
    if _AGENT_REVISION is None:
        return
    if _read_agent_revision(_AGENT_SOURCE_DIR) != _AGENT_REVISION:
        raise AgentRuntimeChangedError(_RESTART_MESSAGE)


def require_ai_agent_class():
    """Import ``AIAgent`` after proving the loaded source revision is current."""
    ensure_agent_runtime_current()
    from run_agent import AIAgent  # noqa: PLC0415

    _capture_loaded_agent_revision()
    return AIAgent


def get_ai_agent_class():
    """Return ``AIAgent`` while preserving the existing lazy-import retry."""
    global _AIAgent, _AGENT_REVISION

    with _RUNTIME_LOCK:
        ensure_agent_runtime_current()
        if _AIAgent is None:
            try:
                agent_class = require_ai_agent_class()
            except ImportError:
                return None
            _AIAgent = agent_class
        return _AIAgent
