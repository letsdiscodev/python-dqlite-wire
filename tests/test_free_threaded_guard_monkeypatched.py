"""Cross-host CI coverage for the free-threading guard via
``sys._is_gil_enabled`` monkeypatch in a subprocess.

The sibling ``test_free_threaded_guard`` only runs when a real
``python3.13t`` interpreter is found at well-known uv paths; on hosts
without that interpreter installed (the common CI case before
free-threading goes mainstream) every test there ``pytest.skip``-s.

These tests synthesise the no-GIL condition by spawning a standard
interpreter that monkeypatches ``sys._is_gil_enabled`` to return
False BEFORE importing dqlitewire. That gives the guard CI coverage
on every host, not just hosts with the free-threaded interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

needs_runtime_check = pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled"),
    reason="sys._is_gil_enabled is Python 3.13+",
)


def _run_subprocess(
    snippet: str, *, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``snippet`` under the standard interpreter as a fresh
    subprocess. Sets PYTHONPATH so the in-tree ``src`` is exercised."""
    repo_src = Path(__file__).resolve().parent.parent / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_src)
    env.pop("DQLITEWIRE_ALLOW_FREE_THREADED", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        env=env,
    )


@needs_runtime_check
def test_guard_raises_importerror_on_fake_no_gil() -> None:
    """Monkeypatch ``sys._is_gil_enabled()`` to return False; import
    dqlitewire; expect ImportError mentioning free-threading."""
    result = _run_subprocess(
        """
        import sys
        sys._is_gil_enabled = lambda: False
        try:
            import dqlitewire
        except ImportError as exc:
            print(f"GUARD_FIRED: {exc}", flush=True)
            sys.exit(0)
        else:
            print("GUARD_DID_NOT_FIRE", flush=True)
            sys.exit(1)
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARD_FIRED" in result.stdout
    combined = result.stdout.lower()
    assert "free-threaded" in combined or "free threaded" in combined or "no-gil" in combined


@needs_runtime_check
def test_guard_warns_on_fake_no_gil_with_opt_in() -> None:
    """Monkeypatch ``sys._is_gil_enabled()`` to return False AND set
    ``DQLITEWIRE_ALLOW_FREE_THREADED=1``; expect RuntimeWarning, no
    ImportError."""
    result = _run_subprocess(
        """
        import sys
        import warnings
        sys._is_gil_enabled = lambda: False
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            import dqlitewire
        for w in captured:
            if issubclass(w.category, RuntimeWarning):
                print(f"WARN_FIRED: {w.message}", flush=True)
                break
        else:
            print("NO_RUNTIMEWARNING", flush=True)
            sys.exit(1)
        sys.exit(0)
        """,
        env_extra={"DQLITEWIRE_ALLOW_FREE_THREADED": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN_FIRED" in result.stdout
    # Pin the load-bearing actionable phrase, mirroring
    # test_free_threaded_guard.py:102's existing assertion under the
    # python3.13t path.
    assert "single-owner-per-instance" in result.stdout


@needs_runtime_check
def test_guard_does_not_fire_under_real_gil_enabled_build() -> None:
    """Negative pin: standard interpreter (GIL enabled) imports
    cleanly. Skip when the test host happens to be running a
    free-threaded interpreter — there the import would correctly
    fire the guard, inverting the expected outcome."""
    if not sys._is_gil_enabled():
        pytest.skip("running on a free-threaded build; guard correctly fires")
    result = _run_subprocess(
        """
        import dqlitewire
        print("OK", flush=True)
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
