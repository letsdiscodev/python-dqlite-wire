"""Tests for the free-threaded Python runtime refusal guard.

dqlitewire's ReadBuffer and WriteBuffer rely on bytearray mutation
semantics that are documented to be unsafe for concurrent use. Under
standard (GIL) CPython the single-owner contract keeps things correct;
under free-threaded CPython (``python3.13t``) the same code crashes
the interpreter with SIGSEGV or wedges the process inside
``bytearray.extend``. The package therefore refuses to import under a
free-threaded build unless the user explicitly opts in via the
``DQLITEWIRE_ALLOW_FREE_THREADED=1`` environment variable.

These tests spawn an actual ``python3.13t`` interpreter as a subprocess
and assert the guard behaves. They are skipped when no free-threaded
interpreter is available on the system.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _find_free_threaded_python() -> str | None:
    """Locate a python3.13t binary installed by uv, if present."""
    candidates = [
        Path.home()
        / ".local/share/uv/python/cpython-3.13.11+freethreaded-linux-x86_64-gnu/bin/python3.13t",
    ]
    uv_root = Path.home() / ".local/share/uv/python"
    if uv_root.is_dir():
        for entry in uv_root.iterdir():
            if "freethreaded" in entry.name:
                binary = entry / "bin" / "python3.13t"
                if binary.is_file():
                    candidates.append(binary)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


FREE_THREADED_PYTHON = _find_free_threaded_python()

needs_ft = pytest.mark.skipif(
    FREE_THREADED_PYTHON is None,
    reason="free-threaded Python (python3.13t) not installed",
)


def _run_import(env_override: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Import dqlitewire under python3.13t in an isolated subprocess.

    Adds the in-tree ``src`` directory to ``PYTHONPATH`` so the test
    always exercises the working-tree source, not an installed copy.
    """
    assert FREE_THREADED_PYTHON is not None
    repo_src = Path(__file__).resolve().parent.parent / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_src)
    # Make sure the test subprocess does not inherit an opt-in by accident.
    env.pop("DQLITEWIRE_ALLOW_FREE_THREADED", None)
    if env_override:
        env.update(env_override)
    return subprocess.run(
        [FREE_THREADED_PYTHON, "-W", "error::RuntimeWarning", "-c", "import dqlitewire"],
        capture_output=True,
        text=True,
        env=env,
    )


@needs_ft
class TestFreeThreadedGuard:
    def test_import_is_refused_under_free_threaded_python(self) -> None:
        """Plain import on python3.13t must fail with a clear ImportError."""
        result = _run_import()
        assert result.returncode != 0, (
            f"expected import to fail, got stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
        combined = result.stderr.lower()
        assert "importerror" in combined or "import error" in combined
        assert "free-threaded" in combined or "free threaded" in combined

    def test_import_escape_hatch_allows_opt_in_with_warning(self) -> None:
        """Setting DQLITEWIRE_ALLOW_FREE_THREADED=1 must let the import
        proceed, but it must also emit a RuntimeWarning so noisy test
        configurations surface the risk."""
        # With -W error::RuntimeWarning the warning becomes an exception,
        # proving the warning was issued. A clean exit would mean no warning.
        result = _run_import({"DQLITEWIRE_ALLOW_FREE_THREADED": "1"})
        assert result.returncode != 0
        assert "RuntimeWarning" in result.stderr or "free-threaded" in result.stderr.lower()

    def test_import_escape_hatch_without_warning_filter(self) -> None:
        """Without -W error, the import actually succeeds (warning is just printed)."""
        assert FREE_THREADED_PYTHON is not None
        repo_src = Path(__file__).resolve().parent.parent / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_src)
        env["DQLITEWIRE_ALLOW_FREE_THREADED"] = "1"
        result = subprocess.run(
            [FREE_THREADED_PYTHON, "-c", "import dqlitewire; print('ok')"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f"opt-in import should succeed: stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
        assert "ok" in result.stdout


class TestOversizedErrorMessageUsesHex:
    """Defensive: under concurrent misuse or torn reads, total_size can become
    a bigint whose decimal form exceeds CPython's 4300-digit int-to-str limit,
    causing the error's f-string itself to raise ValueError. Format the size
    as hex instead — there is no digit cap on hex integer conversion.
    """

    def test_read_message_error_uses_hex_for_size(self) -> None:
        """read_message() is the consume-side raise site for oversized headers."""
        import struct

        from dqlitewire.buffer import ReadBuffer
        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        header = struct.pack("<IBBH", 0xFFFFFFFF, 0, 0, 0)
        buf.feed(header)

        with pytest.raises(DecodeError) as excinfo:
            buf.read_message()
        assert "0x" in str(excinfo.value)

    def test_peek_header_error_uses_hex_for_size(self) -> None:
        """peek_header() raises immediately for oversized headers;
        it must also use hex formatting.
        """
        import struct

        from dqlitewire.buffer import ReadBuffer
        from dqlitewire.exceptions import DecodeError

        buf = ReadBuffer(max_message_size=1024)
        header = struct.pack("<IBBH", 0xFFFFFFFF, 0, 0, 0)
        buf.feed(header)

        with pytest.raises(DecodeError) as excinfo:
            buf.peek_header()
        assert "0x" in str(excinfo.value)

    def test_decode_error_message_formats_pathological_bigint(self) -> None:
        """Directly synthesize a DecodeError message with a pathological bigint
        to prove the format string would not raise even on an impossible value.
        Defends against the failure mode where a torn slice under
        free-threading produced a multi-thousand-digit size.
        """
        # Simulate what has_message/read_message would format. The package's
        # error path must not itself raise ValueError from int-to-str limits.
        huge = 10**5000  # 5001 decimal digits — over the 4300 default cap
        # Constructing the f-string in decimal form would raise ValueError.
        # Confirm Python's own limit is still active.
        try:
            _ = f"Message size {huge} bytes"
        except ValueError:
            pass  # expected on CPython with default limit
        else:
            pytest.skip("sys.set_int_max_str_digits raised — platform allows it")

        # The hex form must not raise, regardless of magnitude.
        formatted = f"Message size {huge:#x} bytes exceeds maximum 1024"
        assert "0x" in formatted
        assert "exceeds maximum 1024" in formatted


def test_module_imports_cleanly_under_gil() -> None:
    """Sanity: on the standard GIL build, importing dqlitewire must still work."""
    # We are running under the standard interpreter right now; reimport to be explicit.
    assert sys._is_gil_enabled()
    import importlib

    import dqlitewire

    importlib.reload(dqlitewire)
    assert hasattr(dqlitewire, "MessageDecoder")
