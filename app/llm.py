"""Thin wrapper that invokes the `claude` CLI in non-interactive mode.

Why the CLI instead of the SDK?
- Reuses your existing Claude Code auth — no separate API key wiring.
- Keeps dependencies light (no anthropic SDK needed at runtime).

Windows notes:
- npm installs `claude` as a `.cmd` shim, which CreateProcess can't launch directly.
  We resolve to `claude.cmd` / `claude.exe` and, for `.cmd`, invoke via `cmd.exe /c`.
- The system prompt contains newlines. To avoid CMD quoting issues, we fold system
  and user text into one stdin payload instead of passing system as an argv flag.
"""
from __future__ import annotations

import os
import shutil
import subprocess


class ClaudeCLIError(RuntimeError):
    pass


def _resolve_binary() -> str:
    for name in ("claude.exe", "claude.cmd", "claude.bat", "claude"):
        path = shutil.which(name)
        if path:
            return path
    raise ClaudeCLIError(
        "`claude` CLI not found on PATH. Install Claude Code and ensure `claude` is runnable."
    )


def _build_argv(binary: str, model: str | None) -> list[str]:
    args = ["-p"]
    if model:
        args += ["--model", model]
    # On Windows always go through cmd.exe — CreateProcess can't launch .cmd/.ps1
    # shims directly, and `claude` from the native installer may be a shell shim
    # with no extension at all. Let CMD's PATHEXT resolve it.
    if os.name == "nt":
        return ["cmd.exe", "/c", "claude", *args]
    return [binary, *args]


def complete(system: str, user: str, model: str | None = None, timeout: int = 180) -> str:
    """Run `claude -p` and return stdout. System + user are concatenated onto stdin."""
    binary = _resolve_binary()
    argv = _build_argv(binary, model)
    # Fold system into the stdin payload — avoids shell/arg-quoting issues for
    # multiline prompts on Windows.
    payload = f"SYSTEM INSTRUCTIONS:\n{system}\n\n---\n\n{user}"

    proc = subprocess.run(
        argv,
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise ClaudeCLIError(
            f"claude CLI failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()
