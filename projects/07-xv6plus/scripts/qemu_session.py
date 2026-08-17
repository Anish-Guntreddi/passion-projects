"""xv6-plus: minimal QEMU serial-console driver used by the test harness.

`make qemu` (upstream) runs qemu-system-riscv64 in the foreground and
expects a human to quit it with the QEMU monitor escape (Ctrl-A x).
That is awkward to drive reliably from a non-interactive script, so
this module launches qemu-system-riscv64 directly -- with the same
flags `make qemu` would pass (see QEMUOPTS in ../Makefile) -- as a
child process the harness owns outright, and just terminates it when
a test is done. See docs/toolchain.md for the flag-by-flag rationale.
"""

from __future__ import annotations

import os
import re
import select
import subprocess
import time
from pathlib import Path

QEMU_BIN = "qemu-system-riscv64"


class BootTimeout(RuntimeError):
    """Raised when an expected pattern never showed up on the console."""


class QemuSession:
    """One running QEMU instance of xv6-plus, driven over its serial console.

    All output (kernel prints, user program prints, and our own
    "unknown sys call" error path) shares a single UART, so stdout and
    stderr of the QEMU process are merged into one byte stream here,
    exactly as a human watching a real serial console would see it.
    """

    def __init__(self, project_root: Path, cpus: int = 3, mem: str = "128M"):
        self.project_root = project_root
        args = [
            QEMU_BIN,
            "-machine", "virt",
            "-bios", "none",
            "-kernel", "kernel/kernel",
            "-m", mem,
            "-smp", str(cpus),
            "-nographic",
            "-global", "virtio-mmio.force-legacy=false",
            "-drive", "file=fs.img,if=none,format=raw,id=x0",
            "-device", "virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0",
        ]
        self.proc = subprocess.Popen(
            args,
            cwd=str(project_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        os.set_blocking(self.proc.stdout.fileno(), False)
        self._buf = b""

    # -- raw I/O --------------------------------------------------------
    def _pump(self, timeout: float) -> None:
        if self.proc.stdout is None:
            return
        r, _, _ = select.select([self.proc.stdout], [], [], max(timeout, 0))
        if r:
            try:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
            except BlockingIOError:
                chunk = b""
            if chunk:
                self._buf += chunk

    def output(self) -> str:
        """Full accumulated console output so far, decoded leniently."""
        return self._buf.decode("utf-8", "replace")

    def send_line(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    # -- higher-level waits ----------------------------------------------
    def wait_for(self, pattern: str, timeout: float = 20.0, start: int = 0) -> re.Match:
        """Block until `pattern` (a regex) appears at/after byte offset `start`.

        `start` matters: without it, waiting for a generic prompt like
        `\\$ ` after sending a *second* command would immediately
        "succeed" against the *first* command's leftover prompt still
        sitting in the buffer, before the second command has actually
        produced any new output. Callers that care about "new output
        only" (mainly run_cmd, below) always pass the buffer length at
        the moment they sent their input.
        """
        # monotonic, not time.time(): see run_tests.py for why (WSL2 VM
        # clock has been observed to step backwards across host
        # suspend/resume, which would otherwise make `remaining` go
        # negative and time out immediately).
        deadline = time.monotonic() + timeout
        regex = re.compile(pattern, re.MULTILINE)
        while True:
            m = regex.search(self.output()[start:])
            if m:
                return m
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BootTimeout(
                    f"timed out after {timeout}s waiting for {pattern!r}; "
                    f"console output so far:\n{self.output()}"
                )
            self._pump(min(0.2, remaining))

    def drain(self, quiet_for: float = 0.5, timeout: float = 5.0) -> None:
        """Pump output until nothing new has arrived for `quiet_for` seconds."""
        deadline = time.monotonic() + timeout
        last_len = len(self._buf)
        last_change = time.monotonic()
        while time.monotonic() < deadline:
            self._pump(0.1)
            cur = len(self._buf)
            if cur != last_len:
                last_len = cur
                last_change = time.monotonic()
            elif time.monotonic() - last_change >= quiet_for:
                return

    def boot_to_shell(self, timeout: float = 20.0) -> None:
        """Wait for the kernel banner and the first shell prompt."""
        self.wait_for(r"xv6 kernel is booting", timeout=timeout)
        self.wait_for(r"\$ ", timeout=timeout)

    def run_cmd(self, cmd: str, prompt_timeout: float = 20.0) -> str:
        """Send a shell command and wait for the next prompt.

        Returns the console output produced strictly after the command
        was sent (i.e. not including anything from before), which is
        what most tests want to pattern-match against.
        """
        start = len(self._buf)
        self.send_line(cmd)
        self.wait_for(r"\$ ", timeout=prompt_timeout, start=start)
        return self._buf[start:].decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass

    def __enter__(self) -> "QemuSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
