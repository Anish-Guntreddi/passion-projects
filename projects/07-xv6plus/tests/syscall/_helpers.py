"""Shared helpers for the Phase 1 (syscall tracing) test suite."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


def load_syscall_numbers(project_root: Path) -> Dict[str, int]:
    """Parse kernel/syscall.h so tests build trace masks from the real
    syscall numbers instead of hardcoded magic constants that could
    silently drift out of sync with the kernel."""
    text = (project_root / "kernel" / "syscall.h").read_text()
    numbers: Dict[str, int] = {}
    for m in re.finditer(r"#define\s+SYS_(\w+)\s+(\d+)", text):
        numbers[m.group(1)] = int(m.group(2))
    return numbers


def mask_for(project_root: Path, *names: str) -> int:
    """Build a trace() bitmask covering the named syscalls."""
    nums = load_syscall_numbers(project_root)
    mask = 0
    for name in names:
        mask |= 1 << nums[name]
    return mask
