#!/usr/bin/env python3
"""xv6-plus test harness -- invoked by scripts/run-tests.sh.

Stages:
  1. Build kernel/kernel and fs.img via `make` (optionally `make clean`
     first).
  2. Boot smoke test (Phase 0 exit criterion): a stock, untraced boot
     reaches a shell prompt and can run a command. This check is not
     one of the discovered tests/<category>/test_*.py modules -- it
     verifies build/boot reproducibility of the pinned upstream base,
     not an original xv6-plus feature -- so it lives here directly.
     If it fails, the feature tests are skipped: they would not be
     telling us anything meaningful about a kernel that can't boot.
  3. Discover and run every tests/<category>/test_*.py module. Each
     module exposes `run(project_root: pathlib.Path) -> None` and
     signals failure by raising (AssertionError for an expectation
     that didn't hold, qemu_session.BootTimeout if the console never
     produced the expected output in time).
  4. Print a pass/fail summary and exit non-zero if anything failed.

Usage:
  scripts/run-tests.sh                 # build + full suite
  scripts/run-tests.sh --clean         # make clean first
  scripts/run-tests.sh --only syscall  # just tests/syscall/*
  scripts/run-tests.sh --skip-build    # reuse the existing build
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from qemu_session import QemuSession, BootTimeout  # noqa: E402


def run_make(*targets: str) -> None:
    cmd = ["make", "-C", str(PROJECT_ROOT), *targets]
    print(f"+ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise SystemExit(f"build failed: {' '.join(cmd)} (exit {result.returncode})")


def boot_smoke_test() -> None:
    """Phase 0 exit criterion: clean xv6-plus boots to a usable shell."""
    print("[boot] booting kernel/kernel + fs.img in QEMU ...")
    with QemuSession(PROJECT_ROOT) as q:
        q.boot_to_shell(timeout=25.0)
        out = q.run_cmd("echo xv6plus_boot_ok", prompt_timeout=10.0)
        assert "xv6plus_boot_ok" in out, (
            "boot smoke test: shell did not echo back the marker string; "
            f"got:\n{out}"
        )
    print("[boot] OK: kernel boots, init starts sh, shell runs commands")


def discover_tests(only: Optional[str]) -> List[Path]:
    tests_dir = PROJECT_ROOT / "tests"
    found: List[Path] = []
    for category_dir in sorted(tests_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        if only and category_dir.name != only:
            continue
        for f in sorted(category_dir.glob("test_*.py")):
            found.append(f)
    return found


def run_test_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AssertionError(f"{path} does not define run(project_root)")
    module.run(PROJECT_ROOT)


def print_summary(results: List[Tuple[str, bool, str]], elapsed: float) -> int:
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 60)
    print(f"xv6-plus test summary: {passed}/{len(results)} passed in {elapsed:.1f}s")
    for name, ok, _msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    if failed:
        print("\nFailure details:")
        for name, ok, msg in results:
            if not ok:
                print(f"\n--- {name} ---")
                print(msg)
    print("=" * 60)
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true", help="make clean before building")
    ap.add_argument("--only", help="only run tests under tests/<CATEGORY>/")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing build")
    args = ap.parse_args()

    if not args.skip_build:
        if args.clean:
            run_make("clean")
        run_make("kernel/kernel", "fs.img")

    results: List[Tuple[str, bool, str]] = []
    # monotonic, not time.time(): the WSL2 VM clock has been observed to
    # step backwards across host suspend/resume (also the source of the
    # occasional "make: warning: Clock skew detected" from the C build),
    # which would otherwise make this elapsed-time print go negative.
    t0 = time.monotonic()

    try:
        boot_smoke_test()
        results.append(("boot smoke test", True, ""))
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure here is a report-and-continue signal
        results.append(("boot smoke test", False, str(e)))
        print("[boot] FAILED -- skipping feature tests, a broken boot makes them meaningless")
        return print_summary(results, time.monotonic() - t0)

    for test_path in discover_tests(args.only):
        rel = test_path.relative_to(PROJECT_ROOT)
        print(f"[test] {rel} ...")
        try:
            run_test_module(test_path)
            print(f"[test] {rel}: PASS")
            results.append((str(rel), True, ""))
        except BootTimeout as e:
            print(f"[test] {rel}: FAIL (timeout)")
            results.append((str(rel), False, str(e)))
        except AssertionError as e:
            print(f"[test] {rel}: FAIL")
            results.append((str(rel), False, str(e)))
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[test] {rel}: ERROR")
            results.append((str(rel), False, tb))

    return print_summary(results, time.monotonic() - t0)


if __name__ == "__main__":
    raise SystemExit(main())
