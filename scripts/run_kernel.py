#!/usr/bin/env python3
"""Push a Kaggle kernel, poll it to completion, pull its output.

Everything this project runs on Kaggle goes through here, so that
`results/experiment_log.csv` records how every number was produced.

Three operating rules are enforced here rather than left to discipline:

  1. ONE KERNEL AT A TIME. A lock file makes a second concurrent run impossible.
     Running several training notebooks at once exhausts the session's resources
     and they fail together.
  2. FAIL FAST. A kernel that ends in any state other than `complete` halts the
     pipeline immediately, dumps its log, and returns non-zero. Nothing downstream
     runs on a broken upstream artifact.
  3. T4 x2 ON EVERY KERNEL. `--check-accelerator` (default on) refuses to push a
     kernel whose metadata does not request nvidiaTeslaT4x2.

Usage:
    python scripts/run_kernel.py nb00_probe
    python scripts/run_kernel.py nb02_e1 --timeout 43200
    python scripts/run_kernel.py --pipeline nb01_preprocess nb02_e1 nb03_e2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "kaggle"
RESULTS = ROOT / "results"
LOG = RESULTS / "experiment_log.csv"
LOCK = ROOT / ".kaggle_run.lock"

# Kaggle's field is `machine_shape`, not `accelerator` -- an `accelerator` key in
# kernel-metadata.json is silently DROPPED and the kernel falls back to a single P100.
# `NvidiaTeslaT4` is Kaggle's dual-T4 offering (shown as "GPU T4 x2" in the UI).
REQUIRED_ACCELERATOR = "NvidiaTeslaT4"
TERMINAL_OK = {"complete"}
TERMINAL_BAD = {"error", "cancelrequested", "cancelacknowledged"}


# --------------------------------------------------------------------- utilities

def sh(cmd: list[str], check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{out}")
    return out


def git_commit() -> str:
    try:
        return sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).strip()
    except RuntimeError:
        return "unknown"


def meta_of(nb_dir: Path) -> dict:
    return json.loads((nb_dir / "kernel-metadata.json").read_text())


# ------------------------------------------------------------------ rule 1: lock

class RunLock:
    """Guarantees a single kernel in flight across every invocation of this script."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "RunLock":
        if LOCK.exists():
            held = LOCK.read_text().strip()
            raise SystemExit(
                f"refusing to start {self.name}: another Kaggle run is in flight ({held}).\n"
                "Only one notebook trains at a time. Wait for it, or remove "
                f"{LOCK} if you know it is stale."
            )
        LOCK.write_text(f"{self.name} pid={os.getpid()} since={dt.datetime.now().isoformat(timespec='seconds')}")
        return self

    def __exit__(self, *exc) -> None:
        LOCK.unlink(missing_ok=True)


# ------------------------------------------------------- rule 3: accelerator gate

def check_accelerator(nb_dir: Path) -> None:
    m = meta_of(nb_dir)
    if "accelerator" in m:
        raise SystemExit(
            f'{nb_dir.name}: kernel-metadata.json uses "accelerator", which Kaggle '
            f'ignores. Use "machine_shape": "{REQUIRED_ACCELERATOR}" instead.'
        )
    shape = m.get("machine_shape")
    if not m.get("enable_gpu") or shape != REQUIRED_ACCELERATOR:
        raise SystemExit(
            f"{nb_dir.name}: kernel-metadata.json must set "
            f'"enable_gpu": true and "machine_shape": "{REQUIRED_ACCELERATOR}"; '
            f'found enable_gpu={m.get("enable_gpu")!r} machine_shape={shape!r}'
        )


def verify_accelerator_took(slug: str) -> str:
    """Read back what Kaggle actually stored.

    The metadata round-trip is the only way to know the accelerator request was
    honoured rather than silently downgraded.
    """
    tmp = ROOT / ".kmeta"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "pull", slug, "-p", str(tmp), "-m"], check=False)
    f = tmp / "kernel-metadata.json"
    if not f.exists():
        return "unknown"
    return str(json.loads(f.read_text()).get("machine_shape", "unknown"))


# ------------------------------------------------------------------- kaggle verbs

def push(nb_dir: Path, accelerator: str | None = REQUIRED_ACCELERATOR) -> str:
    cmd = ["kaggle", "kernels", "push", "-p", str(nb_dir)]
    if accelerator:
        cmd += ["--accelerator", accelerator]  # belt and braces alongside machine_shape
    out = sh(cmd)
    print(out.strip())
    if "successfully pushed" not in out.lower():
        raise RuntimeError(f"push did not confirm success:\n{out}")
    return meta_of(nb_dir)["id"]


def status(slug: str) -> str:
    out = sh(["kaggle", "kernels", "status", slug], check=False).lower()
    if "max retries exceeded" in out or "failed to resolve" in out or "nodename nor servname" in out:
        raise ConnectionError("kaggle api unreachable")
    for state in ("complete", "error", "cancelrequested", "cancelacknowledged", "running", "queued"):
        if state in out:
            return state
    return "unknown"


def dump_logs(slug: str, tail: int = 120) -> str:
    """Pull the kernel's own log so a failure is diagnosable without the browser."""
    raw = sh(["kaggle", "kernels", "output", slug, "-p", str(ROOT / ".kfail")], check=False)
    lines: list[str] = []
    logf = next((ROOT / ".kfail").glob("*.log"), None)
    if logf:
        try:
            for rec in json.loads("[" + logf.read_text().strip().rstrip(",") + "]"):
                lines.append(str(rec.get("data", "")).rstrip())
        except Exception:  # noqa: BLE001
            lines = logf.read_text().splitlines()
    return "\n".join(lines[-tail:]) if lines else raw[-4000:]


def try_cancel(slug: str) -> str:
    """Best-effort remote cancel.

    The Kaggle CLI exposes no `kernels cancel` verb, so a run that has already
    started cannot be killed from here. What this script CAN guarantee -- and what
    actually matters -- is that a failed or hung kernel stops the pipeline dead and
    never has a successor pushed alongside it.
    """
    return (f"note: the Kaggle CLI has no cancel verb; {slug} was abandoned, not killed. "
            "Stop it from the web UI if it is still consuming quota.")


def wait(slug: str, timeout: int, poll: int = 45) -> str:
    """Poll to completion, tolerating transient local network failures.

    The kernel runs on Kaggle and is entirely unaffected by this machine losing DNS
    or Wi-Fi. An exception here previously killed the poller and left a perfectly
    healthy finished run un-pulled, so connectivity errors are logged and retried
    rather than raised. Only a long unbroken outage gives up.
    """
    started = time.time()
    last = None
    consecutive_errors = 0
    max_consecutive_errors = 40  # ~30 min of unbroken failure at the default poll

    while time.time() - started < timeout:
        try:
            st = status(slug)
            consecutive_errors = 0
        except Exception as e:  # noqa: BLE001
            consecutive_errors += 1
            print(f"[{(time.time() - started) / 60:6.1f} min] status check failed "
                  f"({consecutive_errors}/{max_consecutive_errors}): "
                  f"{type(e).__name__}. The kernel is unaffected; retrying.", flush=True)
            if consecutive_errors >= max_consecutive_errors:
                return "unreachable"
            time.sleep(poll)
            continue

        if st != last:
            print(f"[{(time.time() - started) / 60:6.1f} min] {slug} -> {st}", flush=True)
            last = st
        if st in TERMINAL_OK or st in TERMINAL_BAD:
            return st
        time.sleep(poll)
    return "timeout"


def pull(slug: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(sh(["kaggle", "kernels", "output", slug, "-p", str(dest)], check=False).strip())


def log_run(nb: str, slug: str, state: str, elapsed: float) -> None:
    RESULTS.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc", "notebook", "kernel", "git_commit", "state", "minutes"])
        w.writerow([
            dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            nb, slug, git_commit(), state, f"{elapsed / 60:.1f}",
        ])


# ------------------------------------------------------------------------ driver

def run_one(nb: str, timeout: int, enforce_accel: bool) -> bool:
    nb_dir = KAGGLE_DIR / nb
    if not nb_dir.is_dir():
        raise SystemExit(f"no such notebook directory: {nb_dir}")
    if enforce_accel:
        check_accelerator(nb_dir)

    with RunLock(nb):
        t0 = time.time()
        slug = push(nb_dir, REQUIRED_ACCELERATOR if enforce_accel else None)
        if enforce_accel:
            got = verify_accelerator_took(slug)
            if got != REQUIRED_ACCELERATOR:
                print(f"\n=== {nb} ABORTED: Kaggle stored machine_shape={got!r}, "
                      f"expected {REQUIRED_ACCELERATOR!r} ===", file=sys.stderr)
                print("Refusing to run on the wrong accelerator.", file=sys.stderr)
                return False
            print(f"accelerator confirmed: {got}")
        state = wait(slug, timeout)
        elapsed = time.time() - t0
        log_run(nb, slug, state, elapsed)

        if state == "complete":
            pull(slug, ROOT / "kaggle_out" / nb)
            print(f"\n=== {nb} complete in {elapsed / 60:.1f} min ===")
            return True

        print(f"\n=== {nb} FAILED: state '{state}' after {elapsed / 60:.1f} min ===", file=sys.stderr)
        print(try_cancel(slug), file=sys.stderr)
        print("\n--- kernel log (tail) ---", file=sys.stderr)
        print(dump_logs(slug), file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", help="directory names under kaggle/")
    ap.add_argument("--timeout", type=int, default=43200, help="seconds per notebook")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="skip the T4 x2 accelerator check (not for training runs)")
    args = ap.parse_args()

    for i, nb in enumerate(args.notebooks):
        if i:
            print(f"\n{'=' * 70}\nnext: {nb}\n{'=' * 70}")
        if not run_one(nb, args.timeout, enforce_accel=not args.allow_cpu):
            remaining = args.notebooks[i + 1:]
            if remaining:
                print(f"halting pipeline; not running: {', '.join(remaining)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
