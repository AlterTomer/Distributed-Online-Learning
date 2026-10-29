r"""The reproduction driver, stages 1--3: the environment, the tests, every runner's smoke.

    python scripts/reproduce.py                     # all three stages, on the CPU
    python scripts/reproduce.py --device cuda       # the same, smokes on the GPU
    python scripts/reproduce.py --stage 3           # one stage
    python scripts/reproduce.py --stage 3 --from run_adamw_pass   # resume the smoke tier

`REPRODUCE.md` sections 14--17 are the recipe this walks. Stages 4--7 (calibration, the
experiments, the reports, the figures) are not here yet; they follow the merge of the
two branches (`docs/schedule.md`, "The reproduction package").

## What each stage does

1. **The environment.** Records the git SHA, Python, torch, CUDA, the GPU and
   `pip freeze`; downloads MNIST once (`check_data.py`); runs `check_environment.py`.
2. **The tests.** The full suite, `python -m pytest` (1 505 tests on `main`; one is
   marked slow, so there is nothing to gain from skipping it).
3. **Smoke.** Every runner with a `--smoke` mode, 20-odd steps at one seed, in the
   order they depend on each other. The producers come first -- X19's lr pass, X20,
   X25, X26, X25+ and P5.3 -- because the AdamW pass, P5.12, the float32 and delta
   probes and the codec read their smoke cells. That chain is what lets the tier run
   on a fresh clone, where none of the real cells exist.

On a laptop CPU (i7-13620H) the three stages take about 3.4 hours (D141).

## Decisions (asked and answered, design note D141)

* **Every smoke runs with `--fresh`.** Runs are cached by name, so without it a
  second smoke would find its cells complete and test nothing.
* **A failure does not stop the driver.** Every step runs, and the summary lists what
  ran, what refused and what failed; the exit code is nonzero if anything did not pass.
  A dependent of a failed producer refuses, and the refusal names the missing cells.
* **Output goes to `reproduce/out/<timestamp>/`** (git-ignored): `environment.json`,
  `pip_freeze.txt`, one log per step under `logs/`, and `summary.txt`.
* **The CPU by default, `--device cuda` on request.** The smoke tier must run on any
  machine; a GPU makes it faster, and every runner honours the device it is given.

A smoke's numbers mean nothing. What it shows is that every code path a real pass
takes runs end to end on this machine, and that the merge gates hold where a smoke
compares like with like (the AdamW pass and P5.12 do).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

#: (name, script, extra flags). Order is dependency order: a step reads only what the
#: steps above it wrote. `--smoke`, `--fresh` and `--device` are added to every one.
SMOKE_STEPS: list[tuple[str, str, list[str]]] = [
    # The producers: their smoke cells are what the dependents below are rebuilt from.
    ("x19 --lr", "run_diffusion_ekf.py", ["--lr"]),
    ("x20 --tune", "run_diffusion_tuning.py", ["--tune"]),
    ("x20", "run_diffusion_tuning.py", []),
    ("x25 --lr", "run_diffusion_skew.py", ["--lr"]),
    ("x25", "run_diffusion_skew.py", []),
    ("x26 --lr", "run_atc_plain.py", ["--lr"]),
    ("x26", "run_atc_plain.py", []),
    ("x25+", "run_skew_topup.py", []),
    ("p53 --lr", "run_diffusion_topology.py", ["--lr"]),
    ("p53", "run_diffusion_topology.py", []),
    # REPRODUCE.md section 17, in its order.
    *[(f"{stem} {flag}".strip(), f"run_{stem}.py", [flag] if flag else [])
      for stem in ("network_size", "p57_heterogeneous_drift", "belief_calibration",
                   "break_rate", "label_shift", "label_sparsity", "midtransient_shift",
                   "adamw_pass", "inner_steps")
      for flag in ("--lr", "")],
    ("disagreement", "run_disagreement.py", []),
    ("float32_probe", "run_float32_probe.py", []),
    ("delta_entropy_probe", "run_delta_entropy_probe.py", []),
    ("covariance_blocks", "run_covariance_blocks.py", []),
    ("codec --calibrate", "run_codec.py", ["--calibrate"]),
    ("codec --validate", "run_codec.py", ["--validate"]),
    ("codec --report-runs", "run_codec.py", ["--report-runs"]),
]

#: What a merge gate prints when it does not hold (the AdamW pass, P5.12).
GATE_FAILURES = ("DOES NOT REPRODUCE", "FAILS")

CUDA_NOTE = """\
################################################################################
#                                                                              #
#   Running on the CPU.  If you have a CUDA GPU, use it:                       #
#                                                                              #
#       python scripts/reproduce.py --device cuda                              #
#                                                                              #
#   The CPU is the default only so that this runs on any machine. Every        #
#   runner honours the device it is given, and the GPU is much faster.         #
#                                                                              #
################################################################################"""


@dataclass
class Step:
    """One line of the summary."""

    stage: int
    name: str
    status: str = "not run"     # passed / refused / failed / timeout / not run
    minutes: float = 0.0
    note: str = ""
    log: str = ""


@dataclass
class Run:
    out: Path
    device: str
    timeout: float
    steps: list[Step] = field(default_factory=list)


def call(run: Run, stage: int, name: str, command: list[str], log_name: str) -> Step:
    """Run one command, log everything it prints, and classify the outcome."""
    step = Step(stage, name, log=f"logs/{log_name}.txt")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    print(f"[stage {stage}] {name:<34}", end="", flush=True)
    started = time.time()
    try:
        done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=run.timeout)
        code, out = done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired as expired:
        code, out = None, _text(expired.stdout) + _text(expired.stderr)
    step.minutes = (time.time() - started) / 60
    (run.out / step.log).write_text(out, encoding="utf-8")

    refusal = next((line.strip() for line in out.splitlines() if "REFUSED" in line), "")
    gate = next((line.strip() for line in out.splitlines()
                 if any(f in line for f in GATE_FAILURES)), "")
    if code is None:
        step.status, step.note = "timeout", f"killed after {run.timeout / 60:.0f} min"
    elif refusal:
        step.status, step.note = "refused", refusal
    elif code != 0:
        step.status = "failed"
        step.note = f"exit {code}: " + (out.strip().splitlines() or [""])[-1].strip()
    elif gate:
        step.status, step.note = "failed", f"merge gate: {gate}"
    else:
        step.status = "passed"
    print(f"{step.status:<8} {step.minutes:6.1f} min", flush=True)
    run.steps.append(step)
    return step


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _probe(command: list[str]) -> str:
    """What a tool prints, or why it printed nothing."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable ({type(error).__name__})"
    if done.returncode != 0:
        return f"exit {done.returncode}: {(done.stderr or done.stdout).strip()[:200]}"
    return done.stdout.strip()


def stage_environment(run: Run) -> None:
    """Stage 1: what this is running on, the data, and the environment check."""
    import torch  # noqa: PLC0415

    record = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": _probe(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": bool(_probe(["git", "-C", str(ROOT), "status", "--porcelain"])),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": _probe(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"]),
        "device": run.device,
    }
    (run.out / "environment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (run.out / "pip_freeze.txt").write_text(_probe([PY, "-m", "pip", "freeze"]) + "\n",
                                            encoding="utf-8")
    print(f"  git {record['git_sha'][:7]}{' (dirty)' if record['git_dirty'] else ''}, "
          f"torch {record['torch']}, CUDA {record['torch_cuda']}, GPU {record['gpu']}\n",
          flush=True)
    if run.device == "cuda" and not record["cuda_available"]:
        run.steps.append(Step(1, "cuda", "failed", note="--device cuda, but torch sees no GPU"))
        print("  --device cuda was asked for, but torch sees no GPU.", flush=True)

    call(run, 1, "check_data", [PY, "-c", (
        "import sys; sys.path.insert(0, 'scripts'); import check_data; "
        "raise SystemExit(check_data.main(show=False))")], "1_check_data")
    call(run, 1, "check_environment", [PY, "scripts/check_environment.py"],
         "1_check_environment")


def stage_tests(run: Run) -> None:
    """Stage 2: the whole suite."""
    step = call(run, 2, "pytest", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                "2_pytest")
    tail = (run.out / step.log).read_text(encoding="utf-8").strip().splitlines()
    step.note = step.note or (tail[-1].strip("= ") if tail else "")


def stage_smoke(run: Run, start: str | None) -> None:
    """Stage 3: every runner's smoke, producers first."""
    names = [name for name, _s, _f in SMOKE_STEPS]
    first = 0
    if start:
        # An exact name wins, so `--from midtransient_shift` skips its `--lr` step.
        matches = [i for i, name in enumerate(names) if start == name] or \
                  [i for i, name in enumerate(names)
                   if start in (name.split()[0], f"run_{name.split()[0]}")]
        if not matches:
            raise SystemExit(f"--from {start!r} names no step; the steps are: {', '.join(names)}")
        first = matches[0]
    for index, (name, script, flags) in enumerate(SMOKE_STEPS):
        if index < first:
            continue
        command = [PY, f"scripts/{script}", *flags, "--smoke", "--fresh",
                   "--device", run.device]
        call(run, 3, name, command, f"3_{index + 1:02d}_{name.replace(' ', '_').replace('-', '')}")


def write_summary(run: Run, stages: list[int], started: float) -> int:
    counts: dict[str, int] = {}
    for step in run.steps:
        counts[step.status] = counts.get(step.status, 0) + 1
    lines = [f"stage{'s' if len(stages) > 1 else ''} {', '.join(map(str, stages))} "
             f"on {run.device}, "
             f"{(time.time() - started) / 60:.1f} min; "
             + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())), ""]
    lines += [f"  {s.stage}  {s.name:<34} {s.status:<8} {s.minutes:6.1f} min  {s.note}"
              for s in run.steps]
    text = "\n".join(lines) + "\n"
    (run.out / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"logs and records: {run.out}")
    return 0 if all(s.status == "passed" for s in run.steps) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3],
                        help="which stages to run")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="where the smokes run; cuda is faster if you have it")
    parser.add_argument("--from", dest="start", metavar="STEP",
                        help="start the smoke tier at this step, e.g. run_adamw_pass")
    parser.add_argument("--timeout", type=float, default=120,
                        help="minutes before one step is killed")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default reproduce/out/<timestamp>)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    out = args.out or ROOT / "reproduce" / "out" / time.strftime("%Y%m%d-%H%M%S")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    run = Run(out=out, device=args.device, timeout=args.timeout * 60)
    if args.device == "cpu":
        print(CUDA_NOTE + "\n", flush=True)
    started = time.time()
    stages = sorted(set(args.stage))
    if 1 in stages:
        stage_environment(run)
    if 2 in stages:
        stage_tests(run)
    if 3 in stages:
        stage_smoke(run, args.start)
    code = write_summary(run, stages, started)
    if args.device == "cpu":
        print("\n" + CUDA_NOTE)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
