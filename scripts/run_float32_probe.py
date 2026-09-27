r"""The float32 probe: can the filters run in single precision, and what would it save?

    python scripts/run_float32_probe.py --device cuda            # lockstep, seeds 0-1 (~1-1.5 h)
    python scripts/run_float32_probe.py --cells --device cuda    # float32 twins of the source cells
    python scripts/run_float32_probe.py --report-only
    python scripts/run_float32_probe.py --smoke                  # the path, on CPU, in minutes

Run with the GPU otherwise idle: the lockstep's timings are the point, and a second
process on the card makes every one of them wrong.

## Why

Every run so far is float64 (D58). The reasons, strongest first: the covariance
update $\boldsymbol P-\boldsymbol A\boldsymbol S^{-1}\boldsymbol A^{\mathsf T}$ subtracts two nearly equal
matrices without the Joseph form's protection (D62); the Diff-EKF note says the
recursion loses definiteness in single precision -- uncited, and about the
*unsymmetrised* recursion (D133); the exactness gates (X0, M1) need $10^{-12}$; and
`run.dtype` is one setting per run, so the gradient baselines ride along. **None of it
has been measured on this code.** Consumer Ada GPUs run FP64 at 1/64 of their FP32
rate, and the filter's cost is dense $p\times p$ algebra, so if float32 holds, much of
the GPU queue shrinks.

## Two parts

**1. Lockstep** (default; the measurement). Per seed, the source cell's filters are
built twice -- float64 exactly as recorded, and float32 -- from **one** $\boldsymbol\theta_0$,
drawn in float64 and cast, and advanced **side by side on identical observations** (the
float64 environment's, cast). Every difference is therefore arithmetic: no data, no
initialisation. (A float32 run draws its own $\boldsymbol\theta_0$ in float32, which
consumes the seed's stream differently and starts somewhere else entirely; the first
version of this probe did that and measured two initialisations, not two precisions.)
Each learner's adapt/combine is timed per step, CUDA synchronised. Every
`--check-every` steps, per filter: the mean's relative distance from float64 (worst
agent), $\boldsymbol P$'s smallest diagonal, agent 0's smallest and largest eigenvalue in
each precision, $\boldsymbol P$'s asymmetry, a dtype-leak check, and **both precisions
scored on the same held-out set** -- the settled error and, on Mackey--Glass, the
`variance_ratio`.

**2. Cells** (`--cells`; end to end). Float32 twins of the source cells through the
ordinary runner, seeds 0--1. They draw their own $\boldsymbol\theta_0$, and on Mackey--Glass
the data depend on the horizon (D129), so they are compared with float64 as samples,
not per seed: a sanity check that a whole run completes, stays healthy and lands in
the float64 range.

## Pass criteria, named before the run (revised 2026-09-28, before any real run)

float32 is **adoptable** for a task only if all of these hold for every filter, on
every lockstep seed:

1. **Health.** No guard trips, no dtype leak, and at every checkpoint
   $\lambda_{\min}(\boldsymbol P_{32}) \ge -10^{-5}\,\lambda_{\max}(\boldsymbol P_{32})$ -- rounding
   in a $p\approx2\,500$ matrix can put a tiny eigenvalue a hair below zero; a
   loss of definiteness is orders beyond that.
2. **Fidelity** (lockstep). Settled error (RMSE on Mackey--Glass, error rate on MNIST)
   within 0.0005 of float64, on each seed: identical start and data, so any gap is
   precision's alone.
3. **Calibration** (lockstep, Mackey--Glass). Settled `variance_ratio` within 0.01.
4. **Worth it.** The filters' per-step time at least 2x faster.

The mean's drift from float64 has no criterion: the online trajectory amplifies
rounding the way it amplifies any perturbation, and what matters is the outcome, which
criteria 2--3 measure.

## What adopting it would cost

**Bitwise pairing with every float64 cell on disk.** A float32 cell cannot pass a
reproduction gate against a float64 one, so M7, M9, M10, M13 and M14 -- all built on
M6's recorded cells -- and the AdamW backfill stay float64 unless their reference
cells are re-run in float32. The exactness gates stay float64 regardless.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402

from dekf_bench.evaluation import protocol  # noqa: E402
from dekf_bench.learners.registry import POOLING, build_learners  # noqa: E402
from dekf_bench.likelihoods.registry import build_likelihood  # noqa: E402
from dekf_bench.models.registry import build_model_from_config  # noqa: E402
from dekf_bench.runner.simulate import _advance  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: The source cells, by task: the filters' groups a and b, float64 as recorded.
SOURCES = {
    "mackey_glass": {"a": "m6_stationary_a", "b": "m6_stationary_b"},
    "mnist": {"a": "p53_er030_a", "b": "p53_er030_b"},
}
#: Reference arms not needed to answer a precision question: M6's gamma = 0.9995 twins.
#: Task-specific -- on MNIST `centralized_ekf_gamma` *is* the centralised filter.
DROPPED = {"mackey_glass": {"centralized_ekf_gamma", "diffusion_ekf_onehop_mean_receiver_gamma"},
           "mnist": set()}
METRIC = {"mackey_glass": "rmse", "mnist": "error_rate"}
LOCKSTEP_SEEDS = [0, 1]
CELL_SEEDS = [0, 1]
HEALTH_RATIO = 1e-5
FIDELITY = 5e-4
CALIBRATION = 0.01
SPEEDUP = 2.0
SETTLED = 0.8
SMOKE_HORIZON, SMOKE_CHECK = 30, 10
DATA_ROOT = ROOT / "data"


def task_of() -> str:
    """Which task this checkout's M6/P5.3 cells belong to: the series branch has M6."""
    return "mackey_glass" if (ROOT / "results" / SOURCES["mackey_glass"]["a"]).exists() else "mnist"


def source_path(cell: str) -> Path:
    return ROOT / "results" / cell / "config.yaml"


def is_filter(entry: dict, task: str) -> bool:
    return "ekf" in entry["name"] and entry["name"] not in DROPPED[task]


def filter_entries(groups: list[str], task: str) -> list[dict]:
    out = []
    for group in groups:
        recorded = yaml.safe_load(source_path(SOURCES[task][group]).read_text(encoding="utf-8"))
        out += [dict(e) for e in recorded["learners"] if is_filter(e, task)]
    return out


def config_at(task: str, dtype: str, entries: list[dict], args, seed: int,
              horizon: int | None = None, name: str = "f32_probe_lockstep"):
    run = {"name": name, "dtype": dtype, "device": args.device, "seeds": [seed]}
    if horizon:
        run["horizon"] = horizon
    return load_config(source_path(SOURCES[task]["a"]), overrides={"run": run, "learners": entries})


def splits(config):
    """MNIST's splits, or (None, None) for a generated series."""
    if getattr(config.env, "is_series", False):
        return None, None
    from dekf_bench.data.registry import load_dataset  # noqa: PLC0415

    return load_dataset(config.env.dataset, DATA_ROOT, download=False)


def build_task(config, seed, train, test):
    try:
        from dekf_bench.runner.task import build_task as task_builder  # noqa: PLC0415
    except ImportError:  # main, before the series task: images only
        from dekf_bench.env.environment import build_environment  # noqa: PLC0415
        from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: PLC0415

        environment = build_environment(config, seed, train)
        return environment, build_evalsets(config, environment, test)
    return task_builder(config, seed, train, test)


def device_of(environment) -> torch.device:
    """Where the run's tensors live: a series environment says; an image one holds its data."""
    if hasattr(environment, "device"):
        return torch.device(environment.device)
    return environment.train.images.device


def assemble(config, train, test, seed: int, theta0: torch.Tensor | None = None):
    """(environment, evalsets, learners, likelihood, theta0) for one seed.

    ``theta0``, when given, is cast to this config's dtype and used instead of a fresh
    draw -- which is how the lockstep's two precisions start from one point.
    """
    environment, evalsets = build_task(config, seed, train, test)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    learners = build_learners(config, model, likelihood)
    dtype = getattr(torch, config.run.dtype)
    if theta0 is None:
        theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    theta0 = theta0.to(device=device_of(environment), dtype=dtype)
    for learner in learners.values():
        learner.init(theta0)
    return environment, evalsets, learners, likelihood, theta0


def build(config, train, test, seed: int):
    """(environment, learners): the delta probe's entry point."""
    environment, _evalsets, learners, _likelihood, _theta0 = assemble(config, train, test, seed)
    return environment, learners


def cast(value, dtype):
    """Floating tensors to `dtype`; labels and everything else untouched."""
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(dtype)
    return value


def cast_observation(observation, dtype):
    changes = {f.name: cast(getattr(observation, f.name), dtype)
               for f in dataclasses.fields(observation)
               if isinstance(getattr(observation, f.name), torch.Tensor)}
    return dataclasses.replace(observation, **changes)


def pool(environment, observations):
    """The union of every agent's batch: a method on the series environment, a module
    function on main's image environment."""
    if hasattr(environment, "pool"):
        return environment.pool(observations)
    from dekf_bench.env.environment import pool as pool_images  # noqa: PLC0415

    return pool_images(observations)


def synchronise(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def health(name: str, low, high, n_nodes: int, eig: bool) -> dict:
    """Drift, smallest diagonal, dtype leak, and (agent 0) the eigenvalues in each precision."""
    nodes = [0] if name in POOLING else range(n_nodes)
    drift, diag = 0.0, math.inf
    for node in nodes:
        a, b = low.state(node), high.state(node)
        drift = max(drift, float((a.theta.double() - b.theta).norm() / b.theta.norm()))
        diag = min(diag, float(a.extras["P"].diagonal().min()))
    # torch promotes float32 x float64 to float64 without a word, so a "float32" filter
    # can quietly be a float64 one. Checked, not assumed.
    leaked = sorted({str(t.dtype) for node in nodes
                     for t in (low.state(node).theta, low.state(node).extras["P"])
                     if t.dtype != torch.float32})
    row = {"drift": drift, "min_diag32": diag, "dtype_leak": leaked}
    if eig:
        p32 = low.state(0).extras["P"].double()
        p64 = high.state(0).extras["P"]
        e32 = torch.linalg.eigvalsh(0.5 * (p32 + p32.T))
        e64 = torch.linalg.eigvalsh(0.5 * (p64 + p64.T))
        row.update({"eig_min32": float(e32[0]), "eig_max32": float(e32[-1]),
                    "eig_min64": float(e64[0]), "eig_max64": float(e64[-1]),
                    "asym32": float((p32 - p32.T).abs().max())})
    return row


def scores(evalsets, learner, likelihood, config, step: int, nodes: list[int],
           to_dtype: torch.dtype | None) -> dict[str, float]:
    """The learner on the `current` set, mean over agents; a float32 learner is fed
    float32 inputs and its outputs are read back in float64, so both precisions are
    scored by one metric implementation on one set."""
    def predict(node, x):
        out = learner.predict(node, x if to_dtype is None else x.to(to_dtype))
        return out.double()

    kwargs = {}
    if "predict_variance" in inspect.signature(protocol.full_evaluate).parameters \
            and getattr(likelihood, "is_regression", False) \
            and hasattr(learner, "logit_covariance"):
        def variance(node, x):
            cov = learner.logit_covariance(node, x if to_dtype is None else x.to(to_dtype))
            return cov.diagonal(dim1=-2, dim2=-1).double()
        kwargs["predict_variance"] = variance
    result = protocol.full_evaluate(evalsets, predict, likelihood, step=step, nodes=nodes,
                                    evalsets=["current"], batch_size=config.eval.batch_size,
                                    **kwargs)
    totals: dict[str, list[float]] = {}
    for row in result.as_rows():
        if row.get("evalset") == "current":
            totals.setdefault(row["metric"], []).append(float(row["value"]))
    return {metric: sum(values) / len(values) for metric, values in totals.items()}


def lockstep_seed(task: str, args, seed: int, smoke: bool) -> dict:
    groups = ["a", "b"] if args.full_sharing else ["a"]
    entries = filter_entries(groups, task)
    horizon = SMOKE_HORIZON if smoke else None
    high_config = config_at(task, "float64", entries, args, seed, horizon)
    low_config = config_at(task, "float32", entries, args, seed, horizon)
    train, test = splits(high_config)
    environment, evalsets, high, likelihood, theta0 = assemble(high_config, train, test, seed)
    # One theta_0 for both: drawn once in float64, cast. The float32 build's own
    # environment is only a vehicle for its learners; every step uses the float64 one.
    _low_environment, _low_sets, low, _l, _t = assemble(low_config, train, test, seed,
                                                        theta0=theta0)
    del _low_environment, _low_sets
    names = list(high)
    check = SMOKE_CHECK if smoke else args.check_every
    horizon = environment.horizon
    nodes = list(range(environment.n_nodes))
    # float64 for both, as a real float32 run has it: the graph builds its weights in
    # float64 whatever run.dtype says (env/graph.py), so this is what the runner passes.
    weights = environment.graph.weights
    timing = {n: {"float64": 0.0, "float32": 0.0} for n in names}
    checkpoints: dict[str, list] = {n: [] for n in names}
    failed: dict[str, str] = {}
    print(f"  seed {seed}: {task}, T={horizon}, {', '.join(names)}, device {args.device}",
          flush=True)
    started = time.time()
    for step in range(horizon):
        observations = environment.step(step)
        pooled_x, pooled_y = pool(environment, observations)
        observations32 = {n: cast_observation(o, torch.float32) for n, o in observations.items()}
        pooled32 = (cast(pooled_x, torch.float32), cast(pooled_y, torch.float32))
        for name in names:
            synchronise(args.device)
            t0 = time.perf_counter()
            _advance(high[name], name, observations, nodes, weights, pooled_x, pooled_y)
            synchronise(args.device)
            t1 = time.perf_counter()
            timing[name]["float64"] += t1 - t0
            if name in failed:
                continue
            try:
                _advance(low[name], name, observations32, nodes, weights, *pooled32)
            except Exception as error:  # noqa: BLE001 - the guard's FilterError, or worse
                failed[name] = f"step {step}: {type(error).__name__}: {str(error)[:120]}"
                print(f"  FLOAT32 {name} FAILED at step {step}: {str(error)[:120]}", flush=True)
                continue
            synchronise(args.device)
            timing[name]["float32"] += time.perf_counter() - t1
        if step % check == 0 or step == horizon - 1:
            for name in names:
                if name in failed:
                    continue
                row = health(name, low[name], high[name], len(nodes),
                             eig=(step % (check * args.eig_every) == 0 or step == horizon - 1))
                row["step"] = step
                row["scores64"] = scores(evalsets, high[name], likelihood, high_config, step,
                                         nodes, None)
                row["scores32"] = scores(evalsets, low[name], likelihood, high_config, step,
                                         nodes, torch.float32)
                checkpoints[name].append(row)
            print(f"    step {step:>5}/{horizon}  {(time.time() - started) / 60:.1f} min",
                  flush=True)
    return {"horizon": horizon, "timing": timing, "checkpoints": checkpoints, "failed": failed,
            "steps_timed": horizon}


def out_file(task: str, suffix: str) -> Path:
    return ROOT / "results" / f"f32_probe_{task}{suffix}.json"


def cell_name(group: str, task: str, suffix: str) -> str:
    return f"f32_{SOURCES[task][group]}{suffix}"


def run_cells(args, task: str, suffix: str) -> None:
    for group in (["a", "b"] if args.full_sharing else ["a"]):
        run = {"name": cell_name(group, task, suffix), "dtype": "float32",
               "device": args.device, "seeds": CELL_SEEDS}
        if suffix:
            run.update({"horizon": SMOKE_HORIZON, "eval_every": 5, "seeds": [0]})
        config = load_config(source_path(SOURCES[task][group]), overrides={"run": run})
        train, test = splits(config)
        started = time.time()
        note = run_one(config, train, test, args.fresh)
        print(f"  {config.run.name:<28} {note:<14} {(time.time() - started) / 60:.1f} min",
              flush=True)


def settled_by_seed(cell: str, learner: str, metric: str) -> dict[int, float]:
    import pandas as pd  # noqa: PLC0415

    out = {}
    for path in sorted((ROOT / "results" / cell).glob("seed_*.parquet")):
        frame = pd.read_parquet(path)
        rows = frame[(frame.learner == learner) & (frame.metric == metric)
                     & (frame.evalset == "current")]
        if len(rows):
            rows = rows[rows.t >= int(SETTLED * rows.t.max())]
            out[int(path.stem.split("_")[1])] = float(rows.value.mean())
    return out


def report(task: str, suffix: str) -> None:
    verdicts: dict[str, list[str]] = {}
    metric = METRIC[task]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731

    def fail(name: str, why: str) -> None:
        verdicts.setdefault(name, []).append(why)

    path = out_file(task, suffix)
    if not path.exists():
        print(f"  LOCKSTEP: not run ({path.name} missing)")
    else:
        probe = json.loads(path.read_text(encoding="utf-8"))
        for seed, result in probe["seeds"].items():
            print(f"\n  LOCKSTEP: {task}, seed {seed}, T={result['horizon']}, {probe['device']}")
            print("  one theta_0 and one data stream for both precisions\n")
            print(f"    {'filter':<40}{'f64 s/step':>11}{'f32 s/step':>11}{'speedup':>9}")
            for name, t in result["timing"].items():
                per64 = t["float64"] / result["steps_timed"]
                per32 = t["float32"] / result["steps_timed"]
                speed = per64 / per32 if per32 else float("nan")
                print(f"    {name:<40}{per64:>11.4f}{per32:>11.4f}{speed:>8.1f}x")
                if not speed >= SPEEDUP:
                    fail(name, f"seed {seed}: speedup {speed:.1f}x < {SPEEDUP:g}x")
            print(f"\n    {'filter':<40}{'max drift':>10}{'min diag':>10}{'min eig/max':>12}"
                  f"{'f64 same':>10}{'settled f64':>12}{'f32 - f64':>11}{'vr f32-f64':>11}")
            for name, rows in result["checkpoints"].items():
                if name in result["failed"]:
                    print(f"    {name:<40}FAILED: {result['failed'][name]}")
                    fail(name, f"seed {seed}: guard tripped")
                    continue
                eig = [r for r in rows if "eig_min32" in r]
                worst32 = min((r["eig_min32"] / r["eig_max32"] for r in eig), default=float("nan"))
                worst64 = min((r["eig_min64"] / r["eig_max64"] for r in eig), default=float("nan"))
                last = rows[-1]["step"]
                settled = [r for r in rows if r["step"] >= SETTLED * last] or rows[-1:]
                e64 = mean([r["scores64"].get(metric, float("nan")) for r in settled])
                e32 = mean([r["scores32"].get(metric, float("nan")) for r in settled])
                vr64 = mean([r["scores64"].get("variance_ratio", float("nan")) for r in settled])
                vr32 = mean([r["scores32"].get("variance_ratio", float("nan")) for r in settled])
                print(f"    {name:<40}{max(r['drift'] for r in rows):>10.2e}"
                      f"{min(r['min_diag32'] for r in rows):>10.2e}{worst32:>12.2e}{worst64:>10.2e}"
                      f"{e64:>12.4f}{e32 - e64:>+11.5f}{vr32 - vr64:>+11.4f}")
                leaks = sorted({d for r in rows for d in r.get("dtype_leak", [])})
                if leaks:
                    print(f"    {'':<40}DTYPE LEAK: float32 state became {leaks}")
                    fail(name, f"seed {seed}: state promoted to {leaks}")
                if not worst32 >= -HEALTH_RATIO:
                    fail(name, f"seed {seed}: lambda_min/lambda_max {worst32:.1e}")
                if not abs(e32 - e64) <= FIDELITY:
                    fail(name, f"seed {seed}: settled {metric} moved {e32 - e64:+.5f}")
                if task == "mackey_glass" and not abs(vr32 - vr64) <= CALIBRATION:
                    fail(name, f"seed {seed}: variance_ratio moved {vr32 - vr64:+.4f}")

    print(f"\n  CELLS (end to end; own theta_0, so samples, not pairs): settled {metric}")
    for group in ("a", "b"):
        ours = cell_name(group, task, suffix)
        if not (ROOT / "results" / ours).exists():
            continue
        recorded = yaml.safe_load(source_path(SOURCES[task][group]).read_text(encoding="utf-8"))
        for entry in recorded["learners"]:
            name = entry["name"]
            f32 = settled_by_seed(ours, name, metric)
            f64 = settled_by_seed(SOURCES[task][group], name, metric)
            if not f32:
                continue
            print(f"    {name:<44} float32 {' '.join(f'{v:.4f}' for v in f32.values())}"
                  f"   float64 range {min(f64.values()):.4f}-{max(f64.values()):.4f}")

    if suffix:
        print("\n  SMOKE: 30 rounds on CPU. The verdict below means nothing; the path is checked.")
    print("\n  VERDICT (criteria named in the docstring):")
    if not verdicts:
        print("    every criterion measured so far holds")
    for name, reasons in verdicts.items():
        print(f"    {name:<40}" + "; ".join(reasons))


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=LOCKSTEP_SEEDS,
                          device="cuda")
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--eig-every", type=int, default=2,
                        help="eigenvalues on every k-th checkpoint (they cost ~1 s each)")
    parser.add_argument("--full-sharing", action="store_true", help="also group b's filters")
    parser.add_argument("--cells", action="store_true", help="float32 twins of the source cells")
    parser.add_argument("--smoke", action="store_true", help="30 rounds on CPU")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.device, args.seeds = "cpu", [0]
    task = task_of()
    if args.report_only:
        report(task, suffix)
        return 0
    missing = [c for c in SOURCES[task].values() if not (ROOT / "results" / c / "_complete").exists()]
    if missing:
        print(f"  REFUSED: the probe is built from {missing}, which is not finished.")
        return 1
    if args.cells:
        run_cells(args, task, suffix)
    else:
        print(f"float32 probe, lockstep: {task}, seeds {args.seeds}\n", flush=True)
        result = {"task": task, "device": args.device,
                  "seeds": {str(s): lockstep_seed(task, args, s, bool(suffix)) for s in args.seeds}}
        out_file(task, suffix).write_text(json.dumps(result, indent=2), encoding="utf-8")
    report(task, suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
