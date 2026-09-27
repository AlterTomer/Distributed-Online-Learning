r"""The differential-coding probe: how many bits would each learner's messages need?

    python scripts/run_delta_entropy_probe.py --device cuda      # stationary and abrupt, one seed
    python scripts/run_delta_entropy_probe.py --report-only
    python scripts/run_delta_entropy_probe.py --smoke            # the path, on CPU, in minutes

The measurement that gates C2 (`docs/communication_plan.md` §4.2--4.3, D136). The
proposal (`Diff_EKF_Huffman_Communication_Summary`) is to send each message as a
quantised difference against a public copy $\tilde{\boldsymbol\psi}$ every neighbour
holds, with a dead zone that turns small changes into zeros, then entropy-code the
symbols. Its hypothesis: successive posterior means are so correlated that
$H(Q(\Delta\boldsymbol\psi))\ll64$ bits per parameter. This measures that, for the
filters and for the gradient baselines, before anything is built.

## Open-loop, on the real messages

Every learner mixes exactly as always. A recording channel sees each vector at the
moment it is mixed -- $\boldsymbol\psi$, and the moments for ATC and ATC AdamW -- and runs
the codec beside it with its own public copy: $e=\boldsymbol x-\tilde{\boldsymbol x}$,
$q=\operatorname{round}(e/\Delta)$, $\tilde{\boldsymbol x}\leftarrow\tilde{\boldsymbol x}+q\Delta$.
**No separate residual**: the public copy advances only by what was sent, so it already
holds every untransmitted change, and adding a residual on top would count that error
twice and overshoot (D136). What the probe reports is the rate the codec would need to
*track* each learner's actual trajectory. Closed-loop, where compression feeds back
into learning, is C4.

## Two dead-zone rules

* **plain** -- uniform step $\Delta=\varepsilon\cdot\mathrm{rms}(\boldsymbol\theta_0)$ for $\boldsymbol\psi$, the
  same for every learner, so every learner faces the same worst-case distortion
  $\Delta/2$ and their rates compare **at matched distortion**. The moments use
  $\varepsilon$ times their own first message's rms. $\varepsilon\in\{10^{-2},10^{-3},10^{-4}\}$.
* **filter** -- filters only: as plain, and a change within $\kappa\sqrt{P_{ii}}$ of the
  public copy is sent as zero ($\kappa=1$: a change within one posterior standard
  deviation is not news). The decision is the sender's, and amplitudes still use
  $\Delta$, so a receiver never needs $\boldsymbol P$ -- which a mean-only filter does
  not send.

## What it reports (exploratory, one seed)

Per step and pooled over agents: the empirical entropy of the symbols (bits per
parameter under an ideal entropy coder: the note's lower bound), the fraction of
zeros, and the tracking distortion $\mathrm{rms}(\boldsymbol x-\tilde{\boldsymbol x})/\Delta$. Then, per
learner: the run mean and settled rate, and under abrupt drift the rate in the five
steps after each jump against the five before it -- the note's "the rate rises when
the learner needs to adapt".

**The two questions it answers:** is the filter's $\Delta\boldsymbol\psi$ more
compressible than the baselines' at matched distortion (the filter's gain shrinks;
ATC's constant step and AdamW's normalised one do not), and does the rate rise after
a jump and decay? If neither holds, C2's codec is not worth building.

One-hop also forwards its raw batch, uncompressed (C-1): its total per-link cost is the
rate here times $p$ plus the data, which the report states beside it.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_float32_probe import build, pool, splits  # noqa: E402

from dekf_bench.compression import Channel  # noqa: E402
from dekf_bench.env.drift import build_drift  # noqa: E402
from dekf_bench.runner.simulate import _advance  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

SOURCES = {
    "mackey_glass": {"stationary": "m6_stationary_a", "abrupt": "m6_abrupt_a"},
    "mnist": {"stationary": "p53_er030_a", "abrupt": "x20_every25_jump15_erdos_renyi"},
}
#: The mean-only diffusing learners worth measuring, in report order.
PROBED = ["diffusion_ekf", "diffusion_ekf_onehop_mean_receiver", "diffusion_ekf_onehop_mean",
          "diffusion_sgd_atc", "diffusion_sgd_atc_plain", "diffusion_atc_adamw"]
EPSILONS = [1e-2, 1e-3, 1e-4]
KAPPA = 1.0
WINDOW = 5
SMOKE_HORIZON = 30


def task_of() -> str:
    return "mackey_glass" if (ROOT / "results" / SOURCES["mackey_glass"]["stationary"]).exists() \
        else "mnist"


def entropy_bits(symbols: torch.Tensor) -> float:
    """Empirical entropy of the symbol stream, bits per symbol."""
    _values, counts = torch.unique(symbols, return_counts=True)
    p = counts.double() / counts.sum()
    return float(-(p * torch.log2(p)).sum())


class RecordingChannel(Channel):
    """Mixes exactly, and runs the differential codec beside every mixed vector."""

    def __init__(self, learner, theta_rms: float) -> None:
        super().__init__()
        self.learner = learner
        self.theta_rms = theta_rms
        self.is_filter = hasattr(learner, "covariance_sharing")
        self.call = 0
        self.scales: dict[int, float] = {}
        self.copies: dict[tuple, torch.Tensor] = {}
        # (vector, rule, eps) -> {"H": [...], "zeros": [...], "distortion": [...]}
        self.series: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def mix(self, mixing: torch.Tensor, stack: torch.Tensor) -> torch.Tensor:
        out = super().mix(mixing, stack)
        self._record(self.call, stack.detach().double())
        self.call += 1
        return out

    def end_step(self) -> None:
        self.call = 0

    def _sqrt_p(self, n: int) -> torch.Tensor:
        return torch.stack([self.learner.state(v).extras["P"].diagonal().double().clamp_min(0).sqrt()
                            for v in range(n)])

    def _record(self, k: int, x: torch.Tensor) -> None:
        rules = ["plain", "filter"] if (self.is_filter and k == 0) else ["plain"]
        if k not in self.scales:
            # The first message is the keyframe, sent in full; the codec starts after it.
            rms = float(x.pow(2).mean().sqrt())
            self.scales[k] = self.theta_rms if k == 0 else (rms or self.theta_rms)
            for rule in rules:
                for eps in EPSILONS:
                    self.copies[(k, rule, eps)] = x.clone()
            return
        sqrt_p = self._sqrt_p(x.shape[0]) if "filter" in rules else None
        for rule in rules:
            for eps in EPSILONS:
                delta = eps * self.scales[k]
                copy = self.copies[(k, rule, eps)]
                change = x - copy
                symbols = torch.round(change / delta)
                if rule == "filter":
                    symbols = torch.where(change.abs() < KAPPA * sqrt_p, torch.zeros_like(symbols),
                                          symbols)
                copy.add_(symbols * delta)
                row = self.series[(k, rule, eps)]
                row["H"].append(entropy_bits(symbols))
                row["zeros"].append(float((symbols == 0).double().mean()))
                row["distortion"].append(float((x - copy).pow(2).mean().sqrt() / delta))


def run_condition(task: str, condition: str, args, smoke: bool) -> dict:
    source = ROOT / "results" / SOURCES[task][condition] / "config.yaml"
    recorded = yaml.safe_load(source.read_text(encoding="utf-8"))
    entries = [dict(e) for e in recorded["learners"] if e["name"] in PROBED]
    run = {"name": "delta_probe", "seeds": [args.seed], "device": args.device}
    if smoke:
        run["horizon"] = SMOKE_HORIZON
    config = load_config(source, overrides={"run": run, "learners": entries})
    train, test = splits(config)
    environment, learners = build(config, train, test, args.seed)
    theta_rms = float(next(iter(learners.values())).flat_params(0).double().pow(2).mean().sqrt())
    channels = {name: RecordingChannel(learner, theta_rms) for name, learner in learners.items()}
    for name, learner in learners.items():
        learner.channel = channels[name]
    drift = build_drift(config)
    rotations = [float(drift.rotation_at(t)) for t in range(environment.horizon)]
    jumps = [t for t in range(1, environment.horizon) if rotations[t] != rotations[t - 1]]
    nodes = list(range(environment.n_nodes))
    weights = environment.graph.weights
    print(f"  {condition}: {SOURCES[task][condition]}, seed {args.seed}, T={environment.horizon}, "
          f"{', '.join(learners)}", flush=True)
    started = time.time()
    for step in range(environment.horizon):
        observations = environment.step(step)
        pooled_x, pooled_y = pool(environment, observations)
        for name, learner in learners.items():
            _advance(learner, name, observations, nodes, weights, pooled_x, pooled_y)
            channels[name].end_step()
        if step % 250 == 0:
            print(f"    step {step:>5}  {(time.time() - started) / 60:.1f} min", flush=True)
    p = next(iter(learners.values())).model.num_params
    batch = {name: getattr(learner, "_batch_scalars", 0) for name, learner in learners.items()}
    return {
        "source": SOURCES[task][condition], "horizon": environment.horizon, "jumps": jumps,
        "p": p, "theta_rms": theta_rms, "data_scalars": batch,
        "series": {name: {f"{k}|{rule}|{eps:g}": dict(values)
                          for (k, rule, eps), values in channel.series.items()}
                   for name, channel in channels.items()},
    }


def out_file(task: str, suffix: str) -> Path:
    return ROOT / "results" / f"delta_probe_{task}{suffix}.json"


VECTOR = {0: "psi", 1: "moment 1", 2: "moment 2"}


def report(task: str, suffix: str) -> None:
    path = out_file(task, suffix)
    if not path.exists():
        print(f"  {path.name} missing: run the probe first")
        return
    probe = json.loads(path.read_text(encoding="utf-8"))
    if suffix:
        print("  SMOKE: 30 steps on CPU. The numbers mean nothing; the path is checked.\n")
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    for condition, result in probe["conditions"].items():
        horizon = result["horizon"]
        settled_from = int(0.8 * horizon)
        print(f"\n  ===== {task}, {condition} ({result['source']}), p = {result['p']} =====")
        print("  bits per parameter for psi under an ideal entropy coder (fp64 reference: 64);")
        print("  zeros = fraction sent as zero; dist = rms tracking error in steps of Delta.")
        print("  Every learner shares Delta = eps * rms(theta_0): matched worst-case distortion.\n")
        for eps in EPSILONS:
            print(f"    eps = {eps:g}")
            print(f"    {'learner':<40}{'rule':<8}{'run':>7}{'settled':>9}{'zeros':>8}{'dist':>7}")
            for name, series in result["series"].items():
                for rule in ("plain", "filter"):
                    row = series.get(f"0|{rule}|{eps:g}")
                    if not row:
                        continue
                    settled = row["H"][settled_from - 1:]
                    print(f"    {name:<40}{rule:<8}{mean(row['H']):>7.2f}{mean(settled):>9.2f}"
                          f"{mean(row['zeros'][settled_from - 1:]):>8.3f}"
                          f"{mean(row['distortion']):>7.2f}")
            print()
        print("  the moments (ATC's momentum; AdamW's m and v), plain rule, run mean bits/param")
        for name, series in result["series"].items():
            cells = [f"{VECTOR[k]} {mean(series[f'{k}|plain|{eps:g}']['H']):.2f}@{eps:g}"
                     for k in (1, 2) for eps in EPSILONS if f"{k}|plain|{eps:g}" in series]
            if cells:
                print(f"    {name:<40}" + "  ".join(cells))
        data = {n: s for n, s in result["data_scalars"].items() if s}
        if data:
            print(f"  one-hop also forwards {data} raw data scalars per message, uncompressed")
        jumps = result["jumps"]
        if len(jumps) > 1:
            print(f"\n  around the {len(jumps)} jumps: psi's rate in the {WINDOW} steps after each")
            print(f"  jump against the {WINDOW} before it, plain rule (the codec's steps are")
            print("  offset by one: the first message is the keyframe)")
            for eps in EPSILONS:
                for name, series in result["series"].items():
                    h = series[f"0|plain|{eps:g}"]["H"]
                    after = [h[j - 1 + i] for j in jumps for i in range(WINDOW) if j - 1 + i < len(h)]
                    before = [h[j - 1 - i] for j in jumps for i in range(1, WINDOW + 1)
                              if 0 <= j - 1 - i < len(h)]
                    print(f"    eps {eps:<7g}{name:<40} after {mean(after):6.2f}  before "
                          f"{mean(before):6.2f}  ratio {mean(after) / max(mean(before), 1e-12):5.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0], device="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=["stationary", "abrupt"],
                        choices=["stationary", "abrupt"])
    parser.add_argument("--smoke", action="store_true", help="30 steps on CPU")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.device = "cpu"
    task = task_of()
    if args.report_only:
        report(task, suffix)
        return 0
    missing = [c for c in SOURCES[task].values() if not (ROOT / "results" / c / "_complete").exists()]
    if missing:
        print(f"  REFUSED: the probe is built from {missing}, which is not finished.")
        return 1
    print(f"delta probe: {task}, device {args.device}\n", flush=True)
    result = {"task": task, "seed": args.seed, "epsilons": EPSILONS, "kappa": KAPPA,
              "conditions": {c: run_condition(task, c, args, bool(suffix)) for c in args.conditions}}
    out_file(task, suffix).write_text(json.dumps(result), encoding="utf-8")
    report(task, suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
