r"""The recorder: the column contract, flushing, and exact resumption.

The load-bearing test is :func:`test_a_resumed_run_matches_an_uninterrupted_one`.
Resumption is only useful if it is *exact*, and it is exact here for a reason
worth pinning: the run loop consumes no randomness, so there is no RNG state to
restore. If that ever stops being true, this test fails rather than the results
quietly acquiring a discontinuity at the resume point.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.env.environment import build_environment
from dekf_bench.evaluation.evalsets import build_evalsets
from dekf_bench.learners.registry import build_learners
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.recording import recorder as rec
from dekf_bench.recording.schema import (
    COLUMNS,
    REQUIRED,
    RunContext,
    SchemaError,
    empty_frame,
    validate,
)
from dekf_bench.runner import simulate
from dekf_bench.utils.config import deep_merge, load_config

HORIZON = 60
EVAL_EVERY = 20


def split(n: int = 4000, seed: int = 0) -> MnistSplit:
    generator = torch.Generator().manual_seed(seed)
    return MnistSplit(
        images=torch.rand(n, 1, 28, 28, generator=generator),
        labels=torch.arange(n, dtype=torch.int64) % 10,
        split="synthetic",
    )


def config_for(**overrides):
    return load_config(
        "x1_stationary",
        overrides=deep_merge({"run": {"horizon": HORIZON, "eval_every": EVAL_EVERY}}, overrides),
    )


def pieces(config, seed: int = 0):
    train, test = split(), split(300, seed=1)
    environment = build_environment(config, seed, train)
    model = build_model_from_config(config)
    likelihood = Categorical(10)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    return environment, learners, build_evalsets(config, environment, test), likelihood, theta0


def recorder_for(config, out_dir, seed: int = 0) -> rec.Recorder:
    environment = build_environment(config, seed, split())
    context = RunContext.from_config(
        config, seed, environment.graph, run_id="testrun", git_sha="deadbeef"
    )
    return rec.Recorder(out_dir, context, config)


def a_row(**overrides) -> dict:
    # `learner` is required but is *not* a run constant -- several learners
    # share one run -- so the runner stamps it per row rather than RunContext.
    base = {
        "t": 0,
        "node_id": 0,
        "metric": "nll",
        "value": 1.5,
        "evalset": "prequential",
        "learner": "diffusion_sgd_atc",
    }
    base.update(overrides)
    return base


# =========================================================================== #
# 1. the column contract
# =========================================================================== #


def test_every_row_carries_the_run_constants(tmp_path) -> None:
    """Stamped centrally, so a caller that forgets one cannot produce rows that
    look fine and refuse to group."""
    recorder = recorder_for(config_for(), tmp_path)
    recorder.log(a_row())
    stamped = recorder._buffer[0]
    for field in ("run_id", "git_sha", "experiment", "seed", "topology", "n_nodes"):
        assert stamped[field] is not None


def test_a_missing_required_field_is_rejected() -> None:
    with pytest.raises(SchemaError, match="missing required field"):
        validate({"t": 0, "metric": "nll"})


def test_an_unknown_column_is_rejected(tmp_path) -> None:
    """As the config loader treats unknown keys: a typo that becomes a new
    column makes every groupby silently incomplete."""
    recorder = recorder_for(config_for(), tmp_path)
    with pytest.raises(SchemaError, match="unknown column"):
        recorder.log(a_row(erorr_rate=0.5))


def test_a_counted_metric_must_carry_its_counts(tmp_path) -> None:
    recorder = recorder_for(config_for(), tmp_path)
    with pytest.raises(SchemaError, match="must carry n_correct and n_samples"):
        recorder.log(a_row(metric="error_rate", value=0.5))


def test_counts_must_be_consistent(tmp_path) -> None:
    recorder = recorder_for(config_for(), tmp_path)
    with pytest.raises(SchemaError, match=r"outside \[0, 2\]"):
        recorder.log(a_row(metric="error_rate", value=0.5, n_correct=5, n_samples=2))


def test_aggregate_node_names_are_allowed(tmp_path) -> None:
    """E_agree is a network quantity, so its node_id is 'mean' rather than an
    integer -- which is why the column is a string."""
    recorder = recorder_for(config_for(), tmp_path)
    recorder.log(a_row(node_id="mean", metric="e_agree", value=0.01))
    assert recorder.n_rows == 1


def test_a_nonsense_node_id_is_rejected(tmp_path) -> None:
    recorder = recorder_for(config_for(), tmp_path)
    with pytest.raises(SchemaError, match="neither an integer nor"):
        recorder.log(a_row(node_id="agent-seven"))


def test_the_empty_frame_still_has_every_column() -> None:
    """So concatenating a run that produced nothing does not drop columns."""
    assert list(empty_frame().columns) == list(COLUMNS)


def test_required_columns_are_a_subset_of_the_schema() -> None:
    assert set(REQUIRED) <= set(COLUMNS)


# =========================================================================== #
# 2. counts aggregate exactly where rates would not
# =========================================================================== #


def test_averaging_rates_disagrees_with_summing_counts() -> None:
    """The reason counts are stored. Two agents, unequal batch sizes: the mean
    of the rates is not the pooled rate."""
    rows = [
        {"n_correct": 1, "n_samples": 2},  # 50% of 2
        {"n_correct": 8, "n_samples": 8},  # 100% of 8
    ]
    mean_of_rates = sum(r["n_correct"] / r["n_samples"] for r in rows) / len(rows)
    pooled = sum(r["n_correct"] for r in rows) / sum(r["n_samples"] for r in rows)
    assert mean_of_rates == pytest.approx(0.75)
    assert pooled == pytest.approx(0.9)
    assert mean_of_rates != pytest.approx(pooled)


def test_n_correct_means_correct_not_wrong() -> None:
    """A field named n_correct holding the complement would make the natural
    expression return accuracy while the metric column said error."""
    from dekf_bench.evaluation.protocol import NodeScore

    score = NodeScore(
        node=0, evalset="prequential", step=0, n_samples=4, n_correct=3, rotation_degrees=0.0
    )
    row = next(r for r in score.as_rows() if r["metric"] == "error_rate")
    assert row["n_correct"] == 3
    assert row["value"] == pytest.approx(0.25)
    assert 1 - row["n_correct"] / row["n_samples"] == pytest.approx(row["value"])


# =========================================================================== #
# 3. writing and flushing
# =========================================================================== #


def test_flushing_writes_a_readable_file(tmp_path) -> None:
    recorder = recorder_for(config_for(), tmp_path)
    recorder.log_many([a_row(t=step) for step in range(5)])
    recorder.flush(step=4)

    frame = rec.read_results(recorder.data_path)
    assert len(frame) == 5
    assert list(frame.columns) == list(COLUMNS)


def test_a_partial_file_is_readable(tmp_path) -> None:
    """The point of flushing on evaluation steps: an interrupted run leaves
    something usable rather than nothing."""
    recorder = recorder_for(config_for(), tmp_path)
    recorder.log_many([a_row(t=step) for step in range(3)])
    recorder.flush(step=2)
    recorder.log_many([a_row(t=step) for step in range(3, 6)])
    # no second flush -- simulating a crash
    assert len(rec.read_results(recorder.data_path)) == 3


def test_writes_are_atomic(tmp_path) -> None:
    recorder = recorder_for(config_for(), tmp_path)
    recorder.log(a_row())
    recorder.flush(step=0)
    assert not list(tmp_path.glob("*.tmp"))


def test_reading_a_directory_concatenates_seeds(tmp_path) -> None:
    config = config_for()
    for seed in (0, 1):
        recorder = recorder_for(config, tmp_path, seed=seed)
        recorder.log_many([a_row(t=step) for step in range(4)])
        recorder.finalize()
    frame = rec.read_results(tmp_path)
    assert len(frame) == 8
    assert set(frame["seed"]) == {0, 1}


def test_reading_an_empty_directory_says_so(tmp_path) -> None:
    with pytest.raises(rec.RecorderError, match="no results"):
        rec.read_results(tmp_path / "nothing")


# =========================================================================== #
# 4. resumption
# =========================================================================== #


def test_a_fresh_run_resumes_from_zero(tmp_path) -> None:
    config = config_for()
    _, learners, _, _, theta0 = pieces(config)
    for learner in learners.values():
        learner.init(theta0)
    assert recorder_for(config, tmp_path).resume(learners) == 0


def test_a_checkpoint_records_where_it_stopped(tmp_path) -> None:
    config = config_for()
    _, learners, _, _, theta0 = pieces(config)
    for learner in learners.values():
        learner.init(theta0)

    recorder = recorder_for(config, tmp_path)
    recorder.log(a_row())
    recorder.flush(step=19, learners=learners)

    checkpoint = recorder.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint.step == 19
    assert checkpoint.resume_from == 20


def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path) -> None:
    """The property resumption is worth having only if it has.

    Exact rather than approximate because the loop consumes no randomness: the
    environment is positional and every draw happened at construction. If that
    changes, this test fails rather than the results acquiring a silent
    discontinuity at the resume point.
    """
    config = config_for()

    # Uninterrupted.
    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    whole = recorder_for(config, tmp_path / "whole")
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, recorder=whole)
    whole.finalize()
    reference = {
        node: learners["diffusion_sgd_atc"].flat_params(node).clone() for node in range(10)
    }

    # Interrupt at the second evaluation. The *same* config -- shortening the
    # horizon would change alpha and therefore the data, which is why the
    # fingerprint covers it.
    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    partial = recorder_for(config, tmp_path / "resumed")
    simulate.run(
        config,
        environment,
        learners,
        evalsets,
        likelihood,
        theta0,
        recorder=partial,
        stop_after=2 * EVAL_EVERY - 1,
    )
    partial.finalize()

    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    resumed = recorder_for(config, tmp_path / "resumed")
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, recorder=resumed)
    resumed.finalize()

    for node in range(10):
        assert torch.equal(
            learners["diffusion_sgd_atc"].flat_params(node), reference[node]
        ), f"agent {node} diverged across the resume boundary"


def test_a_resumed_run_does_not_duplicate_rows(tmp_path) -> None:
    config = config_for()

    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    partial = recorder_for(config, tmp_path)
    simulate.run(
        config,
        environment,
        learners,
        evalsets,
        likelihood,
        theta0,
        recorder=partial,
        stop_after=2 * EVAL_EVERY - 1,
    )
    partial.finalize()

    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    resumed = recorder_for(config, tmp_path)
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, recorder=resumed)
    resumed.finalize()

    frame = rec.read_results(resumed.data_path)
    keys = ["t", "learner", "node_id", "metric", "evalset"]
    assert not frame.duplicated(subset=keys).any()
    assert frame["t"].max() == HORIZON - 1


def test_resuming_a_different_config_is_refused(tmp_path) -> None:
    """Splicing two experiments into one file would produce results that look
    like a single coherent run."""
    config = config_for()
    _, learners, _, _, theta0 = pieces(config)
    for learner in learners.values():
        learner.init(theta0)

    recorder = recorder_for(config, tmp_path)
    recorder.log(a_row())
    recorder.flush(step=0, learners=learners)

    other = recorder_for(config_for(env={"label_availability": 0.5}), tmp_path)
    with pytest.raises(rec.RecorderError, match="different\\s+configuration"):
        other.load_checkpoint()


def test_a_damaged_checkpoint_starts_fresh_rather_than_raising(tmp_path) -> None:
    config = config_for()
    _, learners, _, _, theta0 = pieces(config)
    for learner in learners.values():
        learner.init(theta0)
    recorder = recorder_for(config, tmp_path)
    recorder.checkpoint_path.write_bytes(b"not a checkpoint")
    assert recorder.load_checkpoint() is None


def test_the_config_fingerprint_is_stable() -> None:
    assert rec.config_fingerprint(config_for()) == rec.config_fingerprint(config_for())


def test_the_fingerprint_changes_with_the_config() -> None:
    assert rec.config_fingerprint(config_for()) != rec.config_fingerprint(
        config_for(env={"label_availability": 0.5})
    )


# =========================================================================== #
# 5. run-level artefacts
# =========================================================================== #


def test_metadata_records_the_config_and_the_environment(tmp_path) -> None:
    import json

    config = config_for()
    path = rec.write_metadata(tmp_path, config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config_hash"] == rec.config_fingerprint(config)
    assert "torch_version" in payload["environment"]
    assert (tmp_path / "config.yaml").is_file()


def test_the_ledger_is_written_once_per_experiment(tmp_path) -> None:
    import json

    rows = [{"learner": "atc", "scalars_per_step": 58160, "note": "psi only"}]
    path = rec.write_ledger(tmp_path, rows)
    assert json.loads(path.read_text(encoding="utf-8"))[0]["scalars_per_step"] == 58160


def test_run_ids_are_unique() -> None:
    assert len({rec.new_run_id() for _ in range(100)}) == 100


# =========================================================================== #
# 6. what the runner actually writes
# =========================================================================== #


def test_a_run_records_every_learner_and_metric(tmp_path) -> None:
    config = config_for()
    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    recorder = recorder_for(config, tmp_path)
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, recorder=recorder)
    recorder.finalize()

    frame = rec.read_results(recorder.data_path)
    assert set(frame["learner"]) == set(learners)
    assert {"prequential", "current", "canonical"} <= set(frame["evalset"].dropna())
    assert {"error_rate", "nll", "e_agree"} <= set(frame["metric"])


def test_cumulative_communication_grows_with_the_step(tmp_path) -> None:
    """So F2 can plot error against it without joining the ledger."""
    config = config_for()
    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    recorder = recorder_for(config, tmp_path)
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, recorder=recorder)
    recorder.finalize()

    frame = rec.read_results(recorder.data_path)
    atc = frame[frame["learner"] == "diffusion_sgd_atc"].sort_values("t")
    assert atc["cum_scalars_tx"].is_monotonic_increasing
    assert int(atc["cum_scalars_tx"].iloc[-1]) > 0


def test_the_non_diffusing_learners_transmit_nothing(tmp_path) -> None:
    config = config_for()
    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    recorder = recorder_for(config, tmp_path)
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, recorder=recorder)
    recorder.finalize()

    frame = rec.read_results(recorder.data_path)
    for name in ("local_only", "centralized_sgd"):
        assert (frame[frame["learner"] == name]["cum_scalars_tx"] == 0).all()


def test_the_plain_atc_variant_sends_half_of_what_momentum_atc_sends(tmp_path) -> None:
    """The pairing the phase-5 claim rests on (design note D29)."""
    config = config_for()
    environment, learners, evalsets, likelihood, theta0 = pieces(config)
    recorder = recorder_for(config, tmp_path)
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, recorder=recorder)
    recorder.finalize()

    frame = rec.read_results(recorder.data_path)
    last = frame["t"].max()
    at_end = frame[frame["t"] == last]
    momentum = at_end[at_end["learner"] == "diffusion_sgd_atc"]["cum_scalars_tx"].iloc[0]
    plain = at_end[at_end["learner"] == "diffusion_sgd_atc_plain"]["cum_scalars_tx"].iloc[0]
    assert int(momentum) == 2 * int(plain)


# =========================================================================== #
# atomic replace under a transient lock
# =========================================================================== #


def test_a_transient_lock_on_the_target_is_retried(tmp_path) -> None:
    """``os.replace`` is atomic on POSIX but on Windows fails outright with
    PermissionError when the target is held open by *any* process -- a scanner,
    an indexer, a cloud-sync client, or a handle not yet reaped from a killed
    run.

    This crashed a real experiment mid-run: the write was complete and correct,
    the rename simply could not land at that instant, and the whole run was lost
    to a condition that clears in milliseconds.
    """
    from dekf_bench.utils import atomic as module

    staging = tmp_path / "payload.tmp"
    target = tmp_path / "payload.pt"
    staging.write_text("done", encoding="utf-8")

    attempts = {"n": 0}
    real_replace = type(staging).replace

    def flaky(self, destination):
        attempts["n"] += 1
        if attempts["n"] < 3:  # clears on the third try
            raise PermissionError(5, "Access is denied")
        return real_replace(self, destination)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(type(staging), "replace", flaky)
    monkey.setattr(module, "BACKOFF_S", 0.001)
    try:
        module.replace_with_retry(staging, target)
    finally:
        monkey.undo()

    assert attempts["n"] == 3
    assert target.read_text(encoding="utf-8") == "done"
    assert not staging.exists()


def test_a_permanent_permission_error_still_raises(tmp_path) -> None:
    """The retry is narrow on purpose. A read-only directory raises the same
    class, so it must not be retried forever and then swallowed -- after the
    last attempt the original error propagates."""
    from dekf_bench.utils import atomic as module

    staging = tmp_path / "payload.tmp"
    staging.write_text("done", encoding="utf-8")

    def always_denied(self, destination):
        raise PermissionError(5, "Access is denied")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(type(staging), "replace", always_denied)
    monkey.setattr(module, "BACKOFF_S", 0.001)
    try:
        with pytest.raises(PermissionError):
            module.replace_with_retry(staging, tmp_path / "payload.pt")
    finally:
        monkey.undo()
