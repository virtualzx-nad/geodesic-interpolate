"""Bounded midpoint timing and independent correctness checks for issue #17.

Example::

    .venv/bin/python benchmarks/validate_midpoint.py --cases methane diels --output /tmp/midpoints.json
    .venv/bin/python benchmarks/validate_midpoint.py --source /tmp/baseline --cases methane diels

Each sample uses a fresh, single-threaded process and one untimed warm-up.
The raw first/last endpoints are aligned once, then both endpoint-near starts
use the same random seed. The independent gradients describe the actual
least-squares solutions before candidate alignment and scoring. Timings are
not matched-convergence speed comparisons unless both methods reach the same
gradient tolerance on the same pair lists. Peak RSS includes warm-up/runtime.
"""
import argparse
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from validate_smoothing import independent_clearance


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "methane": "H+CH4_CH3+H2.xyz",
    "diels": "DielsAlder.xyz",
    "trp": "TrpCage_unfold.xyz",
    "collagen": "collagen.xyz",
    "calcium": "calcium_binding.xyz",
}


def _digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def independent_midpoint_values(x, pairs, scaler, reference, x0, friction):
    """Differentiate the midpoint least-squares objective by atom accumulation."""
    geometry = np.asarray(x).reshape(-1, 3)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    delta = geometry[pairs[:, 0]] - geometry[pairs[:, 1]]
    distance = np.sqrt(np.sum(delta * delta, axis=1))
    values, derivative = scaler(distance)
    residual = values - reference
    contribution = delta * (residual * derivative / distance)[:, None]
    gradient = np.zeros_like(geometry)
    np.add.at(gradient, pairs[:, 0], contribution)
    np.add.at(gradient, pairs[:, 1], -contribution)
    motion = (x - x0) * friction
    gradient = gradient.ravel() + friction * motion
    return dict(gradient_inf=float(np.max(np.abs(gradient), initial=0)),
                objective=float(0.5 * (residual @ residual + motion @ motion)))


def _worker(args):
    sys.path.insert(0, str(Path(args.source).resolve()))
    import scipy
    from geodesic_interpolate.coord_utils import align_path, get_bond_list, ATOMIC_RADIUS
    from geodesic_interpolate.fileio import read_xyz
    import geodesic_interpolate.interpolation as interpolation

    logging.disable(logging.CRITICAL)
    atoms, frames = read_xyz(ROOT / "test_cases" / CASES[args.cases[0]])
    prepared = align_path(np.asarray([frames[0], frames[-1]]))[1]
    original = prepared.copy()
    scipy_solve = interpolation.least_squares
    solved_problems = []

    def traced_solve(fun, x0, jac, **kwargs):
        # Read the two implementations' problem descriptions, without using
        # their residual or Jacobian to validate the returned solution.
        if inspect.ismethod(fun):
            objective = fun.__self__
            pairs = objective.coordinates.pairs
            scaler = objective.scaler
            reference = objective.reference
            friction = objective.friction
        else:
            scope = dict(zip(fun.__code__.co_freevars, (c.cell_contents for c in fun.__closure__)))
            pairs, scaler, reference, friction = (scope[k] for k in ("rijlist", "scaler", "w", "friction"))
        result = scipy_solve(fun, x0, jac, **kwargs)
        solved_problems.append((result.x.copy(), int(result.nfev), pairs, scaler, reference, x0, friction))
        return result

    interpolation.least_squares = traced_solve

    def solve():
        np.random.seed(args.seed)
        return interpolation.mid_point(atoms, prepared[0], prepared[-1], tol=args.tol)

    solve()
    solved_problems.clear()
    begin = time.perf_counter()
    midpoint = solve()
    elapsed = time.perf_counter() - begin
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    trials = []
    for solution, nfev, pairs, scaler, reference, x0, friction in solved_problems:
        record = independent_midpoint_values(solution, pairs, scaler, reference, x0, friction)
        record.update(pair_digest=_digest(np.asarray(pairs, dtype=np.int64)),
                      pairs=len(pairs), nfev=nfev, start_digest=_digest(x0),
                      reached_tolerance=record["gradient_inf"] <= args.tol)
        trials.append(record)
    path = np.asarray([prepared[0], midpoint, prepared[-1]])
    pairs, _ = get_bond_list(path, atoms, threshold=4)
    result = dict(case=args.cases[0], atoms=len(atoms), seed=args.seed, tol=args.tol,
                  time_s=elapsed, peak_rss_mib=rss_bytes / 1024 ** 2,
                  trials=trials, endpoint_digest=_digest(prepared),
                  endpoint_error=float(np.max(np.abs(prepared - original))),
                  finite=bool(np.isfinite(path).all()),
                  python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__)
    result.update(independent_clearance(path, pairs, atoms, ATOMIC_RADIUS))
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["methane", "diels"])
    parser.add_argument("--source", default=str(ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60,
                        help="Maximum seconds per sample, including warm-up")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    results = []
    for case in args.cases:
        samples = []
        for repeat in range(args.repeats):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--cases", case]
            for name in ("source", "seed", "tol"):
                command.extend(("--" + name, str(getattr(args, name))))
            try:
                completed = subprocess.run(command, text=True, capture_output=True, check=True,
                                           timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print("{} exceeded {:.0f}s per sample including warm-up".format(case, args.timeout), flush=True)
                break
            sample = json.loads(completed.stdout)
            samples.append(sample)
            print("{} sample {}/{}: {:.3f}s, peak {:.1f} MiB, trial gradients {}".format(
                case, repeat + 1, args.repeats, sample["time_s"], sample["peak_rss_mib"],
                [round(t["gradient_inf"], 6) for t in sample["trials"]]), flush=True)
        if samples:
            result = samples[0].copy()
            result["time_s"] = statistics.median(s["time_s"] for s in samples)
            result["peak_rss_mib"] = statistics.median(s["peak_rss_mib"] for s in samples)
            result["samples"] = samples
        else:
            result = dict(case=case, timeout_s=args.timeout, samples=[])
        results.append(result)
        if args.output:
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
