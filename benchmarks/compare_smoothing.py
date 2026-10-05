"""Matched full-path smoothing with baseline/current evaluation implementations.

Example::

    python benchmarks/compare_smoothing.py --baseline /tmp/baseline \
        --cases methane trp --output /tmp/matched-smoothing.json

This is a controlled comparison, NOT the behavior of unmodified master.
Both implementations use the same driver, prepared endpoints, selected pairs,
Morse scaler, fixed friction reference, trial guard, callback copies, and SciPy
soft-L1 stopping protocol. Only coordinate/residual/Jacobian evaluation and
assembly come from the selected checkout. Every timed solve must independently
reach the common gradient tolerance while preserving endpoints and safety.
"""
import argparse
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_name] = "1"

import numpy as np

from validate_smoothing import (
    CASES, ROOT, _digest, assert_source_path, checked_source_directory,
    independent_clearance, independent_values, reference_module,
)


class ControlledSolve:
    """Common optimizer protocol around each checkout's evaluation kernels."""

    def __init__(self, geodesic, reference, checker, unsafe_error):
        self.geodesic = geodesic
        self.reference = reference.copy()
        self.checker = checker
        self.unsafe_error = unsafe_error
        self.last_x = None
        self.evaluations = 0
        self.rejections = 0

    def evaluate(self, x):
        g = self.geodesic
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return True
        candidate = g.path.copy()
        candidate[1:-1] = np.asarray(x).reshape(candidate[1:-1].shape)
        try:
            self.checker.validate(candidate)
            pairs = np.asarray(g.rij_list, dtype=int)
            for geometries in (candidate, (candidate[:-1] + candidate[1:]) * 0.5):
                delta = geometries[:, pairs[:, 0]] - geometries[:, pairs[:, 1]]
                if np.any(np.all(delta == 0, axis=-1)):
                    raise self.unsafe_error("Coincident selected atoms")
        except self.unsafe_error:
            self.rejections += 1
            return False
        if not np.array_equal(candidate, g.path):
            g.path[:] = candidate
            # Full-path variables include every interior image. Endpoint
            # coordinate caches remain valid; all midpoint caches change.
            g.w[1:-1] = [None] * (g.nimages - 2)
            g.w_mid[:] = [None] * (g.nimages - 1)
        g.compute_disps(1, g.nimages - 1, dx=x - self.reference, friction=g.friction)
        g.compute_disp_grad(1, g.nimages - 1, friction=g.friction)
        self.last_x = np.asarray(x).copy()
        weighted = g.disps / np.hypot(1.0, g.disps)
        self.optimality = float(np.max(np.abs(g.grad.T @ weighted), initial=0))
        self.evaluations += 1
        return True

    def residual(self, x):
        if self.evaluate(x):
            return self.geodesic.disps.copy()
        return np.full(self.geodesic.disps.shape, 1e20)

    def jacobian(self, x):
        from scipy import sparse

        g = self.geodesic
        if self.evaluate(x):
            return g.grad.toarray() if x.size <= 100 else g.grad.copy()
        return np.zeros(g.grad.shape) if x.size <= 100 else sparse.csr_matrix(g.grad.shape)

    def solve(self, tol, max_iter):
        from scipy.optimize import least_squares

        initial = self.geodesic.path[1:-1].ravel().copy()
        if not self.evaluate(initial):
            raise AssertionError("Unsafe prepared initial path")
        if self.optimality <= tol:
            return dict(nfev=0, njev=0, status=1)
        result = least_squares(
            self.residual, initial, jac=self.jacobian, loss="soft_l1",
            ftol=np.finfo(float).eps, xtol=np.finfo(float).eps, gtol=tol,
            max_nfev=max_iter)
        if not self.evaluate(result.x):
            raise AssertionError("Solver returned an unsafe accepted result")
        return dict(nfev=result.nfev, njev=result.njev, status=result.status)


def _worker(args):
    import resource

    source = checked_source_directory(args.source)
    sys.path.insert(0, str(source))
    import scipy
    from geodesic_interpolate.geodesic import Geodesic
    assert_source_path(Geodesic, source, "geodesic.py")

    coordinates = reference_module("coord_utils")
    fileio = reference_module("fileio")
    validation = reference_module("validation")
    logging.disable(logging.CRITICAL)
    atoms, frames = fileio.read_xyz(ROOT / "test_cases" / CASES[args.cases[0]])
    selected = np.asarray(frames)[np.linspace(0, len(frames) - 1, args.images, dtype=int)]
    _, prepared = coordinates.align_path(selected)
    np.random.seed(args.seed)
    pairs, re = coordinates.get_bond_list(prepared, atoms, threshold=3.0)
    reference = prepared[1:-1].ravel().copy()
    scaler = coordinates.morse_scaler(re=re, alpha=1.7)

    def setup():
        np.random.seed(args.seed)
        kwargs = dict(friction=args.friction, scaler=scaler)
        if "align" in inspect.signature(Geodesic).parameters:
            kwargs["align"] = False
        g = Geodesic(atoms, prepared.copy(), **kwargs)
        if not np.array_equal(g.rij_list, pairs):
            raise AssertionError("Source selected different pairs from the shared preparation")
        # The old constructor realigns. Reset it before any evaluation, so
        # both native kernels see the exact same Cartesian preparation.
        g.path = prepared.copy()
        if any(v is not None for v in g.w + g.w_mid):
            raise AssertionError("Constructor unexpectedly evaluated coordinate caches")
        checker = validation.OverlapChecker(len(atoms), pairs, atoms)
        return ControlledSolve(g, reference, checker, validation.UnsafePathError)

    warmup = setup()
    warmup.solve(args.tol, args.max_iter)
    del warmup
    driver = setup()
    started = time.perf_counter()
    solver = driver.solve(args.tol, args.max_iter)
    elapsed = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    g = driver.geodesic
    checked = independent_values(g.path, pairs, scaler, reference, args.friction)
    clearance = independent_clearance(g.path, pairs, atoms, coordinates.COVALENT_RADIUS)
    endpoint_error = float(np.max(np.abs(g.path[[0, -1]] - prepared[[0, -1]])))
    length_error = abs(checked["length"] - g.length)
    gradient_error = abs(checked["optimality"] - driver.optimality)
    valid = (checked["optimality"] <= args.tol and endpoint_error == 0 and
             length_error < 1e-10 and gradient_error < 1e-10 and
             clearance["overlap_violations"] == 0)
    source_hash = hashlib.sha256()
    for filename in sorted((source / "geodesic_interpolate").glob("*.py")):
        source_hash.update(filename.name.encode())
        source_hash.update(filename.read_bytes())
    result = dict(
        case=args.cases[0], implementation=args.implementation,
        protocol="controlled full-path soft-L1 solve; shared geometry/cache/guard driver",
        images=args.images, atoms=len(atoms), pairs=len(pairs), seed=args.seed,
        tol=args.tol, friction=args.friction, max_nfev=args.max_iter,
        time_s=elapsed, peak_rss_mib=rss_bytes / 1024 ** 2,
        objective=checked["objective"], length=checked["length"],
        gradient_inf=checked["optimality"], endpoint_error=endpoint_error,
        length_error=length_error, gradient_error=gradient_error,
        passed=valid, target_evaluations=driver.evaluations,
        rejected_trials=driver.rejections, solver=solver,
        source_digest=source_hash.hexdigest(), input_digest=_digest(prepared),
        endpoint_digest=_digest(prepared[[0, -1]]), pair_digest=_digest(np.asarray(pairs, dtype=np.int64)),
        reference_digest=_digest(reference), metric_radii_digest=_digest(re),
        safety_radii_digest=hashlib.sha256(json.dumps(
            coordinates.COVALENT_RADIUS, sort_keys=True).encode()).hexdigest(),
        python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
        **clearance)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=["methane", "trp"])
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=0.002)
    parser.add_argument("--friction", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", default="current", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    if args.baseline is None:
        parser.error("--baseline must name a checkout of the baseline implementation")
    results = []
    for case in args.cases:
        for implementation, source in (("baseline-adapter", args.baseline), ("current", args.source)):
            samples = []
            for repeat in range(args.repeats):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--implementation", implementation, "--source", str(source),
                           "--cases", case]
                for name in ("images", "seed", "tol", "friction", "max_iter"):
                    command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
                completed = subprocess.run(command, text=True, capture_output=True, check=True)
                sample = json.loads(completed.stdout)
                samples.append(sample)
                print("{} {} {}/{}: {:.4f}s, |g|={:.8g}, passed={}".format(
                    case, implementation, repeat + 1, args.repeats,
                    sample["time_s"], sample["gradient_inf"], sample["passed"]), flush=True)
            summary = samples[0].copy()
            summary["time_s"] = statistics.median(s["time_s"] for s in samples)
            summary["peak_rss_mib"] = statistics.median(s["peak_rss_mib"] for s in samples)
            summary["samples"] = samples
            results.append(summary)
            if args.output:
                args.output.write_text(json.dumps(results, indent=2) + "\n")
    for case in args.cases:
        compared = [r for r in results if r["case"] == case]
        for key in ("input_digest", "endpoint_digest", "pair_digest", "reference_digest",
                    "metric_radii_digest", "safety_radii_digest"):
            if len({s[key] for r in compared for s in r["samples"]}) != 1:
                raise AssertionError("{} has mismatched {}".format(case, key))
        if not all(s["passed"] for r in compared for s in r["samples"]):
            raise AssertionError("{} did not meet the matching acceptance checks".format(case))
    print("\ncase implementation seconds peak_RSS_MiB independent_gradient")
    for r in results:
        print("{case} {implementation} {time_s:.4f} {peak_rss_mib:.1f} {gradient_inf:.8g}".format(**r))


if __name__ == "__main__":
    main()
