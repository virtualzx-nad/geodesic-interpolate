"""Reproducible, independently checked smoothing benchmarks for issue #17.

Run from the checkout, for example::

    .venv/bin/python benchmarks/validate_smoothing.py --cases methane --output /tmp/methane.json

Each sample runs in a fresh, single-threaded process, performs one identical
untimed warm-up, then measures a solve. Peak RSS includes the warm-up and Python
runtime. Timings from methods which do not reach the common gradient tolerance
must not be presented as a convergence speed comparison. ``--source`` can point
to another checkout to measure it with the same independent evaluator.
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

# Set before importing NumPy/SciPy, including when this file is used as a worker.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "methane": "H+CH4_CH3+H2_interpolated.xyz",
    "trp": "TrpCage_interpolated.xyz",
    "collagen": "collagen_interpolated.xyz",
    "calcium": "calcium_binding_interpolated.xyz",
}


def independent_values(path, pairs, scaler, xref=None, friction=0.0,
                       start=1, end=-1):
    """Calculate the soft-L1 objective, gradient and length without optimizer caches.

    Cartesian derivatives are accumulated directly by atom index; this does not
    call the production distance, Jacobian, residual, or length implementations.
    The half-segment residual order is the public target function's order.
    """
    path = np.asarray(path)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if end < 0:
        end += len(path)
    pieces = []
    derivatives = []
    for geometries in (path, (path[:-1] + path[1:]) * 0.5):
        delta = geometries[:, pairs[:, 0]] - geometries[:, pairs[:, 1]]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        values, derivative = scaler(distance)
        pieces.append(values)
        derivatives.append(delta * (derivative / distance)[..., None])
    values, mid_values = pieces
    atom_derivatives, mid_derivatives = derivatives
    left = mid_values[start - 1:end] - values[start - 1:end]
    right = values[start:end + 1] - mid_values[start - 1:end]
    weighted_left = left / np.hypot(1.0, left)
    weighted_right = right / np.hypot(1.0, right)
    gradient = np.zeros_like(path)

    def add(image, derivative, weights):
        contribution = derivative * weights[:, None]
        np.add.at(gradient[image], pairs[:, 0], contribution)
        np.add.at(gradient[image], pairs[:, 1], -contribution)

    for local, segment in enumerate(range(start - 1, end)):
        mid_weight = (weighted_left[local] - weighted_right[local]) * 0.5
        add(segment, atom_derivatives[segment], -weighted_left[local])
        add(segment, mid_derivatives[segment], mid_weight)
        add(segment + 1, mid_derivatives[segment], mid_weight)
        add(segment + 1, atom_derivatives[segment + 1], weighted_right[local])
    dx = (np.zeros(path[start:end].size) if xref is None else
          path[start:end].ravel() - np.asarray(xref).ravel())
    regularizer = friction * dx
    gradient = gradient[start:end].ravel()
    gradient += friction * regularizer / np.hypot(1.0, regularizer)
    residual = np.concatenate((left.ravel(), right.ravel(), regularizer))
    return {
        "residual": residual,
        "objective": float(np.sum(np.hypot(1.0, residual) - 1.0)),
        "gradient": gradient,
        "optimality": float(np.max(np.abs(gradient), initial=0.0)),
        "length": float(np.sum(np.linalg.norm(left, axis=1)) +
                        np.sum(np.linalg.norm(right, axis=1))),
    }


def _digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def independent_clearance(path, pairs, atoms, radii):
    """Brute-force omitted-pair screening, independent of the production tree."""
    atom_i, atom_j = np.triu_indices(len(atoms), 1)
    included = set(map(tuple, pairs))
    omitted = np.array([(i, j) not in included for i, j in zip(atom_i, atom_j)])
    radius = np.array([radii.get(atom.capitalize(), 1.5) for atom in atoms])
    minimum = np.maximum(0.70, 0.60 * (radius[atom_i] + radius[atom_j]))
    closest = []
    clearance = float("inf")
    violations = 0
    for geometries in (path, (path[:-1] + path[1:]) * 0.5):
        closest_group = float("inf")
        for geometry in geometries:
            delta = geometry[atom_i] - geometry[atom_j]
            distance = np.sqrt(np.sum(delta * delta, axis=1))
            closest_group = min(closest_group, float(np.min(distance, initial=np.inf)))
            margin = (distance - minimum)[omitted]
            clearance = min(clearance, float(np.min(margin, initial=np.inf)))
            violations += int(np.count_nonzero(margin < 0))
        closest.append(closest_group)
    return {
        "closest_output_pair_angstrom": closest[0],
        "closest_midpoint_pair_angstrom": closest[1],
        "omitted_pair_clearance_angstrom": clearance if np.isfinite(clearance) else None,
        "overlap_violations": violations,
    }


def _worker(args):
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    source_hash = hashlib.sha256()
    for filename in sorted((source / "geodesic_interpolate").glob("*.py")):
        source_hash.update(filename.name.encode())
        source_hash.update(filename.read_bytes())
    import scipy
    from geodesic_interpolate.coord_utils import align_path, ATOMIC_RADIUS
    from geodesic_interpolate.fileio import read_xyz
    from geodesic_interpolate.geodesic import Geodesic

    logging.disable(logging.CRITICAL)
    atoms, frames = read_xyz(ROOT / "test_cases" / CASES[args.cases[0]])
    indices = np.linspace(0, len(frames) - 1, args.images, dtype=int)
    _, prepared = align_path(np.asarray(frames)[indices])
    fixed_endpoints = prepared[[0, -1]].copy()
    reference = prepared[1:-1].ravel().copy()

    def setup():
        np.random.seed(args.seed)
        kwargs = {"friction": args.friction}
        if "align" in inspect.signature(Geodesic).parameters:
            kwargs["align"] = False
        return Geodesic(atoms, prepared.copy(), **kwargs)

    method = args.methods[0]

    def solve(geodesic):
        if method == "smooth":
            geodesic.smooth(tol=args.tol, max_iter=args.max_iter)
        else:
            geodesic.sweep(tol=args.tol, max_iter=args.sweeps,
                           micro_iter=args.micro_iter)

    warmup = setup()
    solve(warmup)
    del warmup
    geodesic = setup()
    initial = independent_values(geodesic.path, geodesic.rij_list,
                                 geodesic.scaler, reference, args.friction)
    t0 = time.perf_counter()
    solve(geodesic)
    elapsed = time.perf_counter() - t0
    # Capture solver peak before the independent final checks allocate arrays.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    final = independent_values(geodesic.path, geodesic.rij_list,
                               geodesic.scaler, reference, args.friction)
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    result = {
        "case": args.cases[0], "method": method, "atoms": len(atoms),
        "images": args.images, "pairs": len(geodesic.rij_list),
        "seed": args.seed, "tol": args.tol, "friction": args.friction,
        "time_s": elapsed, "peak_rss_mib": rss_bytes / 1024 ** 2,
        "initial_length": initial["length"], "length": final["length"],
        "initial_objective": initial["objective"], "objective": final["objective"],
        "gradient_inf": final["optimality"],
        "reached_tolerance": final["optimality"] <= args.tol,
        "reported_gradient_inf": float(geodesic.optimality),
        "length_error": abs(final["length"] - geodesic.length),
        "endpoint_error": float(np.max(np.abs(geodesic.path[[0, -1]] - fixed_endpoints))),
        "pair_digest": _digest(np.asarray(geodesic.rij_list, dtype=np.int64)),
        "input_digest": _digest(prepared), "endpoint_digest": _digest(fixed_endpoints),
        "target_evaluations": geodesic.neval,
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "source_digest": source_hash.hexdigest(),
    }
    result.update(independent_clearance(geodesic.path, geodesic.rij_list, atoms, ATOMIC_RADIUS))
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["methane"])
    parser.add_argument("--methods", nargs="+", choices=["smooth", "sweep"],
                        default=["smooth", "sweep"])
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=0.002)
    parser.add_argument("--friction", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--sweeps", type=int, default=35)
    parser.add_argument("--micro-iter", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source", default=str(ROOT))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    results = []
    for case in args.cases:
        for method in args.methods:
            samples = []
            for repeat in range(args.repeats):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--cases", case, "--methods", method]
                for name in ("source", "images", "seed", "tol", "friction",
                             "max_iter", "sweeps", "micro_iter"):
                    command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
                completed = subprocess.run(command, text=True, capture_output=True, check=True)
                sample = json.loads(completed.stdout)
                samples.append(sample)
                print("{} {} sample {}/{}: {:.3f}s, |g|={:.6g}, reached={}".format(
                    case, method, repeat + 1, args.repeats, sample["time_s"],
                    sample["gradient_inf"], sample["reached_tolerance"]), flush=True)
            result = samples[0].copy()
            result["time_s"] = statistics.median(s["time_s"] for s in samples)
            result["peak_rss_mib"] = statistics.median(s["peak_rss_mib"] for s in samples)
            result["samples"] = samples
            results.append(result)
            if args.output:
                args.output.write_text(json.dumps(results, indent=2) + "\n")
    for case in args.cases:
        compared = [r for r in results if r["case"] == case]
        for key in ("pair_digest", "input_digest", "endpoint_digest"):
            if len({r[key] for r in compared}) != 1:
                raise RuntimeError("{} did not use the same {}".format(case, key))
    print("\ncase method seconds peak_RSS_MiB gradient_inf reached endpoint_error length_error")
    for r in results:
        print("{case} {method} {time_s:.4f} {peak_rss_mib:.1f} {gradient_inf:.7g} "
              "{reached_tolerance} {endpoint_error:.3g} {length_error:.3g}".format(**r))


if __name__ == "__main__":
    main()
