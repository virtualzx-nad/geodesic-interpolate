# Midpoint comparison for issue #17

These measurements cover one midpoint interpolated between the raw reaction
endpoints, using seed 0, tolerance 0.01, and both endpoint-near initial guesses.
Each of three fresh single-threaded processes performed an identical untimed
warm-up followed by a timed run. The table shows medians; peak RSS includes
the Python runtime and warm-up. Python 3.12.14, NumPy 2.4.6, SciPy 1.17.1.
The baseline is commit `476aa041f562d37d5fb4121b86817a52d75e4c5d`.

| Case | Atoms | Baseline seconds | Updated seconds | Baseline peak MiB | Updated peak MiB | Independent gradient norms, both guesses |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Methane reaction | 6 | 0.003053 | 0.002050 | 74.86 | 75.00 | 0.00354255, 0.00772270 |
| Diels–Alder | 31 | 0.049482 | 0.030987 | 79.28 | 78.61 | 0.00659500, 0.00120446 |

For each case, both versions used identical prepared endpoint, pair-list, and
starting-guess hashes. Their independently calculated least-squares objective
values and gradients were exactly equal, and both guesses met the common
0.01 gradient tolerance. The independent evaluator accumulates derivatives by
atom index and does not use the production residual or Jacobian. Verification
runs after the timed section. These small cases use dense factorization in
both versions; this comparison does not measure large-system sparse speedups
or memory savings. Millisecond timings and small RSS differences should be
interpreted cautiously.

The returned midpoint and supplied endpoints were finite, the supplied
endpoints remained unchanged, and brute-force screening found no omitted-pair
overlaps at any of the three output images or their Cartesian midpoints.
The closest output-image distances were 0.74283 Å for methane and 1.07287 Å
for Diels–Alder. Diels–Alder's closest arithmetic-midpoint distance was
0.54358 Å for a **selected** pair, which is exempt from the omitted-pair guard.
This is initial-path interpolation, not a final smoothed-path or physical
safety guarantee. Both implementations produced this same minimum distance.

Reproduce from the repository root, with an unchanged baseline checkout:

```sh
.venv/bin/python benchmarks/validate_midpoint.py \
  --cases methane diels --repeats 3 --timeout 15 \
  --output /tmp/issue17-midpoint-current.json
.venv/bin/python benchmarks/validate_midpoint.py \
  --source /path/to/baseline --cases methane diels --repeats 3 --timeout 15 \
  --output /tmp/issue17-midpoint-baseline.json
```

The script also accepts `trp`, `collagen`, and `calcium`. Those cases were not
timed here; its per-sample timeout bounds expensive baseline dense solves.

Raw samples: [baseline](results/midpoint-baseline.json),
[updated](results/midpoint-current.json).
