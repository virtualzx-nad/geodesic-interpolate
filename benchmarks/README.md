# Smoothing validation for issue #17

`validate_smoothing.py` compares global smoothing and sweeping against the same
independently calculated soft-L1 gradient, including friction. It does not use
the optimizer's residual, Jacobian, coordinate derivative, or length caches to
calculate this gradient. Cartesian contributions are accumulated directly into
the atoms of each pair. Unit tests also check this calculation against central
finite differences of the scalar objective, with nonzero friction residuals.

## Reproduction

From the repository root, with NumPy and SciPy installed:

```sh
python benchmarks/validate_smoothing.py --cases methane --output /tmp/methane.json
python benchmarks/validate_smoothing.py --cases trp collagen calcium --output /tmp/proteins.json
python -m unittest discover -v
```

The files are the checked-in `test_cases/*_interpolated.xyz` paths. Each run
selects 10 images using `np.linspace(0, 16, 10, dtype=int)`, aligns this input once,
then fixes both prepared endpoints. The NumPy seed is 0; the pair cutoff is
3.0 Angstrom; friction is 0.001; the shared gradient threshold is 0.002. Global
smoothing gets at most 50 solver residual evaluations, and sweeping gets at most
35 sweeps of 20 evaluations per interior image. Both methods use the same
prepared initial path as their fixed friction reference. The output records
input, endpoint, and selected-pair SHA-256 digests and checks that they match
between methods.

Each reported timing is the median of three fresh processes. Each process first
performs one complete, identical, untimed warm-up solve. The numerical library
thread limits are set to one before importing NumPy/SciPy. Timed sections include
only smoothing; path loading, initial alignment, pair selection, warm-up, and
independent final validation are excluded. Peak resident memory is the median
of the process high-water marks immediately after the timed solve: it includes
Python, imports, input preparation, independent initial evaluation, and warm-up.
It is not an allocation delta or a claim that all observed memory is the solver.
The machine was not otherwise reserved exclusively for this benchmark.

## Matched full-path implementation comparison

The review correctly identified that the original master measurements below
do not satisfy the matching criterion: master changes the endpoints and does
not reach the common gradient tolerance. The following controlled comparison
addresses that gap for complete global smoothing solves. It is **an adapted
baseline**, not a claim about unmodified master's correctness or performance.

```sh
python benchmarks/compare_smoothing.py --baseline /path/to/baseline-checkout \
  --cases methane trp --output /tmp/matched-smoothing.json
```

The baseline checkout is commit `476aa041f562d37d5fb4121b86817a52d75e4c5d`.
`compare_smoothing.py` uses each checkout's native `update_intc`,
`compute_disps`, and `compute_disp_grad` implementations inside an identical
whole-path solver driver. Both start from exactly the same prepared Cartesian
path, selected pairs, Morse scaler (alpha 1.7), seed 0, fixed endpoints, and
fixed friction reference (0.001). The driver uses the same corrected overlap
guard, geometry/cache invalidation, private residual/Jacobian copies, soft-L1
loss, and SciPy stopping protocol (`gtol=0.002`, `ftol=xtol=machine epsilon`,
maximum 50 residual evaluations). The baseline constructor's extra alignment
is undone before any coordinate evaluation. No post-solve realignment occurs.

The timed operation is a **complete global optimization**, including trial
screening, coordinate and Jacobian evaluation, and sparse least-squares solves.
This isolates the old and new evaluation/assembly implementations under a
common corrected protocol. It does not compare sweeping, and it does not
present the adapter as unmodified master. Timing and peak-RSS methodology are
the same warmed, single-thread, three-process protocol described above.

The harness asserts matching input, pair, endpoint, reference, metric-radius,
and safety-radius digests. Every measured run must independently reach 0.002,
preserve endpoints exactly, agree on the gradient infinity norm and length
within `1e-10`, and have zero omitted-pair violations under the corrected radii. Failure of
any condition makes the comparison fail rather than yielding a speedup claim.

| Case | Evaluation/assembly implementation | Median full-solve time (s) | Median peak RSS (MiB) | Independent gradient infinity norm |
| --- | --- | ---: | ---: | ---: |
| Methane, 10 images | Baseline through common driver | 0.1530 | 76.6 | 0.0018196057433 |
| Methane, 10 images | Current through common driver | 0.0776 | 76.2 | 0.0018196057433 |
| Trp-cage, 10 images | Baseline through common driver | 6.2821 | 163.6 | 0.0019979845265 |
| Trp-cage, 10 images | Current through common driver | 5.9670 | 151.5 | 0.0019979845265 |

All matching conditions pass for all 12 timed solves. Both implementations
produce exactly the same independently calculated terminal gradient infinity
norm and path length in each case: methane length 1.241201150316995 and Trp-cage
length 5.371678498047353. They also take the same solver steps: 28 residual/23
Jacobian evaluations for methane, and 35 residual/21 Jacobian evaluations for
Trp-cage (11 guarded rejections in each Trp-cage solve). Endpoints remain exact,
and independently enumerated omitted pairs have zero violations at output
images and arithmetic midpoints under the corrected radii. The raw warmed
three-sample measurements are in
[results/matched-smoothing.json](results/matched-smoothing.json).

The controlled comparison therefore meets the matching conditions for global
smoothing on these two paths. It measures the effect of the old/new evaluation
and assembly kernels in a shared full-path solver, not the effect of all changes
to the production optimizer. The unmodified-master numbers below remain
ineligible as convergence-speed evidence.

## Production smoothing and sweep results

Measured on macOS arm64, Python 3.12.14, NumPy 2.4.6, SciPy 1.17.1.

| Case | Method | Median time (s) | Median peak RSS (MiB) | Independent gradient infinity norm | Reached 0.002? |
| --- | --- | ---: | ---: | ---: | --- |
| Methane, 6 atoms | Global | 0.0821 | 76.5 | 0.001819606 | Yes |
| Methane, 6 atoms | Sweep | 0.1035 | 76.3 | 0.001858107 | Yes |
| Trp-cage, 284 atoms | Global | 6.0344 | 136.8 | 0.001997985 | Yes |
| Trp-cage, 284 atoms | Sweep | 28.9495 | 142.6 | 0.011602299 | No |
| Collagen, 460 atoms | Global | 8.3651 | 136.7 | 0.001270873 | Yes |
| Collagen, 460 atoms | Sweep | 29.2495 | 141.7 | 0.001624931 | Yes |
| Calcium binding, 600 atoms (superseded guard) | Global | 9.5947 | 170.6 | 0.003537610 | No |
| Calcium binding, 600 atoms (superseded guard) | Sweep | 35.6805 | 175.3 | 0.017556690 | No |

Methane uses the same selected atom pairs, seed, initial path, and fixed
endpoints for both methods. Both reported final lengths agree exactly with
independent calculation, and neither prepared endpoint changes at all. These
two methods reach the common threshold; their terminal gradients and paths are
not identical. Raw three-sample results are in [results/methane.json](results/methane.json).

On Trp-cage, global smoothing reaches the common threshold within its budget,
while sweeping does not reach it after 35 sweeps. Their times therefore cannot
be compared as time to the same convergence threshold. This documents the
remaining sweep convergence limit instead of claiming comparable convergence.
Both collagen methods reach the common threshold. Neither calcium method
reaches the threshold within its evaluation or sweep budget. These results are for
the specified input paths and budgets, not a guarantee of convergence for other
molecules or initial paths. The complete per-run measurements are in
[results/proteins.json](results/proteins.json).

Every protein run preserves its prepared endpoints exactly. Independent final
lengths agree with the reported values within `2e-15`, and independently
calculated gradients agree within `2e-16`, including for runs which exhaust
their budgets. Both collagen methods have zero independently detected
omitted-pair overlap violations at the sampled locations.

The calcium rows and safety fields in the historical
[results/proteins.json](results/proteins.json) are **superseded** by the radius
correction. Their oracle reused the legacy metric table, which gave calcium
the 1.5 Angstrom fallback instead of its published 1.76 Angstrom covalent
radius. The old sweep result has two omitted-pair violations when independently
checked with corrected radii; it must not be described as passing the requested
safety threshold. Omitted C167-Ca598, for example, reaches 1.3562697300 Angstrom
from an initial 24.6782547176 Angstrom, below its correct 1.512 Angstrom cutoff.
The current benchmark oracles load the complete covalent-radius table from the
current reference checkout even when timing an older implementation. The
legacy metric/scaling table remains unchanged so the objective is preserved.

The corrected 10-image calcium sweep (seed 0, friction 0.001, 35 sweeps,
20 evaluations per interior image) ends with C167-Ca598 at **1.71320765248
Angstrom**, above the 1.512 Angstrom threshold. An independent all-pairs audit
using explicit published C/N/O/S/Ca radii finds **zero violations** at output
images and arithmetic midpoints; the minimum omitted-pair clearance is
`2.1838e-13` Angstrom. Input, selected-pair, and prepared-endpoint digests match
the earlier run exactly, and both endpoints remain fixed. The independent
gradient infinity norm is **0.01279155460118**, so the run is explicitly
**not converged** to 0.002. This is a correctness rerun, not a new warmed
three-sample timing comparison. The before/after measurements and literal-radii
audit are in [results/calcium-guard-review.json](results/calcium-guard-review.json).
The corrected numerical result can be reproduced with:

```sh
python benchmarks/validate_smoothing.py --cases calcium --methods sweep \
  --images 10 --seed 0 --friction 0.001 --tol 0.002 \
  --sweeps 35 --micro-iter 20 --repeats 1 --output /tmp/calcium-corrected.json
```

A final-code rerun confirms unchanged methane and Trp-cage global gradients,
lengths, and fixed endpoints after the SciPy compatibility adjustment. The
single Trp-cage recheck independently found zero omitted-pair violations at
images and arithmetic midpoints; its records are in
[results/final-code-trp.json](results/final-code-trp.json). The timing table uses
the three-sample medians, not this single verification timing.

Midpoint interpolation has a separate [matched-objective comparison](MIDPOINT.md)
with baseline timings, callback counts, independent gradients, and overlap
checks for its two endpoint-biased starting guesses.

### Baseline limitation

The baseline is commit `476aa041f562d37d5fb4121b86817a52d75e4c5d`, measured by
pointing `--source` at a checkout of that commit. It receives the same prepared
input and NumPy seed. Its constructor and optimizers perform additional
alignment, so it does **not** preserve the fixed-endpoint problem. On methane:

| Baseline method | Median time (s) | Median peak RSS (MiB) | Independent gradient infinity norm | Endpoint change (max Cartesian component, Angstrom) | Final length error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global | 0.0720 | 76.4 | 0.009346380 | 0.126 | 0.00966 |
| Sweep | 0.8970 | 76.2 | 0.025476920 | 0.209 | 0.0772 |

Neither baseline method reaches 0.002 for the common objective, and both change
the endpoints. These timings are diagnostic only: they do not support a
baseline-to-patch convergence speedup claim. The gradient is evaluated with the
same fixed initial reference even though baseline sweeping uses changing local
references internally. Raw results are in
[results/baseline-methane.json](results/baseline-methane.json).

### Large derivative storage

For the full 17-image calcium-binding example (600 atoms, seed 0), there are
2,823 selected pairs and a 117,336 by 27,000 whole-path Jacobian. Its 1,043,280
stored entries require **12,988,708 bytes** in CSR form (data, indices, and row
pointers); the corresponding dense matrix would require **25,344,576,000 bytes**.
Cached image/midpoint coordinate Jacobians add 7,080,216 bytes, and the reusable
assembly layout adds 12,988,708 bytes. The compatibility `grad0` matrix, a CSR
copy without the friction rows, adds 12,556,708 bytes. These figures describe
specific arrays, not total process peak memory; no dense equivalent was
allocated. The measured values are in
[results/calcium-memory.json](results/calcium-memory.json).

The memory calculation constructs `Geodesic` from
`test_cases/calcium_binding_interpolated.xyz`, seeds NumPy with 0, and calls
`compute_target_func()`. CSR bytes are `data.nbytes + indices.nbytes +
indptr.nbytes`; dense bytes are the product of the matrix shape times eight.

## Scope of validation

Additional untimed end-to-end checks start from the raw endpoint files,
redistribute to 10 images, and use seed 0, tolerance 0.002, and CLI-default
friction 0.01. Both methods use budgets of 50 (solver evaluations for global,
complete sweeps for sweep):

| Raw-endpoint case | Method | Independent final gradient infinity norm | Independent length | Target calculations |
| --- | --- | ---: | ---: | ---: |
| Methane | Global | 0.00169887596318 | 1.24428424341639 | 26 |
| Methane | Sweep | 0.00188382366557 | 1.24831316313897 | 794 |
| Diels-Alder | Global | 0.00117093069134 | 2.19118441746391 | 18 |
| Diels-Alder | Sweep | 0.00193331547494 | 2.18559660368715 | 1095 |

All four checks preserve their prepared endpoints exactly, agree with independent
lengths and gradients within `1e-12`, and have zero omitted-pair overlap
violations at images and arithmetic midpoints. A built-wheel install and CLI
smoke check with five images also pass.

Regression tests cover robust-gradient finite differences, numeric Morse scale
parameters, dense/sparse derivative equivalence, callback mutation protection,
final length and gradient recalculation, exact endpoint and subsegment
preservation, complete forward/backward sweep coverage, solver selection, and
same-tolerance methane convergence. Separate overlap tests cover omitted pairs,
initial-path rejection, rejected-trial restoration, and Cartesian midpoints.

All 63 unit tests pass in both tested environments: Python 3.12.14 with NumPy
2.4.6/SciPy 1.17.1, and Python 3.9.6 with NumPy 2.0.2/SciPy 1.13.1. Budget
exhaustion is explicitly tested to emit a nonconvergence warning.
The original [GitHub Actions matrix run](https://github.com/virtualzx-nad/geodesic-interpolate/actions/runs/37219675842)
passed all five Python versions (3.8 through 3.12) on implementation commit
`dbf4799`.

The benchmark also independently enumerates all atom pairs at final images and
the arithmetic midpoints, records the closest pairs, and checks omitted-pair
clearance against the specified thresholds. This is a check at sampled
locations, with minimum distance `max(0.70 Angstrom, 0.60 * sum of covalent radii)`
for omitted pairs. It does not establish safety throughout the continuous path between
them. Failure to reach the requested gradient threshold is reported, rather
than being interpreted as convergence from a shorter path or solver status.
