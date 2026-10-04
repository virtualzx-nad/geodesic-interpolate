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
python benchmarks/validate_smoothing.py --cases trp --output /tmp/trp.json
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

## Results

Measured on macOS arm64, Python 3.12.14, NumPy 2.4.6, SciPy 1.17.1.

| Case | Method | Median time (s) | Median peak RSS (MiB) | Independent gradient infinity norm | Reached 0.002? |
| --- | --- | ---: | ---: | ---: | --- |
| Methane, 6 atoms | Global | 0.0861 | 76.9 | 0.001819606 | Yes |
| Methane, 6 atoms | Sweep | 0.1034 | 76.3 | 0.001858107 | Yes |
| Trp-cage, 284 atoms | Global | 6.0344 | 136.8 | 0.001997985 | Yes |
| Trp-cage, 284 atoms | Sweep | 28.9495 | 142.6 | 0.011602299 | No |
| Collagen, 460 atoms | Global | 8.3651 | 136.7 | 0.001270873 | Yes |
| Collagen, 460 atoms | Sweep | 29.2495 | 141.7 | 0.001624931 | Yes |
| Calcium binding, 600 atoms | Global | 9.5947 | 170.6 | 0.003537610 | No |

Methane uses the same selected atom pairs, seed, initial path, and fixed
endpoints for both methods. Both reported final lengths agree exactly with
independent calculation, and neither prepared endpoint changes at all. These
two methods reach the common threshold; their terminal gradients and paths are
not identical. Raw three-sample results are in [results/methane.json](results/methane.json).

On Trp-cage, global smoothing reaches the common threshold within its budget,
while sweeping does not reach it after 35 sweeps. Their times therefore cannot
be compared as time to the same convergence threshold. This documents the
remaining sweep convergence limit instead of claiming comparable convergence.
Both collagen methods reach the common threshold. Calcium global smoothing
exhausts its 50-evaluation budget above the threshold. These results are for
the specified input paths and budgets, not a guarantee of convergence for other
molecules or initial paths.

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
copy without the friction rows, adds 12,556,708 bytes. These figures describe specific arrays,
not total process peak memory; no dense equivalent was allocated. The measured
values are in [results/calcium-memory.json](results/calcium-memory.json).

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

All 56 unit tests pass in both tested environments: Python 3.12.14 with NumPy
2.4.6/SciPy 1.17.1, and Python 3.9.6 with NumPy 2.0.2/SciPy 1.13.1. Budget
exhaustion is explicitly tested to emit a nonconvergence warning.

The benchmark also independently enumerates all atom pairs at final images and
the arithmetic midpoints, records the closest pairs, and checks omitted-pair
clearance against the specified thresholds. This is a check at sampled
locations. It does not establish safety throughout the continuous path between
them. Failure to reach the requested gradient threshold is reported, rather
than being interpreted as convergence from a shorter path or solver status.
