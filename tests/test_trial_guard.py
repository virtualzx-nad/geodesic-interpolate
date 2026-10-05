import unittest
from unittest import mock

import numpy as np

from geodesic_interpolate import geodesic as geodesic_module
from geodesic_interpolate.geodesic import Geodesic
from geodesic_interpolate.validation import UnsafePathError


def guarded_smoother(nimages=5):
    """A varying selected bond plus a distant atom omitted from the metric."""
    path = np.zeros((nimages, 3, 3))
    path[:, 1, 0] = 1.5
    path[1:-1, 1, 0] += np.linspace(.1, .7, nimages - 2)
    path[:, 2, :] = [5., 2., 0.]
    with mock.patch.object(geodesic_module, "get_bond_list", return_value=([(0, 1)], np.array([1.52]))):
        return Geodesic(["C"] * 3, path, align=False)


class TrialGuardTest(unittest.TestCase):
    def test_exhausted_budgets_warn_instead_of_reporting_convergence(self):
        for method in ("smooth", "sweep"):
            with self.subTest(method=method):
                smoother = guarded_smoother()
                kwargs = dict(tol=1e-12, max_iter=1)
                if method == "sweep":
                    kwargs["micro_iter"] = 1
                with self.assertLogs(geodesic_module.logger, level="WARNING") as logs:
                    getattr(smoother, method)(**kwargs)
                self.assertGreater(smoother.optimality, kwargs["tol"])
                self.assertTrue(any("did not converge" in line for line in logs.output))
                self.assert_cached_objective(smoother)

    def test_unsafe_trial_keeps_geometry_and_all_caches_unchanged(self):
        for violation in ["image", "midpoint", "selected_pair", "nonfinite"]:
            with self.subTest(violation=violation):
                smoother = guarded_smoother()
                start, end = 2, 3
                x0 = smoother.path[start:end].ravel().copy()
                kwargs = dict(start=start, end=end, x0=x0, friction=.01)
                smoother.compute_target_func(x0, **kwargs)
                original = smoother.path.copy()
                cached_f = smoother.disps.copy()
                cached_j = smoother.grad.toarray()
                cached_w = list(smoother.w)
                cached_mid = list(smoother.w_mid)
                candidate = smoother.path[start:end].copy()
                if violation == "image":
                    candidate[0, 2] = [.1, .1, 0.]
                elif violation == "midpoint":
                    candidate[0, 2] = [-5., -2., 0.]
                elif violation == "selected_pair":
                    candidate[0, 1] = candidate[0, 0]
                else:
                    candidate[0, 2, 0] = np.nan
                expected_error = {
                    "image": "image 2", "midpoint": "midpoint between images 1 and 2",
                    "selected_pair": "coincident", "nonfinite": "finite coordinates",
                }[violation]
                with self.assertRaisesRegex(UnsafePathError, expected_error):
                    smoother.update_geometry(candidate.ravel(), start, end)
                residual = smoother.target_func(candidate.ravel(), **kwargs)
                dense = smoother.target_deriv(candidate.ravel(), **kwargs)
                sparse = smoother.target_deriv_sparse(candidate.ravel(), **kwargs)
                np.testing.assert_array_equal(smoother.path, original)
                np.testing.assert_array_equal(smoother.disps, cached_f)
                np.testing.assert_array_equal(smoother.grad.toarray(), cached_j)
                self.assertTrue(all(a is b for a, b in zip(cached_w, smoother.w)))
                self.assertTrue(all(a is b for a, b in zip(cached_mid, smoother.w_mid)))
                self.assertEqual(smoother.rejected_trials, 3)
                self.assertEqual(residual.shape, cached_f.shape)
                self.assertTrue(np.all(residual == 1e20))
                np.testing.assert_array_equal(dense, np.zeros_like(cached_j))
                self.assertEqual(sparse.shape, cached_j.shape)
                self.assertEqual(sparse.nnz, 0)

    def test_first_unsafe_callback_builds_safe_fallback_cache(self):
        smoother = guarded_smoother()
        original = smoother.path.copy()
        candidate = original[1:-1].copy()
        candidate[0, 2] = [.1, .1, 0.]
        residual = smoother.target_func(candidate.ravel())
        self.assertTrue(np.all(np.isfinite(residual)))
        self.assertTrue(np.all(residual == 1e20))
        np.testing.assert_array_equal(smoother.path, original)
        self.assertTrue(np.all(np.isfinite(smoother.disps)))
        self.assertTrue(np.all(np.isfinite(smoother.grad.data)))

    def test_rejected_callback_uses_requested_segment_reference_and_friction(self):
        smoother = guarded_smoother()
        original = smoother.path.copy()
        smoother.compute_target_func()
        candidate = original[2:3].copy()
        candidate[0, 2] = [.1, .1, 0.]
        reference = original[2:3].ravel() + .1
        kwargs = dict(start=2, end=3, x0=reference, friction=.04)
        residual = smoother.target_func(candidate.ravel(), **kwargs)
        derivative = smoother.target_deriv_sparse(candidate.ravel(), **kwargs)
        # Two adjacent segments each contribute two selected-distance rows,
        # followed by nine Cartesian friction rows for one optimized image.
        self.assertEqual(residual.shape, (13,))
        self.assertEqual(derivative.shape, (13, 9))
        self.assertEqual(smoother.segment, (2, 3))
        np.testing.assert_allclose(smoother.disps[-9:], np.full(9, -.004))
        np.testing.assert_array_equal(smoother.path, original)

    def test_unsafe_solver_result_restores_starting_segment(self):
        smoother = guarded_smoother()
        original = smoother.path.copy()

        def solver(fun, x0, jac, **options):
            trial = x0.reshape(-1, 3, 3).copy()
            trial[0, 1, 1] = .2
            fun(trial.ravel(), **options["kwargs"])
            self.assertFalse(np.array_equal(smoother.path, original))
            trial[0, 2] = [.1, .1, 0.]
            return {"x": trial.ravel()}

        with mock.patch.object(geodesic_module, "least_squares", side_effect=solver), \
                self.assertLogs(geodesic_module.logger, level="WARNING") as logs:
            smoother.smooth(tol=1e-10)
        np.testing.assert_array_equal(smoother.path, original)
        self.assertTrue(any("restored the initial segment" in line for line in logs.output))
        self.assert_cached_objective(smoother)

    def test_solver_error_or_interrupt_restores_starting_segment_and_cache(self):
        for error_type in [RuntimeError, KeyboardInterrupt]:
            with self.subTest(error_type=error_type):
                smoother = guarded_smoother()
                original = smoother.path.copy()

                def solver(fun, x0, jac, **options):
                    trial = x0.copy()
                    trial[4] += .2
                    fun(trial, **options["kwargs"])
                    self.assertFalse(np.array_equal(smoother.path, original))
                    raise error_type("forced solver failure")

                with mock.patch.object(geodesic_module, "least_squares", side_effect=solver):
                    with self.assertRaises(error_type):
                        smoother.smooth(tol=1e-10, start=2, end=3)
                np.testing.assert_array_equal(smoother.path, original)
                self.assert_cached_objective(smoother, start=2, end=3)

    def test_callbacks_protect_cached_residual_values_and_sparse_layout(self):
        smoother = guarded_smoother()
        x0 = smoother.path[1:-1].ravel().copy()
        kwargs = dict(x0=x0, friction=.03)
        smoother.compute_target_func(x0, **kwargs)
        expected_f = smoother.disps.copy()
        expected_j = smoother.grad.toarray()
        expected_indices = smoother.grad.indices.copy()
        expected_indptr = smoother.grad.indptr.copy()
        residual = smoother.target_func(x0, **kwargs)
        derivative = smoother.target_deriv_sparse(x0, **kwargs)
        residual.fill(123.)
        derivative.data.fill(321.)
        derivative.indices.fill(0)
        derivative.indptr.fill(0)
        np.testing.assert_array_equal(smoother.disps, expected_f)
        np.testing.assert_array_equal(smoother.grad.toarray(), expected_j)
        np.testing.assert_array_equal(smoother.grad.indices, expected_indices)
        np.testing.assert_array_equal(smoother.grad.indptr, expected_indptr)
        np.testing.assert_array_equal(smoother.target_func(x0, **kwargs), expected_f)
        np.testing.assert_array_equal(smoother.target_deriv(x0, **kwargs), expected_j)

    def test_final_values_match_accepted_result_after_later_unaccepted_trial(self):
        smoother = guarded_smoother()
        original = smoother.path.copy()
        accepted = original[1:-1].copy()
        accepted[0, 1, 1] = .2

        def solver(fun, x0, jac, **options):
            fun(accepted.ravel(), **options["kwargs"])
            later_trial = accepted.copy()
            later_trial[0, 1, 1] = .4
            fun(later_trial.ravel(), **options["kwargs"])
            return {"x": accepted.ravel()}

        with mock.patch.object(geodesic_module, "least_squares", side_effect=solver):
            smoother.smooth(tol=1e-10)
        np.testing.assert_array_equal(smoother.path[1:-1], accepted)
        np.testing.assert_array_equal(smoother.path[[0, -1]], original[[0, -1]])
        self.assert_cached_objective(smoother)

    def test_real_partial_solve_preserves_all_unoptimized_images_exactly(self):
        smoother = guarded_smoother(nimages=7)
        original = smoother.path.copy()
        smoother.smooth(tol=1e-10, max_iter=3, start=2, end=4)
        np.testing.assert_array_equal(smoother.path[[0, 1, 4, 5, 6]], original[[0, 1, 4, 5, 6]])
        self.assertFalse(np.array_equal(smoother.path[2:4], original[2:4]))
        self.assert_cached_objective(smoother, start=2, end=4)

    def test_csr_assembly_layout_reused_across_equal_size_segments(self):
        smoother = guarded_smoother(nimages=6)
        smoother.compute_target_func(start=1, end=3, friction=.02)
        first_layout = smoother._layouts[(2, True)]
        smoother.compute_target_func(start=2, end=4, friction=.04)
        self.assertIs(smoother._layouts[(2, True)], first_layout)
        smoother.compute_target_func(start=2, end=4, friction=0.)
        self.assertIn((2, False), smoother._layouts)
        np.testing.assert_array_equal(smoother.grad[-18:].toarray(), np.zeros((18, 18)))

    def assert_cached_objective(self, smoother, start=1, end=-1):
        if end < 0:
            end += smoother.nimages
        # Independently reconstruct path length from selected scalar distances.
        def coordinates(geometry):
            return smoother.scaler(np.linalg.norm(geometry[0] - geometry[1]))[0].item()

        length = 0.
        for left, right in zip(smoother.path[start - 1:end], smoother.path[start:end + 1]):
            midpoint = coordinates((left + right) * .5)
            length += abs(midpoint - coordinates(left)) + abs(coordinates(right) - midpoint)
        self.assertAlmostEqual(smoother.length, float(length), places=12)
        gradient = smoother.grad.T @ (smoother.disps / np.hypot(1., smoother.disps))
        self.assertAlmostEqual(smoother.optimality, np.max(np.abs(gradient)), places=12)
        self.assertAlmostEqual(smoother.cost, np.sum(np.hypot(1., smoother.disps) - 1.), places=12)


if __name__ == "__main__":
    unittest.main()
