import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from geodesic_interpolate.coord_utils import PairCoordinates, align_path, morse_scaler
from geodesic_interpolate.geodesic import Geodesic
from geodesic_interpolate.interpolation import _MidpointObjective, mid_point, redistribute
from geodesic_interpolate.validation import UnsafePathError


class MidpointTest(unittest.TestCase):
    def test_shared_objective_matches_finite_differences_for_both_solver_sizes(self):
        rng = np.random.default_rng(35)
        for natoms in (5, 34):
            with self.subTest(natoms=natoms):
                geom = rng.normal(size=(natoms, 3))
                coordinates = PairCoordinates(natoms, [(i, i + 1) for i in range(natoms - 1)])
                scaler = morse_scaler(alpha=0.8)
                x0 = geom.ravel().copy()
                reference = coordinates.compute(geom, scaler)[0] + 0.1
                objective = _MidpointObjective(coordinates, scaler, reference, x0, 0.03)
                trial = x0 + rng.normal(scale=0.01, size=x0.size)
                with patch.object(coordinates, "compute", wraps=coordinates.compute) as compute:
                    residual = objective.residual(trial)
                    jacobian = objective.derivative(trial)
                    self.assertEqual(compute.call_count, 1)
                self.assertEqual(sparse.issparse(jacobian), natoms * 3 > 100)
                np.testing.assert_allclose(residual[-trial.size:], (trial - x0) * 0.03)
                direction = rng.normal(size=trial.size)
                eps = 1e-6
                numeric = (objective.residual(trial + direction * eps) -
                           objective.residual(trial - direction * eps)) / (2 * eps)
                np.testing.assert_allclose(jacobian @ direction, numeric, rtol=1e-7, atol=1e-9)

    def test_solver_cannot_mutate_cached_objective(self):
        rng = np.random.default_rng(48)
        for natoms in (5, 34):
            with self.subTest(natoms=natoms):
                geom = rng.normal(size=(natoms, 3))
                coordinates = PairCoordinates(natoms, [(0, natoms - 1)])
                x0 = geom.ravel().copy()
                objective = _MidpointObjective(coordinates, morse_scaler(), np.zeros(1), x0, 0.03)
                original_value = objective.residual(x0)
                original_jacobian = objective.derivative(x0)
                objective.residual(x0)[:] = 0
                returned_jacobian = objective.derivative(x0)
                if sparse.issparse(returned_jacobian):
                    returned_jacobian.data[:] = 0
                    original_jacobian = original_jacobian.toarray()
                    actual_jacobian = objective.derivative(x0).toarray()
                else:
                    returned_jacobian[:] = 0
                    actual_jacobian = objective.derivative(x0)
                np.testing.assert_array_equal(objective.residual(x0), original_value)
                np.testing.assert_array_equal(actual_jacobian, original_jacobian)

    def test_large_midpoint_objective_does_not_build_dense_identity(self):
        natoms = 60
        geom = np.arange(natoms * 3, dtype=float).reshape(natoms, 3)
        coordinates = PairCoordinates(natoms, [(i, i + 1) for i in range(natoms - 1)])
        with patch("geodesic_interpolate.interpolation.np.eye", side_effect=AssertionError("dense identity")):
            objective = _MidpointObjective(coordinates, morse_scaler(), np.zeros(natoms - 1), geom.ravel(), 0.01)
            jacobian = objective.derivative(geom.ravel())
        self.assertTrue(sparse.isspmatrix_csr(jacobian))
        self.assertEqual(jacobian.nnz, 6 * (natoms - 1) + 3 * natoms)
        self.assertTrue(np.shares_memory(objective.jacobian.indices, objective.indices))
        self.assertTrue(np.shares_memory(objective.jacobian.indptr, objective.indptr))

    def test_midpoint_retains_two_endpoint_near_randomized_starts(self):
        first = np.array([[0., 0., 0.], [1.2, 0., 0.], [0., 1., 0.]])
        second = np.array([[0., 0., 0.], [1.4, 0., 0.], [0., 1.2, 0.]])
        starts = []

        def solve(residual, x0, derivative, **kwargs):
            starts.append(x0.copy())
            self.assertEqual(derivative(x0).shape, (residual(x0).size, x0.size))
            return dict(x=x0.copy(), nfev=1)

        with patch("geodesic_interpolate.interpolation.least_squares", side_effect=solve), \
                patch("geodesic_interpolate.interpolation.np.random.random_sample", return_value=np.full(first.size, 0.5)):
            result = mid_point(["H"] * 3, first, second, nudge=0.01)
        self.assertEqual(len(starts), 2)
        for coef, start in zip((0.02, 0.98), starts):
            np.testing.assert_allclose(start, (first * coef + (1 - coef) * second).ravel() + 0.005)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_midpoint_tries_other_start_after_unsafe_candidate(self):
        first = np.array([[0., 0., 0.], [1.2, 0., 0.], [0., 1., 0.]])
        second = first * 1.1
        attempts = []

        def score(*args, **kwargs):
            attempts.append(args[1])
            if len(attempts) == 1:
                raise UnsafePathError("unsafe midpoint")
            return Geodesic(*args, **kwargs)

        with patch("geodesic_interpolate.interpolation.Geodesic", side_effect=score):
            result = mid_point(["H"] * 3, first, second, nudge=0)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_array_equal(first, attempts[-1][0])
        np.testing.assert_array_equal(second, attempts[-1][-1])

    def test_midpoint_rejects_two_unsafe_candidates(self):
        first = np.array([[0., 0., 0.], [1.2, 0., 0.], [0., 1., 0.]])
        second = first * 1.1
        with patch("geodesic_interpolate.interpolation.Geodesic", side_effect=UnsafePathError("unsafe midpoint")) as score:
            with self.assertRaisesRegex(UnsafePathError, "Neither bisection candidate.*unsafe midpoint"):
                mid_point(["H"] * 3, first, second, nudge=0)
        self.assertEqual(score.call_count, 2)

    def test_redistribute_aligns_once_and_keeps_prepared_endpoints(self):
        rng = np.random.default_rng(92)
        first = rng.normal(size=(6, 3))
        last = first + rng.normal(scale=0.25, size=(6, 3))
        path = np.array([first, last])
        prepared = align_path(path)[1]
        with patch("geodesic_interpolate.interpolation.align_path", wraps=align_path) as align, \
                patch("geodesic_interpolate.interpolation.mid_point", side_effect=lambda atoms, left, right, tol: (left + right) / 2):
            result = redistribute(["C"] * 6, path, 6)
        self.assertEqual(align.call_count, 1)
        self.assertEqual(len(result), 6)
        np.testing.assert_array_equal(result[0], prepared[0])
        np.testing.assert_array_equal(result[-1], prepared[-1])
        np.testing.assert_array_equal(path, [first, last])

        prepared = align_path(result)[1]
        with patch("geodesic_interpolate.interpolation.align_path", wraps=align_path) as align:
            reduced = redistribute(["C"] * 6, result, 3)
        self.assertEqual(align.call_count, 1)
        np.testing.assert_array_equal(reduced[0], prepared[0])
        np.testing.assert_array_equal(reduced[-1], prepared[-1])


if __name__ == "__main__":
    unittest.main()
