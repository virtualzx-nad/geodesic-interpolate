"""End-to-end and independent numerical checks for the smoothing objective."""
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from benchmarks.validate_smoothing import independent_values
from geodesic_interpolate.coord_utils import align_path
from geodesic_interpolate.fileio import read_xyz
from geodesic_interpolate.geodesic import Geodesic


ROOT = Path(__file__).resolve().parents[1]


def _quadratic_scaler(distance):
    return distance ** 2, 2 * distance


class SmoothingTest(unittest.TestCase):
    def example(self, nimages=5, natoms=4, scaler=_quadratic_scaler):
        rng = np.random.default_rng(154)
        path = rng.normal(size=(nimages, natoms, 3))
        return Geodesic(["C"] * natoms, path, scaler=scaler,
                        threshold=100.0, min_neighbors=0, align=False)

    def test_numeric_scaler_uses_the_requested_alpha(self):
        for alpha in (0.7, 1.7, 1, np.float32(0.7), np.float64(0.7)):
            with self.subTest(alpha=alpha, type=type(alpha)):
                geodesic = self.example(scaler=alpha)
                distance = geodesic.re * 1.2
                value, derivative = geodesic.scaler(distance)
                exponential = np.exp(float(alpha) * (1 - distance / geodesic.re))
                expected = exponential + 0.01 * geodesic.re / distance
                expected_derivative = (-float(alpha) / geodesic.re * exponential -
                                       0.01 * geodesic.re / distance ** 2)
                np.testing.assert_allclose(value, expected, rtol=1e-12)
                np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-12)

    def test_robust_gradient_matches_independent_central_difference_with_friction(self):
        geodesic = self.example()
        start, end, friction = 1, 4, 0.4
        x = geodesic.path[start:end].ravel().copy()
        xref = x + np.linspace(-0.6, 0.8, x.size)
        geodesic.compute_target_func(x, start=start, end=end, x0=xref, friction=friction)
        independent = independent_values(geodesic.path, geodesic.rij_list,
                                          _quadratic_scaler, xref, friction, start, end)
        numeric = np.empty_like(x)
        eps = 1e-6
        for coordinate in range(x.size):
            shifted = geodesic.path.copy()
            shifted[start:end].reshape(-1)[coordinate] += eps
            plus = independent_values(shifted, geodesic.rij_list,
                                      _quadratic_scaler, xref, friction, start, end)
            shifted[start:end].reshape(-1)[coordinate] -= 2 * eps
            minus = independent_values(shifted, geodesic.rij_list,
                                       _quadratic_scaler, xref, friction, start, end)
            numeric[coordinate] = (plus["objective"] - minus["objective"]) / (2 * eps)
        analytic = np.asarray(geodesic.grad.T @ (
            geodesic.disps / np.hypot(1.0, geodesic.disps))).ravel()
        np.testing.assert_allclose(independent["gradient"], numeric, rtol=2e-6, atol=3e-8)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=3e-8)
        np.testing.assert_allclose(geodesic.disps, independent["residual"], atol=1e-13)
        self.assertAlmostEqual(geodesic.optimality, np.max(np.abs(numeric)), places=6)
        self.assertAlmostEqual(geodesic.cost, independent["objective"], places=11)
        # The robust gradient differs materially from the ordinary least-squares one.
        ordinary = np.asarray(geodesic.grad.T @ geodesic.disps).ravel()
        self.assertGreater(np.max(np.abs(ordinary - analytic)), 1.0)

    def test_solver_callbacks_cannot_mutate_cached_residual_or_derivatives(self):
        geodesic = self.example()
        x = geodesic.path[1:-1].ravel().copy()
        kwargs = dict(x0=x + 0.2, friction=0.1)
        residual = geodesic.target_func(x, **kwargs)
        expected_residual = residual.copy()
        residual[:] = 12345.0
        np.testing.assert_array_equal(geodesic.target_func(x, **kwargs), expected_residual)
        jacobian = geodesic.target_deriv_sparse(x, **kwargs)
        expected_jacobian = jacobian.toarray()
        jacobian.data[:] = 12345.0
        np.testing.assert_array_equal(geodesic.target_deriv_sparse(x, **kwargs).toarray(),
                                      expected_jacobian)
        dense = geodesic.target_deriv(x, **kwargs)
        dense[:] = -12345.0
        np.testing.assert_array_equal(geodesic.target_deriv(x, **kwargs), expected_jacobian)

    def test_smooth_and_sweep_preserve_endpoints_and_recalculate_final_values(self):
        for method in ("smooth", "sweep"):
            with self.subTest(method=method):
                geodesic = self.example(scaler=1.7)
                initial = geodesic.path.copy()
                xref = initial[1:-1].ravel().copy()
                if method == "smooth":
                    geodesic.smooth(tol=1e-9, max_iter=3)
                else:
                    geodesic.sweep(tol=1e-9, max_iter=2, micro_iter=2)
                expected = independent_values(geodesic.path, geodesic.rij_list,
                                               geodesic.scaler, xref, geodesic.friction)
                np.testing.assert_array_equal(geodesic.path[[0, -1]], initial[[0, -1]])
                self.assertAlmostEqual(geodesic.length, expected["length"], places=11)
                self.assertAlmostEqual(geodesic.optimality, expected["optimality"], places=11)

    def test_subsegment_preserves_every_unoptimized_image(self):
        for method in ("smooth", "sweep"):
            with self.subTest(method=method):
                geodesic = self.example(nimages=6, scaler=1.7)
                initial = geodesic.path.copy()
                start, end = 2, 4
                kwargs = dict(start=start, end=end, max_iter=2, tol=1e-9)
                getattr(geodesic, method)(**kwargs)
                np.testing.assert_array_equal(geodesic.path[[0, 1, 4, 5]], initial[[0, 1, 4, 5]])
                expected = independent_values(
                    geodesic.path, geodesic.rij_list, geodesic.scaler,
                    initial[start:end], geodesic.friction, start, end)
                self.assertAlmostEqual(geodesic.length, expected["length"], places=11)
                self.assertAlmostEqual(geodesic.optimality, expected["optimality"], places=11)

    def test_each_sweep_visits_every_interior_image(self):
        geodesic = self.example(nimages=6)
        visited = []
        original = geodesic.smooth

        def track(*args, **kwargs):
            visited.append(kwargs["start"])
            return original(*args, **kwargs)

        with mock.patch.object(geodesic, "smooth", side_effect=track):
            geodesic.sweep(tol=1e-12, max_iter=2, micro_iter=1)
        self.assertEqual(visited[:4], [1, 2, 3, 4])
        self.assertEqual(visited[4:], [4, 3, 2, 1])

    def test_small_smooth_uses_dense_solver_callback(self):
        geodesic = self.example(nimages=4)
        with mock.patch.object(geodesic, "target_deriv", wraps=geodesic.target_deriv) as dense:
            geodesic.smooth(tol=1e-12, max_iter=1)
        self.assertGreater(dense.call_count, 0)

    def test_methane_both_methods_reach_same_independent_tolerance(self):
        atoms, frames = read_xyz(ROOT / "test_cases" / "H+CH4_CH3+H2_interpolated.xyz")
        _, prepared = align_path(np.asarray(frames)[np.linspace(0, len(frames) - 1, 10, dtype=int)])
        pairs = None
        for method in ("smooth", "sweep"):
            with self.subTest(method=method):
                np.random.seed(0)
                geodesic = Geodesic(atoms, prepared, align=False)
                if pairs is None:
                    pairs = geodesic.rij_list
                else:
                    self.assertEqual(geodesic.rij_list, pairs)
                xref = prepared[1:-1].ravel().copy()
                if method == "smooth":
                    geodesic.smooth(tol=0.002, max_iter=50)
                else:
                    geodesic.sweep(tol=0.002, max_iter=35)
                expected = independent_values(geodesic.path, geodesic.rij_list,
                                               geodesic.scaler, xref, geodesic.friction)
                self.assertLessEqual(expected["optimality"], 0.002)
                self.assertAlmostEqual(geodesic.optimality, expected["optimality"], places=11)
                self.assertAlmostEqual(geodesic.length, expected["length"], places=11)
                np.testing.assert_array_equal(geodesic.path[[0, -1]], prepared[[0, -1]])


if __name__ == "__main__":
    unittest.main()
