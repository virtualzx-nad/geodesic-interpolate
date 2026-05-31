import unittest

import numpy as np
from scipy import sparse

from geodesic_interpolate.coord_utils import compute_wij, compute_wij_sparse, morse_scaler
from geodesic_interpolate.fileio import read_xyz
from geodesic_interpolate.geodesic import Geodesic


def _sparse_bytes(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def _cached_derivative_bytes(geodesic):
    matrices = [
        matrix for matrix in geodesic.dwdR + geodesic.dwdR_mid
        if matrix is not None
    ]
    return sum(_sparse_bytes(matrix) for matrix in matrices)


def _dense_reference_jacobian(geodesic, start, end, friction):
    ncoords = geodesic.natoms * 3
    nimages = end - start
    nsegments = end - start + 1
    grad = np.zeros((nsegments * 2 * geodesic.nrij + nimages * ncoords,
                     nimages * ncoords))
    grad_l = grad[:nsegments * geodesic.nrij].reshape(
        (nsegments, geodesic.nrij, nimages, ncoords))
    grad_r = grad[nsegments * geodesic.nrij:nsegments * geodesic.nrij * 2].reshape(
        (nsegments, geodesic.nrij, nimages, ncoords))

    dwdR = [
        compute_wij(image, geodesic.rij_list, geodesic.scaler)[1]
        for image in geodesic.path
    ]
    dwdR_mid = [
        compute_wij((left + right) * 0.5, geodesic.rij_list, geodesic.scaler)[1]
        for left, right in zip(geodesic.path, geodesic.path[1:])
    ]

    for i, image in enumerate(range(start, end)):
        dmid1 = dwdR_mid[image - 1] * 0.5
        dmid2 = dwdR_mid[image] * 0.5
        grad_l[i + 1, :, i, :] = dmid2 - dwdR[image]
        grad_l[i, :, i, :] = dmid1
        grad_r[i + 1, :, i, :] = -dmid2
        grad_r[i, :, i, :] = dwdR[image] - dmid1
    for idx in range(nimages * ncoords):
        grad[nsegments * geodesic.nrij * 2 + idx, idx] = friction
    return grad


class SparseGeodesicTest(unittest.TestCase):
    def test_sparse_wij_matches_dense_wij(self):
        rng = np.random.default_rng(7)
        geom = rng.normal(size=(6, 3))
        rij_list = [(0, 1), (0, 4), (2, 3), (3, 5)]
        scaler = morse_scaler(re=np.linspace(1.0, 2.0, len(rij_list)), alpha=1.1)

        wij_dense, bmat_dense = compute_wij(geom, rij_list, scaler)
        wij_sparse, bmat_sparse = compute_wij_sparse(geom, rij_list, scaler)

        self.assertEqual(bmat_sparse.shape, bmat_dense.shape)
        self.assertTrue(sparse.isspmatrix_csr(bmat_sparse))
        np.testing.assert_allclose(wij_sparse, wij_dense)
        np.testing.assert_allclose(bmat_sparse.toarray(), bmat_dense)

    def test_sparse_wij_supports_custom_scalers(self):
        rng = np.random.default_rng(8)
        geom = rng.normal(size=(5, 3))
        rij_list = [(0, 2), (1, 3), (3, 4)]

        def scaler(rij):
            return rij ** 2, 2 * rij

        wij_dense, bmat_dense = compute_wij(geom, rij_list, scaler)
        wij_sparse, bmat_sparse = compute_wij_sparse(geom, rij_list, scaler)

        np.testing.assert_allclose(wij_sparse, wij_dense)
        np.testing.assert_allclose(bmat_sparse.toarray(), bmat_dense)

    def test_geodesic_sparse_jacobian_matches_dense_reference(self):
        rng = np.random.default_rng(11)
        atoms = ["C", "H", "O", "N", "C"]
        path = rng.normal(size=(5, len(atoms), 3))
        geodesic = Geodesic(atoms, path, scaler=1.0, threshold=10.0, min_neighbors=0)
        for start, end, friction in [(1, 4, 0.05), (1, 2, 0.0), (3, 4, 0.1)]:
            with self.subTest(start=start, end=end, friction=friction):
                x0 = geodesic.path[start:end].ravel().copy()

                geodesic.compute_target_func(
                    start=start, end=end, x0=x0, friction=friction)

                self.assertTrue(sparse.isspmatrix_csr(geodesic.grad))
                np.testing.assert_allclose(
                    geodesic.grad.toarray(),
                    _dense_reference_jacobian(geodesic, start, end, friction),
                )
                self.assertIsInstance(
                    geodesic.target_deriv(
                        x0, start=start, end=end, x0=x0, friction=friction),
                    np.ndarray,
                )
                self.assertTrue(sparse.isspmatrix_csr(
                    geodesic.target_deriv_sparse(
                        x0, start=start, end=end, x0=x0, friction=friction)))
                np.testing.assert_allclose(
                    geodesic.target_deriv(
                        x0, start=start, end=end, x0=x0, friction=friction),
                    geodesic.grad.toarray(),
                )

    def test_sparse_jacobian_matches_directional_finite_difference(self):
        rng = np.random.default_rng(17)
        atoms = ["C", "H", "O", "N"]
        path = rng.normal(size=(4, len(atoms), 3))
        geodesic = Geodesic(atoms, path, scaler=1.0, threshold=10.0, min_neighbors=0)
        start, end, friction = 1, 3, 0.02
        x0 = geodesic.path[start:end].ravel().copy()
        direction = rng.normal(size=x0.size)
        direction /= np.linalg.norm(direction)
        eps = 1e-6

        def residual(x):
            geodesic.compute_target_func(
                x, start=start, end=end, x0=x0, friction=friction)
            return geodesic.disps.copy()

        geodesic.compute_target_func(
            x0, start=start, end=end, x0=x0, friction=friction)
        analytic = geodesic.grad @ direction
        numeric = (residual(x0 + eps * direction) -
                   residual(x0 - eps * direction)) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)

    def test_target_cache_accounts_for_friction(self):
        rng = np.random.default_rng(19)
        atoms = ["C", "H", "O"]
        path = rng.normal(size=(4, len(atoms), 3))
        geodesic = Geodesic(atoms, path, scaler=1.0, threshold=10.0, min_neighbors=0)
        start, end = 1, 3
        x0 = geodesic.path[start:end].ravel().copy()

        geodesic.compute_target_func(
            x0, start=start, end=end, x0=x0, friction=0.0)
        no_friction_grad = geodesic.grad.copy()
        geodesic.compute_target_func(
            x0, start=start, end=end, x0=x0, friction=0.5)

        self.assertEqual(geodesic.neval, 2)
        self.assertGreater(geodesic.grad.nnz, no_friction_grad.nnz)
        np.testing.assert_allclose(
            no_friction_grad[-x0.size:].diagonal(),
            np.zeros(x0.size),
        )
        np.testing.assert_allclose(
            geodesic.grad[-x0.size:].diagonal(),
            np.full(x0.size, 0.5),
        )

    def test_bundled_targets_use_sparse_issue_sized_matrices(self):
        cases = [
            ("test_cases/DielsAlder_interpolated.xyz", 1, -1, 5 * 1024 * 1024),
            ("test_cases/calcium_binding_interpolated.xyz", 1, 2, 20 * 1024 * 1024),
        ]
        for filename, start, end, max_sparse_bytes in cases:
            with self.subTest(filename=filename):
                np.random.seed(0)
                atoms, coords = read_xyz(filename)
                geodesic = Geodesic(atoms, coords, friction=0.01)
                if end < 0:
                    resolved_end = geodesic.nimages + end
                else:
                    resolved_end = end
                x0 = geodesic.path[start:resolved_end].ravel().copy()

                geodesic.compute_target_func(
                    start=start, end=end, x0=x0, friction=0.01)

                self.assertTrue(sparse.isspmatrix_csr(geodesic.grad))
                sparse_bytes = (_sparse_bytes(geodesic.grad) +
                                _cached_derivative_bytes(geodesic))
                dense_jacobian_bytes = (
                    geodesic.grad.shape[0] * geodesic.grad.shape[1] * 8)
                self.assertLess(sparse_bytes, max_sparse_bytes)
                self.assertGreater(dense_jacobian_bytes / sparse_bytes, 10)

    def test_smooth_uses_sparse_derivative_wrapper(self):
        rng = np.random.default_rng(23)
        atoms = ["C", "H", "O", "N"]
        path = rng.normal(size=(4, len(atoms), 3))
        geodesic = Geodesic(atoms, path, scaler=1.0, threshold=10.0, min_neighbors=0)

        def fail_dense_derivative(*args, **kwargs):
            raise AssertionError("dense target_deriv should not be used by smooth")

        geodesic.target_deriv = fail_dense_derivative
        geodesic.smooth(tol=1e-12, max_iter=1, start=1, end=3)

    def test_invalid_optimization_segments_raise_clear_error(self):
        rng = np.random.default_rng(29)
        atoms = ["C", "H", "O"]
        path = rng.normal(size=(4, len(atoms), 3))
        geodesic = Geodesic(atoms, path, scaler=1.0, threshold=10.0, min_neighbors=0)

        for start, end in [(0, 2), (1, 4), (2, 2)]:
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, "1 <= start < end <= nimages - 1"):
                    geodesic.compute_target_func(start=start, end=end)


if __name__ == "__main__":
    unittest.main()
