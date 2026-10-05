import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from geodesic_interpolate.coord_utils import (
    ATOMIC_RADIUS, COVALENT_RADIUS, PairCoordinates, compute_rij, compute_rij_sparse, compute_wij,
    compute_wij_sparse, get_bond_list, morse_scaler)


class CoordUtilsTest(unittest.TestCase):
    def test_overlap_radii_cover_every_published_element_through_curium(self):
        symbols = """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca
            Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo
            Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu
            Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po
            At Rn Fr Ra Ac Th Pa U Np Pu Am Cm""".split()
        self.assertEqual(set(COVALENT_RADIUS), set(symbols))
        self.assertEqual(len(COVALENT_RADIUS), 96)
        self.assertEqual(COVALENT_RADIUS["Ca"], 1.76)
        self.assertTrue(all(radius > 0 for radius in COVALENT_RADIUS.values()))

    def test_overlap_radius_update_preserves_legacy_metric_scaling(self):
        geometry = np.array([[0., 0., 0.], [2., 0., 0.]])
        pairs, reference = get_bond_list(geometry, atoms=["C", "Ca"], min_neighbors=0)
        self.assertEqual(pairs, [(0, 1)])
        self.assertNotIn("Ca", ATOMIC_RADIUS)
        np.testing.assert_allclose(reference, [.76 + 1.5])
        self.assertEqual({key: COVALENT_RADIUS[key] for key in ATOMIC_RADIUS}, ATOMIC_RADIUS)

    def test_pair_coordinates_match_independent_values_and_gradients(self):
        rng = np.random.default_rng(14)
        geom = rng.normal(size=(7, 3))
        pairs = [(4, 0), (2, 6), (3, 1), (0, 4), (-1, 0)]
        expected_rij = []
        expected_gradient = np.zeros((len(pairs), len(geom), 3))
        for row, (i, j) in enumerate(pairs):
            delta = geom[i] - geom[j]
            distance = np.sqrt(np.sum(delta ** 2))
            expected_rij.append(distance)
            expected_gradient[row, i] = delta / distance
            expected_gradient[row, j] = -delta / distance
        expected_rij = np.array(expected_rij)
        scaler = morse_scaler(alpha=0.7)
        expected_wij, scale = scaler(expected_rij)

        rij, gradient = compute_rij(geom, pairs)
        np.testing.assert_allclose(rij, expected_rij)
        np.testing.assert_allclose(gradient, expected_gradient)
        rij_sparse, gradient_sparse = compute_rij_sparse(geom, pairs)
        self.assertTrue(sparse.isspmatrix_csr(gradient_sparse))
        self.assertTrue(gradient_sparse.has_sorted_indices)
        np.testing.assert_allclose(rij_sparse, expected_rij)
        np.testing.assert_allclose(gradient_sparse.toarray(), expected_gradient.reshape(len(pairs), -1))
        for evaluator in (compute_wij, compute_wij_sparse):
            wij, gradient = evaluator(geom.ravel(), pairs, scaler)
            if sparse.issparse(gradient):
                gradient = gradient.toarray()
            np.testing.assert_allclose(wij, expected_wij)
            np.testing.assert_allclose(gradient, (expected_gradient * scale[:, None, None]).reshape(len(pairs), -1))

    def test_pair_gradient_matches_directional_difference(self):
        rng = np.random.default_rng(81)
        geom = rng.normal(size=(9, 3))
        direction = rng.normal(size=geom.shape)
        coordinates = PairCoordinates(len(geom), [(0, 8), (7, 3), (1, 4)])
        for scaler in (None, morse_scaler(alpha=1.2)):
            _, gradient = coordinates.compute(geom, scaler)
            eps = 1e-6
            numeric = (coordinates.compute(geom + eps * direction, scaler)[0] -
                       coordinates.compute(geom - eps * direction, scaler)[0]) / (2 * eps)
            np.testing.assert_allclose(gradient @ direction.ravel(), numeric, rtol=2e-8, atol=1e-9)

    def test_pair_layout_is_reused_and_values_are_independent(self):
        geom = np.array([[0., 0., 0.], [1., 2., 3.], [2., -1., 1.]])
        coordinates = PairCoordinates(3, [(2, 0), (0, 1)])
        _, first = coordinates.compute(geom)
        first_values = first.data.copy()
        geom[1, 0] += 0.2
        _, second = coordinates.compute(geom)
        self.assertTrue(np.shares_memory(first.indices, coordinates.indices))
        self.assertTrue(np.shares_memory(first.indptr, coordinates.indptr))
        self.assertTrue(np.shares_memory(second.indices, coordinates.indices))
        self.assertFalse(np.shares_memory(first.data, second.data))
        np.testing.assert_array_equal(first.data, first_values)

    def test_empty_selected_pairs(self):
        geom = np.zeros((3, 3))
        for evaluator, args, expected_shape in (
                (compute_rij, (), (0, 3, 3)),
                (compute_rij_sparse, (), (0, 9)),
                (compute_wij, (morse_scaler(),), (0, 9)),
                (compute_wij_sparse, (morse_scaler(),), (0, 9))):
            with self.subTest(evaluator=evaluator.__name__):
                values, gradient = evaluator(geom, [], *args)
                self.assertEqual(values.shape, (0,))
                self.assertEqual(gradient.shape, expected_shape)

    def test_only_selected_coincident_pairs_raise_clear_error(self):
        geom = np.array([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.]])
        for evaluator, args in ((compute_rij, ()), (compute_rij_sparse, ()),
                                (compute_wij, (morse_scaler(),)),
                                (compute_wij_sparse, (morse_scaler(),))):
            with self.subTest(evaluator=evaluator.__name__):
                with self.assertRaisesRegex(ValueError, r"coincident atoms.*\(0, 1\)"):
                    evaluator(geom, [(1, 0)], *args)
                values, _ = evaluator(geom, [(0, 2)], *args)
                self.assertTrue(np.all(np.isfinite(values)))

    def test_get_bond_list_samples_across_full_path(self):
        geom = np.tile(np.array([[0., 0., 0.], [10., 0., 0.]]), (10, 1, 1))
        geom[8, 1, 0] = 1.0
        with patch("geodesic_interpolate.coord_utils.np.random.choice", return_value=[8]) as choice:
            pairs, _ = get_bond_list(geom, threshold=2, min_neighbors=0, snapshots=3, bond_threshold=0)
        self.assertEqual(list(choice.call_args.args[0]), list(range(1, 9)))
        self.assertIn((0, 1), pairs)

    def test_get_bond_list_min_neighbors_uses_final_frame_tree(self):
        geom = np.array([
            [[100.0, 100.0, 0.0], [101.0, 100.0, 0.0],
             [102.0, 100.0, 0.0], [103.0, 100.0, 0.0]],
            [[100.0, 0.0, 0.0], [101.0, 0.0, 0.0],
             [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        ])

        pairs, _ = get_bond_list(
            geom,
            threshold=0.1,
            bond_threshold=0.1,
            min_neighbors=1,
            snapshots=3,
        )

        self.assertEqual(
            [(int(i), int(j)) for i, j in pairs],
            [(0, 1), (2, 3)],
        )


if __name__ == "__main__":
    unittest.main()
