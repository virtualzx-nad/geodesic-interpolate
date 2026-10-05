import unittest

import numpy as np

from geodesic_interpolate.validation import (
    OverlapChecker, UnsafePathError, validate_nonbonded_overlaps)


def pair_path(distance):
    return np.array([[[0., 0., 0.], [distance, 0., 0.]]])


class OverlapCheckerTest(unittest.TestCase):
    def test_exact_distance_threshold_is_allowed_but_smaller_is_rejected(self):
        for atoms, minimum in [(None, .70), (["H", "H"], .70),
                               (["C", "C"], .60 * 1.52),
                               (["Na", "Na"], .60 * 3.32),
                               (["Ca", "Ca"], .60 * 3.52),
                               (["Og", "Og"], .60 * 3.0),
                               (["unlisted", "unlisted"], .60 * 3.0)]:
            with self.subTest(atoms=atoms):
                checker = OverlapChecker(2, [], atoms)
                checker.validate(pair_path(minimum))
                with self.assertRaisesRegex(UnsafePathError, "omitted atom pair \\(0, 1\\).*image 0"):
                    checker.validate(pair_path(minimum - 1e-8))

    def test_heterogeneous_pair_threshold_uses_both_radii(self):
        checker = OverlapChecker(2, [], ["Na", "H"])
        checker.validate(pair_path(1.20))
        with self.assertRaisesRegex(UnsafePathError, "1.182000"):
            checker.validate(pair_path(1.10))

    def test_calcium_carbon_guard_uses_published_radii(self):
        # Cordero Table 2: Ca=1.76, C(sp3)=0.76 Angstrom. The old 1.5
        # fallback for Ca incorrectly admitted this 1.4 Angstrom contact.
        checker = OverlapChecker(2, [], ["c", "CA"])
        np.testing.assert_array_equal(checker.radii, [0.76, 1.76])
        self.assertAlmostEqual(checker.search_radius, 2.112)
        checker.validate(pair_path(.60 * (0.76 + 1.76)))
        with self.assertRaisesRegex(UnsafePathError, "below 1.512000 Angstrom"):
            checker.validate(pair_path(1.4))

    def test_search_radius_covers_largest_published_elements(self):
        for symbol, radius in [("K", 2.03), ("Cs", 2.44), ("Fr", 2.60),
                               ("U", 1.96), ("Cm", 1.69)]:
            with self.subTest(symbol=symbol):
                checker = OverlapChecker(2, [], [symbol, symbol])
                minimum = .60 * (radius + radius)
                self.assertGreaterEqual(checker.search_radius, minimum)
                checker.validate(pair_path(minimum))
                with self.assertRaises(UnsafePathError):
                    checker.validate(pair_path(minimum - 1e-8))

    def test_included_pairs_are_exempt_in_either_order(self):
        for pairs in [[(0, 1)], [(1, 0)], np.array([[1, 0]])]:
            with self.subTest(pairs=pairs):
                checker = OverlapChecker(2, pairs, ["C", "C"])
                checker.validate(pair_path(0.0))

    def test_midpoint_only_overlap_is_rejected(self):
        path = np.array([[[0., 0., 0.], [2., 0., 0.]],
                         [[2., 0., 0.], [0., 0., 0.]]])
        checker = OverlapChecker(2, [], ["C", "C"])
        for image in path:
            checker.validate(image[None])
        with self.assertRaisesRegex(UnsafePathError, "midpoint between images 0 and 1"):
            checker.validate(path)
        OverlapChecker(2, [(0, 1)], ["C", "C"]).validate(path)

    def test_all_images_and_last_midpoint_are_checked(self):
        safe = pair_path(2.)[0]
        bad = pair_path(.2)[0]
        with self.assertRaisesRegex(UnsafePathError, "image 3"):
            OverlapChecker(2, []).validate(np.array([safe, safe, safe, bad]))
        swapped = safe[::-1].copy()
        with self.assertRaisesRegex(UnsafePathError, "midpoint between images 2 and 3"):
            OverlapChecker(2, []).validate(np.array([safe, safe, safe, swapped]))

    def test_nonfinite_coordinates_rejected_even_for_included_pairs(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                path = pair_path(2.)
                path[0, 0, 0] = value
                with self.assertRaisesRegex(UnsafePathError, "finite coordinates"):
                    OverlapChecker(2, [(0, 1)]).validate(path)

    def test_partial_path_errors_include_global_image_indices(self):
        checker = OverlapChecker(2, [])
        with self.assertRaisesRegex(UnsafePathError, "image 7"):
            checker.validate(pair_path(.1), image_offset=7)
        path = np.array([[[0., 0., 0.], [2., 0., 0.]],
                         [[2., 0., 0.], [0., 0., 0.]]])
        with self.assertRaisesRegex(UnsafePathError, "midpoint between images 7 and 8"):
            checker.validate(path, image_offset=7)

    def test_invalid_layout_has_clear_error(self):
        checker = OverlapChecker(2, [])
        for path in [np.zeros((2, 3)), np.zeros((3, 4, 3)), np.zeros((3, 2, 2))]:
            with self.subTest(shape=path.shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    checker.validate(path)
        with self.assertRaisesRegex(ValueError, "atom symbols"):
            OverlapChecker(2, [], ["C"])

    def test_wrapper_does_not_modify_path(self):
        path = np.repeat(pair_path(2.), 3, axis=0)
        original = path.copy()
        validate_nonbonded_overlaps(path, [], ["c", "h"])
        np.testing.assert_array_equal(path, original)

    def test_kdtree_checks_match_independent_all_pairs_reference(self):
        rng = np.random.default_rng(112)
        atoms = ["H", "C", "O", "Na", "Cl", "K", "Ca", "Fe", "Ce", "U", "Fr", "Cm", "Unknown"]
        # Literal published radii keep this oracle independent of the lookup.
        radii = np.array([.31, .76, .66, 1.66, 1.02, 2.03, 1.76, 1.32, 2.04, 1.96, 2.60, 1.69, 1.5])
        pairs = {(0, 1), (0, 3), (2, 5)}
        checker = OverlapChecker(len(atoms), pairs, atoms)
        for scale in [1., 2., 4.]:
            for trial in range(12):
                path = rng.normal(size=(4, len(atoms), 3)) * scale
                geometries = np.concatenate([path, .5 * (path[:-1] + path[1:])])
                expected_unsafe = any(
                    np.linalg.norm(geometry[i] - geometry[j]) < max(.70, .60 * (radii[i] + radii[j]))
                    for geometry in geometries for i in range(len(atoms))
                    for j in range(i + 1, len(atoms)) if (i, j) not in pairs)
                with self.subTest(scale=scale, trial=trial):
                    if expected_unsafe:
                        with self.assertRaises(UnsafePathError):
                            checker.validate(path)
                    else:
                        checker.validate(path)


if __name__ == "__main__":
    unittest.main()
