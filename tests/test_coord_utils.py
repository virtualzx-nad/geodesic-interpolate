import unittest

import numpy as np

from geodesic_interpolate.coord_utils import get_bond_list


class CoordUtilsTest(unittest.TestCase):
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
