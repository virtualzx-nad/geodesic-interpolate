"""Check that omitted distance coordinates do not hide atom overlaps."""

import numpy as np
from scipy.spatial import KDTree

from .coord_utils import COVALENT_RADIUS


class UnsafePathError(ValueError):
    """A path contains nonfinite coordinates or an omitted-pair overlap."""


class OverlapChecker:
    """Validate images and arithmetic midpoints against omitted-pair overlaps.

    Pairs included in the internal-coordinate metric are exempt. For every
    omitted pair, the minimum distance in Angstrom is ``max(0.70, 0.60 *
    (radius_i + radius_j))``. Without atom symbols, it is 0.70 Angstrom.
    Radii are from Cordero et al. (2008), covering H through Cm. Symbols beyond
    Cm and unrecognized symbols use a 1.5 Angstrom fallback. These screening
    radii are separate from the historical coordinate metric's scaling radii.
    The pair set and radii are cached for repeated optimization evaluations.
    """

    def __init__(self, natoms, rij_list, atoms=None):
        self.natoms = natoms
        self.included_pairs = {tuple(sorted(pair)) for pair in rij_list}
        if atoms is None:
            self.radii = None
            self.search_radius = 0.70
        else:
            if len(atoms) != natoms:
                raise ValueError("The number of atom symbols must match the path")
            self.radii = np.array([
                COVALENT_RADIUS.get(atom.capitalize(), 1.5) for atom in atoms
            ])
            self.search_radius = max(0.70, 1.20 * self.radii.max()) if natoms else 0.70

    def _validate_geometry(self, geometry, location):
        # Only nearby pairs are considered; no dense natoms-by-natoms array is
        # needed, even when the internal-coordinate pair list is sparse.
        candidates = KDTree(geometry).query_pairs(self.search_radius)
        for i, j in sorted(candidates - self.included_pairs):
            minimum = (0.70 if self.radii is None else
                       max(0.70, 0.60 * (self.radii[i] + self.radii[j])))
            distance = np.linalg.norm(geometry[i] - geometry[j])
            if distance < minimum:
                raise UnsafePathError(
                    "Nonbonded overlap for omitted atom pair ({}, {}) at {}: "
                    "distance {:.6f} Angstrom is below {:.6f} Angstrom "
                    "(zero-based indices)".format(i, j, location, distance, minimum))

    def validate(self, path, image_offset=0):
        """Reject unsafe images or midpoints, with optional full-path indices."""
        path = np.asarray(path)
        if path.ndim != 3 or path.shape[1:] != (self.natoms, 3):
            raise ValueError("The path must have shape (nimages, natoms, 3)")
        if not np.isfinite(path).all():
            raise UnsafePathError("The path must contain only finite coordinates")
        for index, geometry in enumerate(path):
            image_index = image_offset + index
            self._validate_geometry(geometry, "image {}".format(image_index))
            if index:
                midpoint = (path[index - 1] + geometry) * 0.5
                self._validate_geometry(
                    midpoint, "midpoint between images {} and {}".format(image_index - 1, image_index))


def validate_nonbonded_overlaps(path, rij_list, atoms=None):
    """Convenience wrapper for validating a single path."""
    path = np.asarray(path)
    if path.ndim != 3:
        raise ValueError("The path must have shape (nimages, natoms, 3)")
    OverlapChecker(path.shape[1], rij_list, atoms).validate(path)
