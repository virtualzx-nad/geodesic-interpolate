"""Simplified geodesic interpolations module, which uses geodesic lengths as criteria
to add bisection points until point count meet desired number.
Will need another following geodesic smoothing to get final path.
"""
import logging

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from .geodesic import Geodesic
from .coord_utils import PairCoordinates, get_bond_list, morse_scaler, align_geom, align_path
from .validation import UnsafePathError


logger = logging.getLogger(__name__)


class _MidpointObjective:
    """Share one coordinate evaluation between the least-squares callbacks."""

    def __init__(self, coordinates, scaler, reference, x0, friction):
        self.coordinates = coordinates
        self.scaler = scaler
        self.reference = reference
        self.x0 = x0.copy()
        self.friction = friction
        self.use_sparse = x0.size > 100
        self.x = self.value = self.jacobian = None
        if self.use_sparse:
            # The friction identity adds one entry to each of its rows.
            index_type = coordinates.indices.dtype
            self.indices = np.concatenate((coordinates.indices, np.arange(x0.size, dtype=index_type)))
            self.indptr = np.concatenate((coordinates.indptr,
                                          coordinates.indptr[-1] + np.arange(1, x0.size + 1, dtype=index_type)))
            self.friction_data = np.full(x0.size, friction)
        else:
            self.friction_jacobian = np.eye(x0.size) * friction

    def _evaluate(self, x):
        if self.x is not None and np.array_equal(x, self.x):
            return
        wx, derivative = self.coordinates.compute(x, self.scaler, sparse_output=self.use_sparse)
        self.value = np.concatenate((wx - self.reference, (x - self.x0) * self.friction))
        if self.use_sparse:
            data = np.concatenate((derivative.data, self.friction_data))
            self.jacobian = sparse.csr_matrix((data, self.indices, self.indptr),
                                              shape=(self.value.size, x.size), copy=False)
            self.jacobian.has_sorted_indices = True
        else:
            self.jacobian = np.vstack((derivative, self.friction_jacobian))
        self.x = x.copy()

    def residual(self, x):
        self._evaluate(x)
        return self.value.copy()

    def derivative(self, x):
        self._evaluate(x)
        return self.jacobian.copy()


def mid_point(atoms, geom1, geom2, tol=1e-2, nudge=0.01, threshold=4):
    """Find the Cartesian geometry that has internal coordinate values closest to the average of
    two geometries.

    Simply perform a least-squares minimization on the difference between the current internal
    and the average of the two end points.  This is done twice, using either end point as the
    starting guess.  DON'T USE THE CARTESIAN AVERAGE AS GUESS, THINGS WILL BLOW UP.

    This is used to generate an initial guess path for the later smoothing routine.
    Genenrally, the added point may not be continuous with the both end points, but
    provides a good enough starting guess.

    Random nudges are added to the initial geometry, so running multiple times may not yield
    the same converged geometry. For larger systems, one will never get the same geometry
    twice.  So one may want to perform multiple runs and check which yields the best result.

    Args:
        geom1, geom2:   Cartesian geometry of the end points
        tol:    Convergence tolarnce for the least-squares minimization process
        nudge:  Random nudges added to the initial geometry, which helps to discover different
                solutions.  Also helps in cases where optimal paths break the symmetry.
        threshold:  Threshold for including an atom-pair in the coordinate system

    Returns:
        Optimized mid-point which bisects the two endpoints in internal coordinates
    """
    # Process the initial geometries, construct coordinate system and obtain average internals
    geom1, geom2 = np.array(geom1), np.array(geom2)
    add_pair = set()
    geom_list = [geom1, geom2]
    # This loop is for ensuring a sufficient large coordinate system.  The interpolated point may
    # have atom pairs in contact that are far away at both end-points, which may cause collision.
    # One can include all atom pairs, but this may blow up for large molecules.  Here the compromise
    # is to use a screened list of atom pairs first, then add more if additional atoms come into
    # contant, then rerun the minimization until the coordinate system is consistant with the
    # interpolated geometry
    while True:
        rijlist, re = get_bond_list(geom_list, threshold=threshold + 1, enforce=add_pair)
        scaler = morse_scaler(alpha=0.7, re=re)
        coordinates = PairCoordinates(len(geom1), rijlist)
        w1, _ = coordinates.compute(geom1, scaler)
        w2, _ = coordinates.compute(geom2, scaler)
        w = (w1 + w2) / 2
        d_min, x_min = np.inf, None
        unsafe_error = None
        friction = 0.1 / np.sqrt(geom1.shape[0])

        # The inner loop performs minimization using either end-point as the starting guess.
        for coef in [0.02, 0.98]:
            x0 = (geom1 * coef + (1 - coef) * geom2).ravel()
            x0 += nudge * np.random.random_sample(x0.shape)
            logger.debug('Starting least-squares minimization of bisection point at %7.2f.', coef)
            objective = _MidpointObjective(coordinates, scaler, w, x0, friction)
            result = least_squares(objective.residual, x0, objective.derivative, ftol=tol, gtol=tol)
            _, x_mid = align_geom(geom1, result['x'].reshape(-1, 3))
            # Take the interpolated geometry, construct new pair list and check for new contacts
            new_list = geom_list + [x_mid]
            new_rij, _ = get_bond_list(new_list, threshold=threshold, min_neighbors=0)
            extras = set(new_rij) - set(rijlist)
            if extras: 
                logger.info('  Screened pairs came into contact. Adding reference point.')
                # Update pair list then go back to the minimization loop if new contacts are found
                geom_list = new_list
                add_pair |= extras
                break
            # Perform local geodesic optimization for the new image.
            try:
                smoother = Geodesic(atoms, [geom1, x_mid, geom2], 0.7, threshold=threshold,
                                    log_level=logging.DEBUG, friction=1, align=False)
            except UnsafePathError as error:
                unsafe_error = error
                logger.debug("Rejecting unsafe bisection candidate: %s", error)
                continue
            smoother.compute_disps()
            width = max([np.sqrt(np.mean((g - smoother.path[1]) ** 2)) for g in [geom1, geom2]])
            dist, x_mid = width + smoother.length, smoother.path[1]
            logger.debug('  Trial path length: %8.3f after %d iterations', dist, result['nfev'])
            if dist < d_min:
                d_min, x_min = dist, x_mid
        else:   # Both starting guesses finished without new atom pairs.  Minimization successful
            if x_min is None:
                raise UnsafePathError("Neither bisection candidate produced a safe path: {}".format(unsafe_error)) from unsafe_error
            break
    return x_min


def redistribute(atoms, geoms, nimages, tol=1e-2):
    """Add or remove images so that the path length matches the desired number.

    If the number is too few, new points are added by bisecting the largest RMSD. If too numerous,
    one image is removed at a time so that the new merged segment has the shortest RMSD.

    Args:
        geoms:      Geometry of the original path.
        nimages:    The desired number of images
        tol:        Convergence tolerance for bisection.

    Returns:
        An aligned and redistributed path with has the correct number of images.
    """
    _, geoms = align_path(geoms)
    geoms = list(geoms)
    # If there are too few images, add bisection points
    while len(geoms) < nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[1:], geoms)]
        max_i = np.argmax(dists)
        logger.info("Inserting image between %d and %d with Cartesian RMSD %10.3f.  New length:%d",
                    max_i, max_i + 1, dists[max_i], len(geoms) + 1)
        insertion = mid_point(atoms, geoms[max_i], geoms[max_i + 1], tol)
        geoms.insert(max_i + 1, insertion)
    # If there are too many images, remove points
    while len(geoms) > nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i = np.argmin(dists)
        logger.info("Removing image %d.  Cartesian RMSD of merged section %10.3f",
                    min_i + 1, dists[min_i])
        del geoms[min_i + 1]
    return geoms
