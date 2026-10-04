"""Geodesic smoothing.   Minimize the path length using redundant internal coordinate
metric to find geodesics directly in Cartesian, to avoid feasibility problems associated
with redundant internals.
"""
import logging
from numbers import Real

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from .coord_utils import (
    align_path, PairCoordinates, get_bond_list, morse_scaler)
from .validation import OverlapChecker, UnsafePathError


logger = logging.getLogger(__name__)


class Geodesic(object):
    """Optimizer to obtain geodesic in redundant internal coordinates.  Core part is the calculation
    of the path length in the internal metric."""
    def __init__(self, atoms, path, scaler=1.7, threshold=3.0, min_neighbors=4, log_level=logging.INFO,
                 friction=1e-3, align=True):
        """Initialize the interpolater
        Args:
            atoms:      Atom symbols, used to lookup radii
            path:       Initial geometries of the path, must be of dimension `nimage * natoms * 3`
            scaler:     Either the alpha parameter for morse potential, or an explicit scaling function.
                        It is easier to get smoother paths with small number of data points using small
                        scaling factors, as they have large range, but larger values usually give
                        better energetics because they better represent the (sharp) energy landscape.
            threshold:  Distance cut-off for constructing inter-nuclear distance coordinates.  Note that
                        any atoms linked by three or less bonds will also be added.
            min_neighbors:  Minimum number of neighbors an atom must have in the atom pair list.
            log_level:  Logging level to use.
            friction:   Friction term in the target function which regularizes the optimization step
                        size to prevent explosion.
            align:      Align the input once. Set False for an already prepared path.
        """
        path = np.array(path, dtype=float, copy=True)
        if path.ndim != 3 or path.shape[2] != 3 or min(path.shape[:2]) < 1:
            raise ValueError('The path must have shape (nimages, natoms, 3)')
        if not np.isfinite(path).all():
            raise UnsafePathError('The path must contain only finite coordinates')
        if align:
            rmsd0, path = align_path(path)
            logger.log(log_level, "Maximum RMSD change in initial path: %10.2f", rmsd0)
        self.path = path
        self.nimages, self.natoms, _ = self.path.shape
        # Construct coordinates
        self.rij_list, self.re = get_bond_list(self.path, atoms, threshold=threshold, min_neighbors=min_neighbors)
        if isinstance(scaler, Real):
            self.scaler = morse_scaler(re=self.re, alpha=scaler)
        else:
            self.scaler = scaler
        self.nrij = len(self.rij_list)
        self.coordinates = PairCoordinates(self.natoms, self.rij_list)
        self.overlap_checker = OverlapChecker(self.natoms, self.rij_list, atoms)
        self._validate_path(self.path)
        self._layouts = {}
        self.rejected_trials = 0
        self.friction = friction
        # Initalize interal storages for mid points, internal coordinates and B matrices
        logger.log(log_level, "Performing geodesic smoothing")
        logger.log(log_level, "  Images: %4d  Atoms %4d Rijs %6d", self.nimages, self.natoms, len(self.rij_list))
        self.neval = 0
        self.w = [None] * len(path)
        self.dwdR = [None] * len(path)
        self.X_mid = [None] * (len(path) - 1)
        self.w_mid = [None] * (len(path) - 1)
        self.dwdR_mid = [None] * (len(path) - 1)
        self.disps = self.grad = self.segment = None
        self._target_x0 = None
        self._target_friction = None
        self.conv_path = []

    def _resolve_segment(self, start, end):
        """Resolve and validate the optimized interior image range."""
        if end < 0:
            end += self.nimages
        if start < 1 or end > self.nimages - 1 or start >= end:
            raise ValueError(
                "Optimization segment must satisfy 1 <= start < end <= nimages - 1")
        return start, end

    def update_intc(self):
        """Adjust unknown locations of mid points and compute missing values of internal coordinates
        and their derivatives.  Any missing values will be marked with None values in internal storage,
        and this routine finds and calculates them.  This is to avoid redundant evaluation of value and
        gradients of internal coordinates."""
        for i, (X, w, dwdR) in enumerate(zip(self.path, self.w, self.dwdR)):
            if w is None:
                self.w[i], self.dwdR[i] = self.coordinates.compute(X, self.scaler)
        for i, (X0, X1, w) in enumerate(zip(self.path, self.path[1:], self.w_mid)):
            if w is None:
                self.X_mid[i] = Xm = (X0 + X1) / 2
                self.w_mid[i], self.dwdR_mid[i] = self.coordinates.compute(Xm, self.scaler)

    def _validate_path(self, path, image_offset=0):
        """Check sampled locations before changing any cached geometry."""
        self.overlap_checker.validate(path, image_offset=image_offset)
        pairs = self.coordinates.pairs
        for geometry in list(path) + list((path[:-1] + path[1:]) * 0.5):
            if np.any(np.all(geometry[pairs[:, 0]] == geometry[pairs[:, 1]], axis=1)):
                raise UnsafePathError('Cannot evaluate coincident atoms in a selected pair')

    def update_geometry(self, X, start, end):
        """Validate a trial, then update its geometry and invalidate affected caches."""
        X = np.asarray(X).reshape(self.path[start:end].shape)
        if np.array_equal(X, self.path[start:end]):
            return False
        candidate = self.path.copy()
        candidate[start:end] = X
        # Only these images and their adjoining midpoints can have changed.
        self._validate_path(candidate[start - 1:end + 1], image_offset=start - 1)
        self.path[start:end] = X
        for i in range(start, end):
            self.w_mid[i] = self.w[i] = None
        self.w_mid[start - 1] = None
        self.segment = None
        return True

    def compute_disps(self, start=1, end=-1, dx=None, friction=1e-3):
        """Compute displacement vectors and total length between two images.
        Only recalculate internal coordinates if they are unknown."""
        start, end = self._resolve_segment(start, end)
        self.update_intc()
        # Calculate displacement vectors in each segment, and the total length
        vecs_l = [wm - wl for wl, wm in zip(self.w[start - 1:end], self.w_mid[start - 1:end])]
        vecs_r = [wr - wm for wr, wm in zip(self.w[start:end + 1], self.w_mid[start - 1:end])]
        self.length = np.sum(np.linalg.norm(vecs_l, axis=1)) + np.sum(np.linalg.norm(vecs_r, axis=1))
        if dx is None:
            trans = np.zeros(self.path[start:end].size)
        else:
            trans = friction * dx  # Translation from initial geometry.  friction term 
        self.disps = np.concatenate(vecs_l + vecs_r + [trans])
        self.disps0 = self.disps[:len(vecs_l) * 2 * self.nrij]

    def _jacobian_layout(self, nimages, with_friction):
        """Cache CSR assembly indices for the four blocks touching each image."""
        key = nimages, with_friction
        if key not in self._layouts:
            ncoords = self.natoms * 3
            nsegments = nimages + 1
            pair_rows = np.repeat(np.arange(self.nrij), 6)
            rows, cols = [], []
            for i in range(nimages):
                for block in (i + 1, i, nsegments + i + 1, nsegments + i):
                    rows.append(pair_rows + block * self.nrij)
                    cols.append(self.coordinates.indices + i * ncoords)
            nvars = nimages * ncoords
            nres = 2 * nsegments * self.nrij + nvars
            if with_friction:
                rows.append(2 * nsegments * self.nrij + np.arange(nvars))
                cols.append(np.arange(nvars))
            rows, cols = np.concatenate(rows), np.concatenate(cols)
            # The template's values encode the permutation from block data to CSR.
            template = sparse.coo_matrix((np.arange(len(rows)), (rows, cols)),
                                         shape=(nres, nvars)).tocsr()
            self._layouts[key] = (template.data, template.indices, template.indptr, template.shape)
        return self._layouts[key]

    def compute_disp_grad(self, start, end, friction=1e-3):
        """Assemble the local-support Jacobian using a reusable sparse layout."""
        start, end = self._resolve_segment(start, end)
        self.update_intc()
        data = []
        for image in range(start, end):
            dmid1 = self.dwdR_mid[image - 1].data * 0.5
            dmid2 = self.dwdR_mid[image].data * 0.5
            deriv = self.dwdR[image].data
            data.extend((dmid2 - deriv, dmid1, -dmid2, deriv - dmid1))
        if friction:
            data.append(np.full((end - start) * self.natoms * 3, friction))
        order, indices, indptr, shape = self._jacobian_layout(end - start, bool(friction))
        self.grad = sparse.csr_matrix((np.concatenate(data)[order], indices, indptr), shape=shape)
        self.grad0 = self.grad[:(end - start + 1) * 2 * self.nrij]

    def compute_target_func(self, X=None, start=1, end=-1, log_level=logging.INFO, x0=None, friction=1e-3):
        """Compute the vectorized target function, which is then used for least
        squares minimization."""
        start, end = self._resolve_segment(start, end)
        x0_array = None if x0 is None else np.asarray(x0).ravel()
        same_geometry = X is not None and not self.update_geometry(X, start, end)
        same_x0 = (x0_array is None and self._target_x0 is None) or (
            x0_array is not None and self._target_x0 is not None and
            np.array_equal(x0_array, self._target_x0))
        if (same_geometry and self.segment == (start, end) and
                friction == self._target_friction and same_x0):
            return
        self.segment = start, end
        self._target_friction = friction
        self._target_x0 = None if x0_array is None else x0_array.copy()
        dx = (np.zeros(self.path[start:end].size) if x0_array is None
              else self.path[start:end].ravel() - x0_array)
        self.compute_disps(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        weighted = self.disps / np.hypot(1, self.disps)
        self.optimality = np.linalg.norm((self.grad.T @ weighted).ravel(), ord=np.inf)
        # Equivalent to sum(hypot(1, f) - 1), without cancellation near zero.
        self.cost = np.sum(self.disps * weighted / (1 + 1 / np.hypot(1, self.disps)))
        logger.log(log_level, "  Iteration %3d: Length %10.3f |gradient|=%7.3e", self.neval, self.length, self.optimality)
        self.conv_path.append(self.path[1].copy())
        self.neval += 1

    def _solver_target(self, X, **kwargs):
        """Keep unsafe trials out of both the geometry and derivative caches."""
        try:
            self.compute_target_func(X, **kwargs)
            return True
        except UnsafePathError as error:
            self.rejected_trials += 1
            logger.debug("Rejecting unsafe trial: %s", error)
            # Public callers may change segment, reference or friction between
            # calls. Derive the penalty shape from the requested safe target.
            self.compute_target_func(**kwargs)
            return False

    def target_func(self, X, **kwargs):
        """Return a private residual copy; robust solvers may modify their input."""
        if self._solver_target(X, **kwargs):
            return self.disps.copy()
        return np.full(self.disps.shape, 1e20)

    def target_deriv(self, X, **kwargs):
        """Dense Jacobian wrapper kept for compatibility with external callers."""
        if self._solver_target(X, **kwargs):
            return self.grad.toarray()
        return np.zeros(self.grad.shape)

    def target_deriv_sparse(self, X, **kwargs):
        """Sparse Jacobian wrapper; do not expose the cached matrix to the solver."""
        if self._solver_target(X, **kwargs):
            return self.grad.copy()
        return sparse.csr_matrix(self.grad.shape)

    def smooth(self, tol=1e-3, max_iter=50, start=1, end=-1, log_level=logging.INFO, friction=None,
               xref=None):
        """Minimize the soft-L1 displacement objective with fixed prepared endpoints.

        ``tol`` bounds the infinity norm of the robust objective gradient.
        ``max_iter`` is the maximum number of residual evaluations. Small
        problems (at most 100 Cartesian variables) use dense factorization;
        larger problems use sparse Jacobians and an iterative solver.
        """
        start, end = self._resolve_segment(start, end)
        X0 = self.path[start:end].ravel().copy()
        xref = X0 if xref is None else np.asarray(xref).ravel().copy()
        if friction is None:
            friction = self.friction
        kwargs = dict(start=start, end=end, log_level=log_level, x0=xref, friction=friction)
        self.compute_target_func(**kwargs)
        if self.optimality > tol and max_iter > 0:
            derivative = self.target_deriv if X0.size <= 100 else self.target_deriv_sparse
            try:
                result = least_squares(
                    self.target_func, X0, derivative,
                    # SciPy 0.19 (the supported minimum) requires numeric
                    # tolerances. Avoid early cost/step stopping at user tol.
                    ftol=np.finfo(float).eps, xtol=np.finfo(float).eps, gtol=tol,
                    max_nfev=max_iter, kwargs=kwargs, loss='soft_l1')
                try:
                    self.update_geometry(result['x'], start, end)
                except UnsafePathError:
                    self.update_geometry(X0, start, end)
                    logger.warning("Rejected unsafe smoothing result; restored the initial segment.")
            except BaseException:
                # A solver error or interruption must not leave its last trial behind.
                self.update_geometry(X0, start, end)
                self.compute_target_func(**kwargs)
                raise
        # The last evaluated trial need not be the accepted result. Recompute
        # length and the same robust objective at the actual returned geometry.
        self.compute_target_func(**kwargs)
        if self.optimality > tol:
            logger.log(logging.WARNING if log_level >= logging.INFO else log_level,
                       "Smoothing did not converge: |gradient|=%.6g exceeds tolerance %.6g",
                       self.optimality, tol)
        else:
            logger.log(log_level, "Smoothing converged: |gradient|=%.6g", self.optimality)
        logger.log(log_level, "Final path length: %12.5f", self.length)
        return self.path

    def sweep(self, tol=1e-3, max_iter=50, micro_iter=20, start=1, end=-1):
        """Optimize every interior image in alternating forward/backward sweeps.

        Each local solve uses the same reference and friction as the full-path
        objective. Convergence is checked with its full robust gradient after
        each sweep; local subproblem gradients cannot establish convergence.
        Sweeping can require more evaluations than global smoothing.
        """
        start, end = self._resolve_segment(start, end)
        reference = self.path.copy()
        kwargs = dict(start=start, end=end, x0=reference[start:end].ravel(),
                      friction=self.friction, log_level=logging.DEBUG)
        self.compute_target_func(**kwargs)
        images = list(range(start, end))
        curr_tol = max(tol * 0.5, self.optimality * 0.1)
        for iteration in range(max_iter):
            if self.optimality <= tol:
                break
            for i in images:
                self.smooth(curr_tol, max_iter=micro_iter, start=i, end=i + 1,
                            log_level=logging.DEBUG, friction=self.friction,
                            xref=reference[i].ravel())
            self.compute_target_func(**kwargs)
            logger.info("Sweep %3d: Length=%.6f |gradient|=%.6g", iteration + 1,
                        self.length, self.optimality)
            curr_tol = max(tol * 0.5, self.optimality * 0.1)
            images.reverse()
        self.compute_target_func(**kwargs)
        if self.optimality > tol:
            logger.warning("Sweeping did not converge: |gradient|=%.6g exceeds tolerance %.6g",
                           self.optimality, tol)
        else:
            logger.info("Sweeping converged: |gradient|=%.6g", self.optimality)
        logger.info("Final path length: %12.5f", self.length)
        return self.path
