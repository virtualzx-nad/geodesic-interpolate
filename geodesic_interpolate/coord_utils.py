"""Coordinate utilities used by the interpolation program"""
import logging

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree


logger = logging.getLogger(__name__)


def align_path(path):
    """Rotate and translate images to minimize RMSD movements along the path.
    Also moves the geometric center of all images to the origin.
    """
    path = np.array(path)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0
    for g, nextg in zip(path, path[1:]):
        rmsd, nextg[:] = align_geom(g, nextg)
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def align_geom(refgeom, geom):
    """Find translation/rotation that moves a given geometry to maximally overlap
    with a reference geometry. Implemented with Kabsch algorithm.

    Args:
        refgeom:    The reference geometry to be rotated to
        geom:       The geometry to be rotated and shifted

    Returns:
        RMSD:       Root-mean-squared difference between the rotated geometry
                    and the reference
        new_geom:   The rotated geometry that maximumally overal with the reference
    """
    center = np.mean(refgeom, axis=0)   # Find the geometric center
    ref2 = refgeom - center
    geom2 = geom - np.mean(geom, axis=0)
    cov = np.dot(geom2.T, ref2)
    v, sv, w = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(w) < 0:
        sv[-1] = -sv[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    new_geom = np.dot(geom2, u) + center
    rmsd = np.sqrt(np.mean((new_geom - refgeom) ** 2))
    return rmsd, new_geom


# Historical radii used to scale the interpolation metric. Keep this limited
# table and its 1.5 Angstrom fallback stable; overlap screening uses the
# separately maintained complete published table below.
ATOMIC_RADIUS = dict(H=0.31, He=0.28,
                     Li=1.28, Be=0.96, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Ne=0.58,
                     Na=1.66, Mg=1.41, Al=1.21, Si=1.11, P=1.07, S=1.05, Cl=1.02, Ar=1.06)


# Covalent radii in Angstrom, Cordero et al., Dalton Trans. (2008), Table 2,
# https://doi.org/10.1039/B801115J. The publication covers H through Cm (Z=96).
# Symbols alone do not specify hybridization/spin: use the sp3 carbon value
# and the first (low-spin) values listed for Mn, Fe and Co. Elements beyond
# Cm and unrecognized symbols retain the explicit 1.5 Angstrom fallback.
COVALENT_RADIUS = dict(
    H=0.31, He=0.28,
    Li=1.28, Be=0.96, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Ne=0.58,
    Na=1.66, Mg=1.41, Al=1.21, Si=1.11, P=1.07, S=1.05, Cl=1.02, Ar=1.06,
    K=2.03, Ca=1.76, Sc=1.70, Ti=1.60, V=1.53, Cr=1.39, Mn=1.39, Fe=1.32,
    Co=1.26, Ni=1.24, Cu=1.32, Zn=1.22, Ga=1.22, Ge=1.20, As=1.19, Se=1.20,
    Br=1.20, Kr=1.16,
    Rb=2.20, Sr=1.95, Y=1.90, Zr=1.75, Nb=1.64, Mo=1.54, Tc=1.47, Ru=1.46,
    Rh=1.42, Pd=1.39, Ag=1.45, Cd=1.44, In=1.42, Sn=1.39, Sb=1.39, Te=1.38,
    I=1.39, Xe=1.40,
    Cs=2.44, Ba=2.15, La=2.07, Ce=2.04, Pr=2.03, Nd=2.01, Pm=1.99, Sm=1.98,
    Eu=1.98, Gd=1.96, Tb=1.94, Dy=1.92, Ho=1.92, Er=1.89, Tm=1.90, Yb=1.87,
    Lu=1.87, Hf=1.75, Ta=1.70, W=1.62, Re=1.51, Os=1.44, Ir=1.41, Pt=1.36,
    Au=1.36, Hg=1.32, Tl=1.45, Pb=1.46, Bi=1.48, Po=1.40, At=1.50, Rn=1.50,
    Fr=2.60, Ra=2.21, Ac=2.15, Th=2.06, Pa=2.00, U=1.96, Np=1.90, Pu=1.87,
    Am=1.80, Cm=1.69)


def get_bond_list(geom, atoms=None, threshold=4, min_neighbors=4, snapshots=30, bond_threshold=1.8,
                  enforce=()):
    """Get the list of all the important atom pairs.
    Samples a number of snapshots from a list of geometries to generate all
    distances that are below a given threshold in any of them.

    Args:
        atoms:      Symbols for each atoms.
        geom:       One or a list of geometries to check for pairs
        threshold:  Threshold for including a bond in the bond list
        min_neighbors: Minimum number of neighbors to include for each atom.
                    If an atom has smaller than this number of bonds, additional
                    distances will be added to reach this number.
        snapshots:  Number of snapshots to be used in the generation, useful
                    for speeding up the process if the path is long and
                    atoms numerous.

    Returns:
        List of all the included interatomic distance pairs.
    """
    # Type casting and value checks on input parameters
    geom = np.asarray(geom)
    if len(geom.shape) < 3:
        # If there is only one geometry or it is flattened, promote to 3d
        geom = geom.reshape(1, -1, 3)
    min_neighbors = min(min_neighbors, geom.shape[1] - 1)

    # Determine which images to be used to determine distances
    snapshots = min(len(geom), snapshots)
    images = [0] if len(geom) == 1 else [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, len(geom) - 1), snapshots - 2, replace=False))
    # Get neighbor list for included geometry and merge them
    rijset = set(enforce)
    for image in images:
        image_tree = KDTree(geom[image])
        pairs = image_tree.query_pairs(threshold)
        rijset.update(pairs)
        bonded = image_tree.query_pairs(bond_threshold)
        neighbors = {i: {i} for i in range(geom.shape[1])}
        for i, j in bonded:
            neighbors[i].add(j)
            neighbors[j].add(i)
        for i, j in bonded:
            for ni in neighbors[i]:
                for nj in neighbors[j]:
                    if ni != nj:
                        pair = tuple(sorted([ni, nj]))
                        if pair not in rijset:
                            rijset.add(pair)
    rijlist = sorted(rijset)
    # Check neighbor count to make sure `min_neighbors` is satisfied
    count = np.zeros(geom.shape[1], dtype=int)
    for i, j in rijlist:
        count[i] += 1
        count[j] += 1
    final_geom = geom[-1]
    final_tree = KDTree(final_geom)
    for idx, ct in enumerate(count):
        if ct < min_neighbors:
            _, neighbors = final_tree.query(final_geom[idx], k=min_neighbors + 1)
            for i in neighbors:
                if i == idx:
                    continue
                pair = tuple(sorted([i, idx]))
                if pair in rijset:
                    continue
                else:
                    rijset.add(pair)
                    rijlist.append(pair)
                    count[i] += 1
                    count[idx] += 1
    if atoms is None:
        re = np.full(len(rijlist), 2.0)
    else:
        radius = np.array([ATOMIC_RADIUS.get(atom.capitalize(), 1.5) for atom in atoms])
        re = np.array([radius[i] + radius[j] for i, j in rijlist])
    logger.debug("Pair list contain %d pairs", len(rijlist))
    return rijlist, re


class PairCoordinates:
    """Evaluate selected pair coordinates using a reusable sparse row layout.

    Each row has six entries: the three Cartesian derivatives for each atom.
    Only those entries are computed, so a sparse evaluation does not allocate
    an array proportional to the number of pairs times the number of atoms.
    """

    def __init__(self, natoms, rij_list):
        self.natoms = natoms
        self.pairs = np.asarray(rij_list, dtype=int).reshape(-1, 2).copy()
        # Support the negative atom indices accepted by NumPy indexing, and
        # sort within each pair so the CSR columns are already ordered.
        self.pairs[self.pairs < 0] += natoms
        if np.any(self.pairs < 0) or np.any(self.pairs >= natoms):
            raise IndexError("Atom pair index is outside the geometry")
        self.pairs.sort(axis=1)
        nrij = len(self.pairs)
        index_type = np.int32 if max(3 * natoms, 6 * nrij) <= np.iinfo(np.int32).max else np.int64
        self.indices = (self.pairs[:, :, None] * 3 + np.arange(3)).ravel().astype(index_type)
        self.indptr = np.arange(nrij + 1, dtype=index_type) * 6

    def compute(self, geom, func=None, sparse_output=True):
        """Return distances (optionally scaled) and their Cartesian Jacobian.

        The Jacobian has shape ``(npairs, 3 * natoms)`` and is CSR by
        default. Dense output is available for small least-squares problems.
        Returned matrices have independent values and share the fixed layout.
        """
        geom = np.asarray(geom, dtype=float).reshape(-1, 3)
        if len(geom) != self.natoms:
            raise ValueError("Geometry atom count does not match the coordinate system")
        left, right = self.pairs.T
        dvec = geom[left] - geom[right]
        rij = np.linalg.norm(dvec, axis=1)
        coincident = np.flatnonzero(rij == 0)
        if coincident.size:
            i, j = self.pairs[coincident[0]]
            raise ValueError("Cannot evaluate coincident atoms in selected pair ({}, {})".format(i, j))
        grad = dvec / rij[:, None]
        if func is not None:
            values, dwdr = func(rij)
            grad *= np.asarray(dwdr)[..., None]
        else:
            values = rij
        if sparse_output:
            data = np.concatenate((grad, -grad), axis=1).ravel()
            bmat = sparse.csr_matrix((data, self.indices, self.indptr),
                                     shape=(len(self.pairs), geom.size), copy=False)
            bmat.has_sorted_indices = True
        else:
            bmat = np.zeros((len(self.pairs), self.natoms, 3))
            rows = np.arange(len(self.pairs))
            bmat[rows, left] = grad
            bmat[rows, right] = -grad
            bmat = bmat.reshape(len(self.pairs), geom.size)
        return values, bmat


def compute_rij(geom, rij_list):
    """Calculate a list of distances and their derivatives

    Takes a set of cartesian geometries then calculate selected distances and their
    cartesian gradients given a list of atom pairs.

    Args:
        geom: Cartesian geometry of all the points.  Must be 2d numpy array or list
            with shape (natoms, 3)
        rij_list: list of indexes of all the atom pairs

    Returns:
        rij (array): Array of all the distances.
        bmat (3d array): Cartesian gradients of all the distances."""
    geom = np.asarray(geom).reshape(-1, 3)
    rij, bmat = PairCoordinates(len(geom), rij_list).compute(geom, sparse_output=False)
    return rij, bmat.reshape(len(rij_list), len(geom), 3)


def compute_wij(geom, rij_list, func):
    """Calculate a list of scaled distances and their derivatives

    Takes a set of cartesian geometries then calculate selected distances and their
    cartesian gradients given a list of atom pairs.  The distances are scaled with
    a given scaling function.

    Args:
        geom: Cartesian geometry of all the points.  Must be 2d numpy array or list
            with shape (natoms, 3)
        rij_list: 2d numpy array of indexes of all the atom pairs
        func: A scaling function, which returns both the value and derivative.  Must
            qualify as a numpy Ufunc in order to be broadcasted to array elements.

    Returns:
        wij (array): Array of all the scaled distances.
        bmat (2d array): Cartesian gradients of all the scaled distances, with the
            second dimension flattened (need this to be used in scipy.optimize)."""
    geom = np.asarray(geom).reshape(-1, 3)
    return PairCoordinates(len(geom), rij_list).compute(geom, func, sparse_output=False)


def compute_rij_sparse(geom, rij_list):
    """Calculate distances and sparse Cartesian gradients for selected pairs.

    This is equivalent to :func:`compute_rij`, except the derivative matrix is
    returned as CSR with shape ``(nrij, 3 * natoms)``.
    """
    geom = np.asarray(geom).reshape(-1, 3)
    return PairCoordinates(len(geom), rij_list).compute(geom)


def compute_wij_sparse(geom, rij_list, func):
    """Calculate scaled distances and sparse Cartesian gradients.

    The scaled coordinate values match :func:`compute_wij`. The derivative
    matrix is CSR with shape ``(nrij, 3 * natoms)``.
    """
    geom = np.asarray(geom).reshape(-1, 3)
    return PairCoordinates(len(geom), rij_list).compute(geom, func)


def morse_scaler(re=1.5, alpha=1.7, beta=0.01):
    """Returns a scaling function that determines the metric of the internal
    coordinates using morse potential

    Takes an internuclear distance, returns the scaled distance, and the
    derivative of the scaled distance with respect to the unscaled one.
    """
    def scaler(x):
        ratio = x / re
        val1 = np.exp(alpha * (1 - ratio))
        val2 = beta / ratio
        dval = -alpha / re * val1 - val2 / x
        return val1 + val2, dval
    return scaler


def elu_scaler(re=2, alpha=2, beta=0.01):
    """Returns a scaling function that determines the metric of the internal
    coordinates using morse potential

    Takes an internuclear distance, returns the scaled distance, and the
    derivative of the scaled distance with respect to the unscaled one.
    """
    def scaler(x):
        val1 = (1 - x / re) * alpha + 1
        dval = np.full(x.shape, -alpha / re)
        large = x > re
        v1l = np.exp(alpha * (1 - x[large] / re))
        val1[large] = v1l
        dval[large] = -alpha / re * v1l
        val2 = beta * re / x
        return val1 + val2, dval - val2 / x
    return scaler
