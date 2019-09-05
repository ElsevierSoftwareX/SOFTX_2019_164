# -*- coding: utf-8 -*-
"""SPHARA basis functions

This module provides a class for determining SPHARA basis functions. Methodes
are provided to determine basis functions using different discretization
scheme of the Laplace-Beltrami operator, as FEM, inverse euclidean and unit.

"""

import numpy as np
from scipy import linalg
import spharapy.trimesh as tm

__author__ = "Uwe Graichen"
__copyright__ = "Copyright 2018-2019, Uwe Graichen"
__credits__ = ["Uwe Graichen"]
__license__ = "BSD-3-Clause"
__version__ = "1.0.12"
__maintainer__ = "Uwe Graichen"
__email__ = "uwe.graichen@tu-ilmenau.de"
__status__ = "Release"


class SpharaBasis(object):
    """SPHARA basis functions class

    This class can be used to determine SPHARA basis functions for spatially
    irregularly sampled functions whose topology is described by a triangular
    mesh.

    Parameters
    ----------
    triangsamples : trimesh object
        A trimesh object from the package spharapy in which the triangulation
        of the spatial arrangement of the sampling points is stored. The SPHARA
        basic functions are determined for this triangulation of the sample
        points.
    mode : {'unit', 'inv_euclidean', 'fem'}, optional
        The discretization method used to estimate the Laplace-Beltrami
        operator. Using the option 'unit' all edges of
        the mesh are weighted by unit weighting function. The option
        'inv_euclidean' results in edge weights corresponding to the
        inverse Euclidean distance of the edge lengths. The option
        'fem' uses a FEM discretization. the default weighting
        function is 'fem'.

    Attributes
    ----------
    triangsamples: trimesh object
        Triangulation of the spatial arrangement of the sampling points
    mode: {'unit', 'inv_euclidean', 'fem'}
        Discretization used to estimate the Laplace-Beltrami
        operator

    """

    def __init__(self, triangsamples=None, mode='fem'):
        self.triangsamples = triangsamples
        self.mode = mode
        self._basis = None
        self._frequencies = None
        self._massmatrix = None

    @property
    def triangsamples(self):
        """Get or set the triangsamples object.

        The parameter `triangsamples` has to be an instance of the
        class `spharapy.trimesh.TriMesh`. Setting the triangsamples
        object will simultaneously check if it in the correct format.

        """

        return self._triangsamples

    @triangsamples.setter
    def triangsamples(self, triangsamples):
        if not isinstance(triangsamples, tm.TriMesh):
            raise TypeError('triangsamples is no instance of TriMesh')
        # pylint: disable=W0201
        self._triangsamples = triangsamples
        self._basis = None
        self._frequencies = None
        self._massmatrix = None

    @property
    def mode(self):
        """Get or set the discretization method.

        The discretization method used to estimate the Laplace-Beltrami
        operator, choosen from {'unit', 'inv_euclidean', 'fem'}. Setting
        the triangsamples object will simultaneously check if it in the
        correct format.

        """

        return self._mode

    @mode.setter
    def mode(self, mode):
        # plausibility test of option 'mode'
        if mode not in ('unit', 'inv_euclidean', 'fem'):
            raise ValueError("Unrecognized mode '%s'" % mode)
        # pylint: disable=W0201
        self._mode = mode
        self._basis = None
        self._frequencies = None
        self._massmatrix = None

    def basis(self):
        """Return the SPHARA basis for the triangulated sample points

        The method determines a SPHARA basis for spatially distributed
        sampling points described by a triangular mesh. A discrete
        Laplace-Beltrami operator in matrix form is determined for the
        given triangular grid. The discretization methods for
        determining the Laplace-Beltrami operator is specified in the
        attribute `mode`. The eigenvectors :math:`\\vec{x}` and the
        eigenvalues :math:`\\lambda` of the matrix :math:`L`
        containing the discrete Laplace-Beltrami operator are the
        SPHARA basis vectors and the natural frequencies,
        respectively, :math:`L \\vec{x} = \\lambda \\vec{x}`.

        Parameters
        ----------

        Returns
        -------
        basis : array, shape (n_points, n_points)
            Matrix, which contains the SPHARA basis functions column by column.
            The number of vertices of the triangular mesh is n_points.
        frequencies : array, shape (n_points, 1)
            The natural frequencies associated to the SPHARA basis functions.

        Examples
        --------

        >>> from spharapy import trimesh as tm
        >>> from spharapy import spharabasis as sb
        >>> testtrimesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])
        >>> sb_fem = sb.SpharaBasis(testtrimesh, mode='fem')
        >>> sb_fem.basis()
        (array([[ 0.53452248, -0.49487166,  1.42857143],
                [ 0.53452248, -0.98974332, -1.14285714],
                [ 0.53452248,  1.48461498, -0.28571429]]),
         array([  2.33627569e-16,   1.71428571e+00,   5.14285714e+00]))

        """

        # lazy evaluation, compute the basis at the first request and store
        # it until the triangular mesh or the discretization method is changed
        if self._basis is None or self._frequencies is None:
            if self.mode == 'fem':
                self._massmatrix = (self.triangsamples
                                    .massmatrix(mode='normal'))
                stiffmatrix = self.triangsamples.stiffnessmatrix()
                self._frequencies, self._basis = linalg.eigh(-stiffmatrix,
                                                             self._massmatrix)
                # self._basis =
            else:  # 'unit' and 'inv_euclidean' discretization
                laplacianmatrix = (self.triangsamples
                                   .laplacianmatrix(mode=self.mode))
                self._frequencies, self._basis = linalg.eigh(laplacianmatrix)

        # make a row vector of natural frequencies
        # print(self._frequencies)
        # self._frequencies = self._frequencies.transpose
        # print(self._frequencies.shape)
        # return the SPHARA basis
        return self._basis, self._frequencies

    def massmatrix(self):
        """Return the massmatrix

        The method returns the mass matrix of the triangular mesh.

        """
        # lazy evaluation, compute the mass matrix at the first request and
        # store it until the triangular mesh or the discretization method
        # is changed
        if self._massmatrix is None:
            self._massmatrix = self.triangsamples.massmatrix(mode='normal')

        return self._massmatrix
