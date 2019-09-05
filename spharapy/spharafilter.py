# -*- coding: utf-8 -*-
"""SPHARA filter

This module provides a class to perform a spatial filtering using a
SPHARA basis. The class is derived from
:class:`spharapy.spharabasis.SpharaBasis`. It provides methodes to
design different types of filters and to apply this filters to
spatially irregularly sampled data.

"""

import numpy as np
from scipy import linalg
from spharapy.spharabasis import SpharaBasis

__author__ = "Uwe Graichen"
__copyright__ = "Copyright 2018-2019, Uwe Graichen"
__credits__ = ["Uwe Graichen"]
__license__ = "BSD-3-Clause"
__version__ = "1.0.12"
__maintainer__ = "Uwe Graichen"
__email__ = "uwe.graichen@tu-ilmenau.de"
__status__ = "Release"


class SpharaFilter(SpharaBasis):
    """SPHARA filter class

    This class is used to design different types of filters and to
    apply this filters to spatially irregularly sampled data.

    Parameters
    ----------
    triangsamples : trimesh object
        A trimesh object from the package spharapy in which the triangulation
        of the spatial arrangement of the sampling points is stored. The SPHARA
        basic functions are determined for this triangulation of the sample
        points.
    mode : {'unit', 'inv_euclidean', 'fem'}, optional
        The discretisation method used to estimate the Laplace-Beltrami
        operator. Using the option 'unit' all edges of
        the mesh are weighted by unit weighting function. The option
        'inv_euclidean' results in edge weights corresponding to the
        inverse Euclidean distance of the edge lengths. The option
        'fem' uses a FEM discretisation. The default weighting
        function is 'fem'.
    specification : integer or array, shape (1, n_points)

        If an integer value for specification is passed to the
        constructor, it must be within the interval (-n_points,
        n_points), where n_points is the number of spatial sample
        points. If a positive integer value is passed, a spatial
        low-pass filter with the corresponding number of SPHARA basis
        functions is created, if a negative integer value is passed, a
        spatial low-pass filter is created. If a vector is passed,
        then all SPHARA basis functions corresponding to nonzero
        elements of the vector are used to create the filter. The
        default value of specification is 0, it means a neutral
        all-pass filter is designed and applied.

    """

    def __init__(self, triangsamples=None, mode='fem', specification=0):
        SpharaBasis.__init__(self, triangsamples, mode)
        self.specification = specification
        self._basis = None
        self._frequencies = None
        self._massmatrix = None
        self._filtermatrix = None

    @property
    def specification(self):
        """Get or set the specification of the filter.

        The parameter `specification` has to be an integer or a vector.
        Setting the `specification` will simultaneously apply a plausibility
        check.

        """

        return self._specification

    @specification.setter
    def specification(self, specification):
        if isinstance(specification, (int)):
            if np.abs(specification) > self._triangsamples.vertlist.shape[0]:
                raise ValueError("""The Number of selected basic functions is
                too large.""")
            else:
                if specification == 0:
                    self._specification = \
                        np.ones(self._triangsamples.vertlist.shape[0])
                else:
                    self._specification = \
                        np.zeros(self._triangsamples.vertlist.shape[0])
                    if specification > 0:
                        self._specification[:specification] = 1
                    else:
                        self._specification[specification:] = 1
        elif isinstance(specification, (list, tuple, np.ndarray)):
            specification = np.asarray(specification)
            if specification.shape[0] != self._triangsamples.vertlist.shape[1]:
                raise IndexError("""The length of the specification vector
                does not match the number of spatial sample points. """)
            else:
                self._specification = specification
        else:
            raise TypeError("""The parameter specification has to be
            int or a vecor""")

    def filter(self, data):
        r"""Perform the SPHARA filtering

        This method performs the spatial SPHARA filtering
        for data defined at spatially distributed sampling points
        described by a triangular mesh. The filtering is
        performed by matrix multiplication of the data matrix and a
        precalculated filter matrix.

        Parameters
        ----------
        data : array, shape(m, n_points)
            A matrix with data to be filtered by spatial SPHARA
            filter. The number of vertices of the triangular mesh is
            n_points. The order of the spatial sample points must
            correspond to that in the vertex list used to determine
            the SPHARA basis functions.

        Returns
        -------
        data_filtered : array, shape (m, n_points)
            A matrix containing the filtered data.

        Examples
        --------

        >>> import numpy as np
        >>> from spharapy import trimesh as tm
        >>> from spharapy import spharafilter as sf
        >>> # define the simple test mesh
        >>> testtrimesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])
        >>> # create a spatial lowpass filter, FEM discretisation
        >>> sf_fem = sf.SpharaFilter(testtrimesh, mode='fem',
        ...                          specification=[1., 1., 0.])
        >>> # create some test data
        >>> data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
        ...                          np.transpose(sf_fem.basis()[0])])
        >>> data
        array([[ 0.        ,  0.        ,  0.        ],
               [ 1.        ,  1.        ,  1.        ],
               [ 0.53452248,  0.53452248,  0.53452248],
               [-0.49487166, -0.98974332,  1.48461498],
               [ 1.42857143, -1.14285714, -0.28571429]])
        >>> # filter the test data
        >>> data_filtered = sf_fem.filter(data)
        >>> data_filtered
        array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
               [  1.00000000e+00,   1.00000000e+00,   1.00000000e+00],
               [  5.34522484e-01,   5.34522484e-01,   5.34522484e-01],
               [ -4.94871659e-01,  -9.89743319e-01,   1.48461498e+00],
               [ -1.69271249e-16,  -2.75762028e-16,   3.10220481e-16]])

        """

        # lazy evaluation, compute the basis at the first request and store
        # it until the triangular mesh or the discretization method is changed
        if self._basis is None or self._frequencies is None:
            self.basis()

        # lazy evaluation, compute the filter matrix at the first request and
        # store it until the triangular mesh or the discretization method is
        # changed
        if self._filtermatrix is None:
            basis_fun_sel = np.matmul(self._basis,
                                      np.diag(self._specification))
            if self._mode == 'fem':
                self._filtermatrix = np.matmul(
                    np.matmul(self._massmatrix, basis_fun_sel),
                    basis_fun_sel.transpose())
            else:
                self._filtermatrix = np.matmul(basis_fun_sel,
                                               basis_fun_sel.transpose())

        data = np.asarray(data)

        # Does the number of spatial samples of the data correspond to
        # that of the basis functions?
        if self._basis.shape[0] != data.shape[1]:
            raise ValueError('Dimension of data and filter matrix not equal.')

        # filter the data
        data_filtered = np.matmul(data, self._filtermatrix)

        return data_filtered
