# -*- coding: utf-8 -*-
"""SPHARA transform

This module provides a class to perform the SPHARA transform. The
class is derived from :class:`spharapy.spharabasis.SpharaBasis`. It
provides methodes the SPHARA anaylsis and synthesis of spatially
irregularly sampled data.

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


class SpharaTransform(SpharaBasis):
    """SPHARA transform class

    This class is used to perform the SPHARA forward (analysis) and
    inverse (synthesis) transformation.

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
        'fem' uses a FEM discretisation. the default weighting
        function is 'fem'.

    """

    def __init__(self, triangsamples=None, mode='fem'):
        SpharaBasis.__init__(self, triangsamples, mode)
        self._basis = None
        self._frequencies = None
        self._massmatrix = None

    def analysis(self, data):
        r"""Perform the SPHARA transform (analysis)

        This method performs the SPHARA transform (analysis) of data
        defined at spatially distributed sampling points described by
        a triangular mesh. The forward transformation is performed by
        matrix multiplication of the data matrix and the matrix with
        SPHARA basis functions :math:`\tilde{X} = X \cdot S`, with the
        SPHARA basis :math:`S`, the data matrix :math:`X` and the
        SPHARA coefficients matix :math:`\tilde{X}`. In the forward
        transformation using SPHARA basic functions determined by
        discretization with FEM approach, the modified scalar product
        including the mass matrix is used :math:`\tilde{X} = X \cdot B
        \cdot S`, with the mass matrix :math:`B`.

        Parameters
        ----------
        data : array, shape(m, n_points)
            A matrix with data to be transformed (analyzed) by
            SPHARA. The number of vertices of the triangular mesh is
            n_points. The order of the spatial sample points must
            correspond to that in the vertex list used to determine
            the SPHARA basis functions.

        Returns
        -------
        coefficients : array, shape (m, n_points)
            A matrix containing the SPHARA coefficients. The coefficients
            are sorted column by column with increasing spatial frequency,
            starting with DC in the first column.

        Examples
        --------

        Import the necessary packages

        >>> import numpy as np
        >>> from spharapy import trimesh as tm
        >>> from spharapy import spharatransform as st
        >>> testtrimesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])
        >>> st_fem_simple = st.SpharaTransform(testtrimesh, mode='fem')
        >>> data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
        ...                        np.transpose(st_fem_simple.basis()[0])])
        >>> data
        array([[ 0.        ,  0.        ,  0.        ],
               [ 1.        ,  1.        ,  1.        ],
               [ 0.53452248,  0.53452248,  0.53452248],
               [-0.49487166, -0.98974332,  1.48461498],
               [ 1.42857143, -1.14285714, -0.28571429]])
        >>> coef_fem_simple = st_fem_simple.analysis(data)
        >>> coef_fem_simple
        array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
               [  1.87082869e+00,   1.09883582e-16,  -4.18977022e-16],
               [  1.00000000e+00,  -2.75573800e-16,  -8.86630311e-18],
               [ -1.14766454e-16,   1.00000000e+00,   2.30648330e-16],
               [  6.52367763e-17,   1.68383874e-16,   1.00000000e+00]])

        """

        # lazy evaluation, compute the basis at the first request and store
        # it until the triangular mesh or the discretization method is changed
        if self._basis is None or self._frequencies is None:
            self.basis()

        data = np.asarray(data)

        # Does the number of spatial samples of the data correspond to
        # that of the basis functions?
        if self._basis.shape[0] != data.shape[1]:
            raise ValueError('Dimension mismatch')

        # compute the SPHARA coefficients
        if self._mode == 'fem':
            coefficients = np.matmul(np.matmul(data, self._massmatrix),
                                     self._basis)
        else:
            coefficients = np.matmul(data, self._basis)

        return coefficients

    def synthesis(self, coefficients):
        r"""Perform the inverse SPHARA transform (synthesis)

        This method performs the inverse SPHARA transform (synthesis)
        for data defined at spatially distributed sampling points
        described by a triangular mesh. The forward transformation is
        performed by matrix multiplication of the data matrix and the
        matrix with SPHARA basis functions :math:`\tilde{X} = X \cdot S`,
        with the SPHARA basis :math:`S`, the data matrix :math:`X` and the
        SPHARA coefficients matix :math:`\tilde{X}`. In the forward
        transformation using SPHARA basic functions determined by
        discretization with FEM approach, the modified scalar product
        including the mass matrix is used
        :math:`\tilde{X} = X \cdot B \cdot S`, with the mass matrix
        :math:`B`.

        Parameters
        ----------
        coefficients : array, shape (m, n_points)
            A matrix containing the SPHARA coefficients. The coefficients
            are sorted column by column with increasing spatial frequency,
            starting with DC in the first column.

        Returns
        -------
        data : array, shape(m, n_points)
            A matrix with data to be forward transformed (analyzed) by
            SPHARA. The number of vertices of the triangular mesh is
            n_points. The order of the spatial sample points must correspond
            to that in the vertex list used to determine the SPHARA basis
            functions.

        Examples
        --------

        >>> import numpy as np
        >>> from spharapy import trimesh as tm
        >>> from spharapy import spharatransform as st
        >>> testtrimesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])
        >>> st_fem_simple = st.SpharaTransform(testtrimesh, mode='fem')
        >>> data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
        ...                        np.transpose(st_fem_simple.basis()[0])])
        >>> data
        array([[ 0.        ,  0.        ,  0.        ],
               [ 1.        ,  1.        ,  1.        ],
               [ 0.53452248,  0.53452248,  0.53452248],
               [-0.49487166, -0.98974332,  1.48461498],
               [ 1.42857143, -1.14285714, -0.28571429]])
        >>> coef_fem_simple = st_fem_simple.analysis(data)
        >>> coef_fem_simple
        array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
               [  1.87082869e+00,   1.09883582e-16,  -4.18977022e-16],
               [  1.00000000e+00,  -2.75573800e-16,  -8.86630311e-18],
               [ -1.14766454e-16,   1.00000000e+00,   2.30648330e-16],
               [  6.52367763e-17,   1.68383874e-16,   1.00000000e+00]])
        >>> recon_fem_simple = st_fem_simple.synthesis(coef_fem_simple)
        >>> recon_fem_simple
        array([[ 0.        ,  0.        ,  0.        ],
               [ 1.        ,  1.        ,  1.        ],
               [ 0.53452248,  0.53452248,  0.53452248],
               [-0.49487166, -0.98974332,  1.48461498],
               [ 1.42857143, -1.14285714, -0.28571429]])

        """

        # lazy evaluation, compute the basis at the first request and store
        # it until the triangular mesh or the discretization method is changed
        if self._basis is None or self._frequencies is None:
            self.basis()

        coefficients = np.asarray(coefficients)

        # Does the number of SPHARA coefficients correspond to that of
        # the basis functions?
        if self._basis.shape[0] != coefficients.shape[1]:
            raise ValueError('Dimension mismatch')

        # compute the data from SPHARA coefficients
        data = np.matmul(coefficients, np.transpose(self._basis))

        return data
