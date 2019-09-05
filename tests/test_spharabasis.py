# -*- coding: utf-8 -*-
""" Module with unit test for spharabasis.py.
"""

from .context import spharapy

import os
import unittest
import csv
import numpy as np

from spharapy import trimesh as tm
from spharapy import spharabasis as sb


# file names for test data
# vertices of triangular mesh
TESTDATA_VERTICES = os.path.join(os.path.dirname(__file__),
                                 'testdata/vertices.csv')

# triangulation of triangular mesh
TESTDATA_TRIANGLES = os.path.join(os.path.dirname(__file__),
                                  'testdata/triangles.csv')

# SPHARA basis unit weight
TESTDATA_SPHARABASISUNITWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/spharabasisunitweight.csv')

# SPHARA natural frequencies unit weight
TESTDATA_SPHARANATFREQUNITWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/spharanatfrequnitweight.csv')

# SPHARA basis inverse euclidean weight
TESTDATA_SPHARABASISIEWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/spharabasisieweight.csv')

# SPHARA natural frequencies inverse euclidean weight
TESTDATA_SPHARANATFREQIEWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/spharanatfreqieweight.csv')

# mass matrix
TESTDATA_MASSMATRIX = os.path.join(os.path.dirname(
    __file__), 'testdata/massmatrix.csv')

# SPHARA basis FEM
TESTDATA_SPHARABASISFEMWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/spharabasisfemweight.csv')

# SPHARA natural frequencies FEM
TESTDATA_SPHARANATFREQFEMWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/spharanatfreqfemweight.csv')


class TestSuiteSpharaBasis(unittest.TestCase):
    """Class that contains the unit test for spharabasis.py.

    """

    def setUp(self):
        # import vertices test data
        with open(TESTDATA_VERTICES, 'r') as f:
            reader = csv.reader(f)
            self.testdatavertices = [[float(y) for y in x]
                                     for x in list(reader)]
            f.close()

        # import triangle test data
        with open(TESTDATA_TRIANGLES, 'r') as f:
            reader = csv.reader(f)
            self.testdatatriangles = [[int(y) for y in x]
                                      for x in list(reader)]
            f.close()

        # import test data, SPHARA basis unit
        with open(TESTDATA_SPHARABASISUNITWEIGHT, 'r') as f:
            reader = csv.reader(f)
            temp = [[float(y) for y in x] for x in list(reader)]
            self.testdataspharabasisunitweight = np.asarray(temp)
            f.close()

        # import test data, SPHARA natural frequencies unit
        with open(TESTDATA_SPHARANATFREQUNITWEIGHT, 'r') as f:
            reader = csv.reader(f)
            temp = np.asarray([[float(y) for y in x] for x in list(reader)])
            self.testdataspharanatfrequnitweight = temp.reshape(-1)
            f.close()

        # import test data, SPHARA basis inverse euclidean
        with open(TESTDATA_SPHARABASISIEWEIGHT, 'r') as f:
            reader = csv.reader(f)
            temp = [[float(y) for y in x] for x in list(reader)]
            self.testdataspharabasisieweight = np.asarray(temp)
            f.close()

        # import test data, SPHARA natural frequencies inverse euclidean
        with open(TESTDATA_SPHARANATFREQIEWEIGHT, 'r') as f:
            reader = csv.reader(f)
            temp = np.asarray([[float(y) for y in x] for x in list(reader)])
            self.testdataspharanatfreqieweight = temp.reshape(-1)
            f.close()

        # import test data, mass matrix
        with open(TESTDATA_MASSMATRIX, 'r') as f:
            reader = csv.reader(f)
            self.testdatamassmatrix = [[float(y) for y in x]
                                       for x in list(reader)]
            f.close()

        # import test data, SPHARA basis FEM
        with open(TESTDATA_SPHARABASISFEMWEIGHT, 'r') as f:
            reader = csv.reader(f)
            temp = [[float(y) for y in x] for x in list(reader)]
            self.testdataspharabasisfemweight = np.asarray(temp)
            f.close()

        # import test data, SPHARA natural frequencies FEM
        with open(TESTDATA_SPHARANATFREQFEMWEIGHT, 'r') as f:
            reader = csv.reader(f)
            temp = np.asarray([[float(y) for y in x] for x in list(reader)])
            self.testdataspharanatfreqfemweight = temp.reshape(-1)
            f.close()

    def test_spharaby_basis_constructor(self):
        """
        Class SpharaBasis, constructor, type exception

        Raise an exception if triangsampel has a wrong type.
        """
        with self.assertRaises(TypeError) as cm:
            sb.SpharaBasis([1, 2, 3, 4])
        err = cm.exception
        self.assertEqual(str(err), 'triangsamples is no instance of TriMesh')

    def test_basis_unit_simple(self):
        """
        Class SpharaBasis, method basis, mode='unit', simple triangular mesh

        Determine the SPHARA basis with unit edge weight for a simple
        triangular mesh, 3 vertices, single triangle.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        sb_unit = sb.SpharaBasis(testtrimesh, mode='unit')
        sb_unit_fun, sb_unit_freq = sb_unit.basis()
        # print(sb_unit_freq.shape)
        self.assertTrue(
            np.allclose(
                sb_unit_fun,
                [[0.57735027, 0.81649658, 0.],
                 [0.57735027, -0.40824829, -0.70710678],
                 [0.57735027, -0.40824829, 0.70710678]])
            and
            np.allclose(
                sb_unit_freq,
                [2.22044605e-15, 3.00000000e+00, 3.00000000e+00]))

    def test_basis_unit_result(self):
        """Class SpharaBasis, method basis, mode='unit', triangular test mesh

        Determine the SPHARA basis with unit edge weight for a
        triangular mesh with 118 vertices. The valid basis vectors of
        the SPHARA basis can point in opposite directions
        (multiplication by -1). To compare the calculated basis with
        the reference basis, the transposed matrix of the calculated
        basis is multiplied by the matrix of the reference basis. If
        the calculated basis is correct, then the result matrix of the
        matrix multiplication contains only the elements 1 and -1 at
        the main diagonal, all other elements of the matrix are 0 or
        very small.

        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        sb_unit = sb.SpharaBasis(testtrimesh, mode='unit')
        sb_unit_fun, sb_unit_freq = sb_unit.basis()
        self.assertTrue(
            np.allclose(np.absolute(np.matmul
                                    (np.transpose(sb_unit_fun),
                                     self.testdataspharabasisunitweight)),
                        np.identity(np.size(sb_unit_freq)))
            and
            np.allclose(sb_unit_freq, self.testdataspharanatfrequnitweight)
        )

    def test_basis_inveuclid_simple(self):
        """
        Class SpharaBasis, method basis, mode='inv_euclidean', simple
        triangular mesh

        Determine the SPHARA basis with inverse euclidean edge weight
        for a simple triangular mesh, 3 vertices, single triangle.

        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        sb_ie = sb.SpharaBasis(testtrimesh, mode='inv_euclidean')
        sb_ie_fun, sb_ie_freq = sb_ie.basis()
        self.assertTrue(
            np.allclose(
                sb_ie_fun,
                [[-0.577350269189626, -0.328082121693334, 0.747682277502862],
                 [-0.577350269189626, -0.483470785430218, -0.657968590665356],
                 [-0.577350269189626, 0.811552907123552, -0.0897136868375066]])
            and
            np.allclose(
                sb_ie_freq,
                [0.0, 0.886644827600501, 1.19493809165832]))

    def test_basis_ie_result(self):
        """Class SpharaBasis, method basis, mode='inv_euclidean', triangular
        test mesh

        Determine the SPHARA basis with inverse euclidean edge weight
        for a triangular mesh with 118 vertices. The valid basis
        vectors of the SPHARA basis can point in opposite directions
        (multiplication by -1). To compare the calculated basis with
        the reference basis, the transposed matrix of the calculated
        basis is multiplied by the matrix of the reference basis. If
        the calculated basis is correct, then the result matrix of the
        matrix multiplication contains only the elements 1 and -1 at
        the main diagonal, all other elements of the matrix are 0 or
        very small.

        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        sb_ie = sb.SpharaBasis(testtrimesh, mode='inv_euclidean')
        sb_ie_fun, sb_ie_freq = sb_ie.basis()
        self.assertTrue(
            np.allclose(np.absolute(np.matmul
                                    (np.transpose(sb_ie_fun),
                                     self.testdataspharabasisieweight)),
                        np.identity(np.size(sb_ie_freq)))
            and
            np.allclose(sb_ie_freq, self.testdataspharanatfreqieweight)
        )

    def test_basis_fem_simple(self):
        """
        Class SpharaBasis, method basis, mode='fem', simple triangular mesh

        Determine the SPHARA basis with fem edge weight
        for a simple triangular mesh, 3 vertices, single triangle.

        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        sb_fem = sb.SpharaBasis(testtrimesh, mode='fem')
        sb_fem_fun, sb_fem_freq = sb_fem.basis()
        self.assertTrue(
            np.allclose(
                sb_fem_fun,
                [[0.53452248, -0.49487166, 1.42857143],
                 [0.53452248, -0.98974332, -1.14285714],
                 [0.53452248,  1.48461498, -0.28571429]])
            and
            np.allclose(
                sb_fem_freq,
                [2.33627569e-16, 1.71428571e+00, 5.14285714e+00]))

    def test_basis_fem_result(self):
        """Class SpharaBasis, method basis, mode='fem', triangular test mesh

        Determine the SPHARA basis with FEM discretization for a
        triangular mesh with 118 vertices. The valid basis vectors of
        the SPHARA basis can point in opposite directions
        (multiplication by -1). To compare the calculated basis with
        the reference basis, the transposed matrix of the calculated
        basis is multiplied by the matrix of the reference basis. If
        the calculated basis is correct, then the result matrix of the
        matrix multiplication contains only the elements 1 and -1 at
        the main diagonal, all other elements of the matrix are 0 or
        very small.

        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        sb_fem = sb.SpharaBasis(testtrimesh, mode='fem')
        sb_fem_fun, sb_fem_freq = sb_fem.basis()
        self.assertTrue(
            np.allclose(np.absolute(np.matmul(np.matmul
                                    (np.transpose(sb_fem_fun),
                                     self.testdatamassmatrix),
                                     self.testdataspharabasisfemweight)),
                        np.identity(np.size(sb_fem_freq)))
            and
            np.allclose(sb_fem_freq, self.testdataspharanatfreqfemweight)
        )


if __name__ == '__main__':
    unittest.main()
