# -*- coding: utf-8 -*-
""" Module with unit test for trimesh.py.
"""


from .context import spharapy

import os
import unittest
import csv
import numpy as np

from spharapy import trimesh as tm


# file names for test data
# vertices of triangular mesh
TESTDATA_VERTICES = os.path.join(os.path.dirname(__file__),
                                 'testdata/vertices.csv')
# triangulation of triangular mesh
TESTDATA_TRIANGLES = os.path.join(os.path.dirname(__file__),
                                  'testdata/triangles.csv')
# weight matrix unit weight
TESTDATA_WEIGHTMATRIXUNITWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/weightmatrixunitweight.csv')

# weight matrix inverse euclidean weight
TESTDATA_WEIGHTMATRIXIEWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/weightmatrixieweight.csv')

# weight matrix half cotangent weight
TESTDATA_WEIGHTMATRIXHALFCOTWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/weightmatrixhalfcotweight.csv')

# mass matrix
TESTDATA_MASSMATRIX = os.path.join(os.path.dirname(
    __file__), 'testdata/massmatrix.csv')

# stiffness matrix
TESTDATA_STIFFNESSMATRIX = os.path.join(os.path.dirname(
    __file__), 'testdata/stiffmatrix.csv')

# laplacian matrix unit weight
TESTDATA_LAPLACIANMATRIXUNITWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/laplacianmatrixunit.csv')

# laplacian matrix inverse euclidean weight
TESTDATA_LAPLACIANMATRIXIEWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/laplacianmatrixie.csv')

# laplacian matrix half cotangent weight
TESTDATA_LAPLACIANMATRIXHALFCOTWEIGHT = os.path.join(os.path.dirname(
    __file__), 'testdata/laplacianmatrixhalfcot.csv')


class TestSuiteTriMesh(unittest.TestCase):
    """Class that contains the unit test for trimesh.py.

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

        # import test data, weight matrix unit
        with open(TESTDATA_WEIGHTMATRIXUNITWEIGHT, 'r') as f:
            reader = csv.reader(f)
            self.testdataweightmatrixunitweight = [[int(y) for y in x]
                                                   for x in list(reader)]
            f.close()

        # import test data, weight matrix inverse euclidean
        with open(TESTDATA_WEIGHTMATRIXIEWEIGHT, 'r') as f:
            reader = csv.reader(f)
            self.testdataweightmatrixieweight = [[float(y) for y in x]
                                                 for x in list(reader)]
            f.close()

        # import test data, weight matrix half cotangent
        with open(TESTDATA_WEIGHTMATRIXHALFCOTWEIGHT, 'r') as f:
            reader = csv.reader(f)
            self.testdataweightmatrixhalfcotweight = [[float(y) for y in x]
                                                      for x in list(reader)]
            f.close()

        # import test data, mass matrix
        with open(TESTDATA_MASSMATRIX, 'r') as f:
            reader = csv.reader(f)
            self.testdatamassmatrix = [[float(y) for y in x]
                                       for x in list(reader)]
            f.close()

        # import test data, stiffness matrix
        with open(TESTDATA_STIFFNESSMATRIX, 'r') as f:
            reader = csv.reader(f)
            self.testdatastiffnessmatrix = [[float(y) for y in x]
                                            for x in list(reader)]
            f.close()

        # import test data, laplacian matrix unit
        with open(TESTDATA_LAPLACIANMATRIXUNITWEIGHT, 'r') as f:
            reader = csv.reader(f)
            self.testdatalaplacianmatrixunitweight = [[int(y) for y in x]
                                                      for x in list(reader)]
            f.close()

        # import test data, laplacian matrix inverse euclidean
        with open(TESTDATA_LAPLACIANMATRIXIEWEIGHT, 'r') as f:
            reader = csv.reader(f)
            self.testdatalaplacianmatrixieweight = [[float(y) for y in x]
                                                    for x in list(reader)]
            f.close()

        # import test data, laplacian matrix half cotangent
        with open(TESTDATA_LAPLACIANMATRIXHALFCOTWEIGHT, 'r') as f:
            reader = csv.reader(f)
            self.testdatalaplacianmatrixhalfcotweight = [[float(y) for y in x]
                                                         for x in list(reader)]
            f.close()

    def test_tri_mesh_constructor_dimension(self):
        """
        Class TriMesh, constructor, dimension exception

        Raise an exception if triangle list has wrong dimension.
        """
        with self.assertRaises(ValueError) as cm:
            tm.TriMesh([1, 2, 3, 4], [1, 2, 3, 4])
        err = cm.exception
        self.assertEqual(str(err), 'Triangle list has to be 2D!')

    def test_tri_mesh_constructor_three_row(self):
        """
        Class TriMesh, constructor, non 3 elem exception

        Raise an exception if the rows of the triangle list dont consists of
        3 integers.
        """
        with self.assertRaises(ValueError) as cm:
            tm.TriMesh([[0, 1], [2, 3]], [[1, 2], [3, 4]])
        err = cm.exception
        self.assertEqual(str(err), 'Each entry of the triangle list has to '
                         'consist of three elements!')

    def test_weightmatrix_unitweight_simple(self):
        """
        Class TriMesh, method weightmatrix, mode='unit', simple triangular mesh

        Determine the weight matrix with unit edge weight for a simple
        triangular mesh, 3 vertices, single triangle, equilateral.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        self.assertTrue(np.allclose(
            testtrimesh.weightmatrix(mode='unit'),
            [[0.0, 1.0, 1.0],
             [1.0, 0.0, 1.0],
             [1.0, 1.0, 0.0]]))

    def test_weightmatrix_unitweight_result(self):
        """
        Class TriMesh, method weightmatrix, mode='unit', triangular test mesh

        Determine the weight matrix with unit edge weight for a triangular mesh
        with 118 vertices.
        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        self.assertTrue(np.array_equal(testtrimesh.weightmatrix(mode='unit'),
                                       self.testdataweightmatrixunitweight))

    def test_weightmatrix_halfcot_simple(self):
        """
        Class TriMesh, method weightmatrix, mode='half_cotangent', simple
        triangular mesh

        Determine the weight matrix with unit edge weight for a simple
        triangular mesh, 3 vertices, single triangle, equilateral.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        self.assertTrue(np.allclose(
            testtrimesh.weightmatrix(mode='half_cotangent'),
            [[0., 0.64285714, 0.28571429],
             [0.64285714, 0., 0.07142857],
             [0.28571429, 0.07142857, 0.]]))

    def test_weightmatrix_halfcot_result(self):
        """
        Class TriMesh, method weightmatrix, mode='half_cotangent', triangular
        test mesh

        Determine the weight matrix with half cotangent edge weight for a
        triangular mesh with 118 vertices.
        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)

        (self.assertTrue(np.allclose(
            testtrimesh.weightmatrix(mode='half_cotangent'),
            self.testdataweightmatrixhalfcotweight)))

    def test_weightmatrix_inveuclid_simple(self):
        """
        Class TriMesh, method weightmatrix, mode='inv_euclidean', simple
        triangular mesh

        Determine the weight matrix with inverse euclidean edge weight for a
        simple triangular mesh, 3 vertices, single triangle, equilateral.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        self.assertTrue(np.allclose(
            testtrimesh.weightmatrix(mode='inv_euclidean'),
            [[0., 0.4472136, 0.31622777],
             [0.4472136, 0., 0.2773501],
             [0.31622777, 0.2773501, 0.]]))

    def test_weightmatrix_inveuclid_result(self):
        """
        Class TriMesh, method weightmatrix, mode='inv_euclidean', triangular
        test mesh

        Determine the weight matrix with inverse euclidean edge weight for a
        triangular mesh with 118 vertices.
        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        (self.assertTrue(np.allclose(
            testtrimesh.weightmatrix(mode='inv_euclidean'),
            self.testdataweightmatrixieweight)))

    def test_area_triangle_result(self):
        """
        Module trimesh, function area_triangle

        test for rectangular and equilateral triangle
        """
        self.assertTrue(tm.area_triangle([1, 0, 0],
                                         [0, 1, 0], [0, 0, 0]) == 0.5 and
                        np.isclose(tm.area_triangle([1, 0, 0],
                                                    [0, 1, 0], [0, 0, 1]),
                        0.866025403784))

    def test_massmatrix_simple(self):
        """Class TriMesh, method massmatrix, simple triangular mesh

        Determine the mass matrix for a simple triangular mesh, 3
        vertices, single triangle, equilateral.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        self.assertTrue(np.allclose(testtrimesh.massmatrix(),
                                    [[0.58333333, 0.29166667, 0.29166667],
                                     [0.29166667, 0.58333333, 0.29166667],
                                     [0.29166667, 0.29166667, 0.58333333]]))

    def test_massmatrix_result(self):
        """
        Class TriMesh, method massmatrix, triangular test mesh

        Determine the mass matrix for a triangular mesh with 118 vertices.
        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        (self.assertTrue(np.allclose(
            testtrimesh.massmatrix(),
            self.testdatamassmatrix)))

    def test_stiffnessmatrix_simple(self):
        """Class TriMesh, method stiffnessmatrix, simple triangular mesh

        Determine the stiffness matrix for a simple triangular mesh, 3
        vertices, single triangle, equilateral.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        self.assertTrue(np.allclose(testtrimesh.stiffnessmatrix(),
                                    [[-0.92857143, 0.64285714, 0.28571429],
                                     [0.64285714, -0.71428571, 0.07142857],
                                     [0.28571429, 0.07142857, -0.35714286]]))

    def test_stiffnessmatrix_result(self):
        """
        Class TriMesh, method stiffnessmatrix, triangular test mesh

        Determine the stiffness matrix for a triangular mesh with 118 vertices.
        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        (self.assertTrue(np.allclose(
            testtrimesh.stiffnessmatrix(),
            self.testdatastiffnessmatrix)))

    def test_laplacianmatrix_unit_simple(self):
        """
        Class TriMesh, method laplacianmatrix, mode='unit', simple triangular
        mesh

        Determine the laplacian matrix (unit edge weighting) for a simple
        triangular mesh, 3 vertices, single triangle, equilateral.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        self.assertTrue(np.allclose(testtrimesh.laplacianmatrix(mode='unit'),
                                    [[2, -1, -1],
                                     [-1, 2, -1],
                                     [-1, -1, 2]]))

    def test_laplacianmatrix_unit_result(self):
        """
        Class TriMesh, method laplacianmatrix, mode='unit', triangular test
        mesh

        Determine the laplacian matrix (unit edge weighting) for a triangular
        mesh with 118 vertices.
        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        (self.assertTrue(np.allclose(
            testtrimesh.laplacianmatrix(mode='unit'),
            self.testdatalaplacianmatrixunitweight)))

    def test_laplacianmatrix_inveuclid_simple(self):
        """
        Class TriMesh, method laplacianmatrix, mode='inv_euclidean', simple
        triangular mesh

        Determine the laplacian matrix (inverse euclidean edge weighting) for
        a simple triangular mesh, 3 vertices, single triangle, equilateral.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        (self.assertTrue(np.allclose(testtrimesh
                                     .laplacianmatrix(mode='inv_euclidean'),
                                     [[0.76344136, -0.4472136, -0.31622777],
                                      [-0.4472136, 0.72456369, -0.2773501],
                                      [-0.31622777, -0.2773501, 0.59357786]])))

    def test_laplacianmatrix_inveuclid_result(self):
        """
        Class TriMesh, method laplacianmatrix, mode='inv_euclidean', triangular
        test mesh

        Determine the laplacian matrix (inverse euclidean edge weighting) for
        a triangular mesh with 118 vertices.
        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        (self.assertTrue(np.allclose(
            testtrimesh.laplacianmatrix(mode='inv_euclidean'),
            self.testdatalaplacianmatrixieweight)))

    def test_laplacianmatrix_halfcot_simple(self):
        """
        Class TriMesh, method laplacianmatrix, mode='half_cotangent', simple
        triangular mesh

        Determine the laplacian matrix (half cotangent edge weighting) for
        a simple triangular mesh, 3 vertices, single triangle, equilateral.
        """
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        (self.assertTrue(np.allclose(testtrimesh
                                     .laplacianmatrix(mode='half_cotangent'),
                                     [[0.92857143, -0.64285714, -0.28571429],
                                      [-0.64285714, 0.71428571, -0.07142857],
                                      [-0.28571429, -0.07142857, 0.35714286]]
                                     )))

    def test_laplacianmatrix_halfcot_result(self):
        """
        Class TriMesh, method laplacianmatrix, mode='half_cotangent',
        triangular test mesh

        Determine the laplacian matrix (half cotangent edge weighting) for
        a triangular mesh with 118 vertices.
        """
        testtrimesh = tm.TriMesh(self.testdatatriangles,
                                 self.testdatavertices)
        (self.assertTrue(np.allclose(
            testtrimesh.laplacianmatrix(mode='half_cotangent'),
            self.testdatalaplacianmatrixhalfcotweight)))


if __name__ == '__main__':
    unittest.main()
