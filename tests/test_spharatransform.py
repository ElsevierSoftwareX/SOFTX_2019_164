# -*- coding: utf-8 -*-
""" Module with unit test for spharatransform.py.
"""

from .context import spharapy

import os
import unittest
import csv
import numpy as np

from spharapy import trimesh as tm
from spharapy import spharatransform as st


# file names for test data
# vertices of triangular mesh
TESTDATA_VERTICES = os.path.join(os.path.dirname(__file__),
                                 'testdata/vertices.csv')

# triangulation of triangular mesh
TESTDATA_TRIANGLES = os.path.join(os.path.dirname(__file__),
                                  'testdata/triangles.csv')


class TestSuiteSpharaTransform(unittest.TestCase):
    """Class that contains the unit test for spharatransform.py.

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

    def test_spharaby_transform_constructor(self):
        """
        Class SpharaTransform, constructor, type exception

        Raise an exception if triangsampel has a wrong type.
        """
        with self.assertRaises(TypeError) as cm:
            st.SpharaTransform([1, 2, 3, 4])
        err = cm.exception
        self.assertEqual(str(err), 'triangsamples is no instance of TriMesh')

    def test_transform_unit_simple(self):
        """Class SpharaTransform, mode='unit', simple triangular mesh

        Determine the SPHARA forward and inverse transform with unit
        edge weight for a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA transform instance for the mesh
        st_unit_simple = st.SpharaTransform(testtrimesh, mode='unit')

        # the data to transform
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(st_unit_simple.basis()[0])])

        # SPHARA analysis
        coef_unit_simple = st_unit_simple.analysis(data)

        # SPHARA synthesis
        recon_unit_simple = st_unit_simple.synthesis(coef_unit_simple)

        self.assertTrue(
            np.allclose(
                np.absolute(coef_unit_simple),
                [[0.0, 0.0, 0.0],
                 [1.73205081, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]])
            and
            np.allclose(
                recon_unit_simple,
                data))

    def test_transform_ie_simple(self):
        """Class SpharaTransform, mode='inv_euclidean', simple triangular mesh

        Determine the SPHARA forward and inverse transform with
        inverse Euclidean edge weight for a simple triangular mesh, 3
        vertices, single triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA transform instance for the mesh
        st_ie_simple = st.SpharaTransform(testtrimesh, mode='inv_euclidean')

        # the data to transform
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(st_ie_simple.basis()[0])])

        # SPHARA analysis
        coef_ie_simple = st_ie_simple.analysis(data)

        # SPHARA synthesis
        recon_ie_simple = st_ie_simple.synthesis(coef_ie_simple)

        self.assertTrue(
            np.allclose(
                np.absolute(coef_ie_simple),
                [[0.0, 0.0, 0.0],
                 [1.73205081, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]])
            and
            np.allclose(
                recon_ie_simple,
                data))

    def test_transform_fem_simple(self):
        """Class SpharaTransform, mode='fem', simple triangular mesh

        Determine the SPHARA forward and inverse transform with fem
        discretisation for a simple triangular mesh, 3 vertices,
        single triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA transform instance for the mesh
        st_fem_simple = st.SpharaTransform(testtrimesh, mode='fem')

        # the data to transform
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(st_fem_simple.basis()[0])])

        # SPHARA analysis
        coef_fem_simple = st_fem_simple.analysis(data)

        # SPHARA synthesis
        recon_fem_simple = st_fem_simple.synthesis(coef_fem_simple)

        self.assertTrue(
            np.allclose(
                np.absolute(coef_fem_simple),
                [[0.0, 0.0, 0.0],
                 [1.87082868, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]])
            and
            np.allclose(
                recon_fem_simple,
                data))


if __name__ == '__main__':
    unittest.main()
