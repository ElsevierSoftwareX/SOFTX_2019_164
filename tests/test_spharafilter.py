# -*- coding: utf-8 -*-
""" Module with unit test for spharafilter.py.
"""

from .context import spharapy

import os
import unittest
import csv
import numpy as np

from spharapy import trimesh as tm
from spharapy import spharafilter as sf


# file names for test data
# vertices of triangular mesh
TESTDATA_VERTICES = os.path.join(os.path.dirname(__file__),
                                 'testdata/vertices.csv')

# triangulation of triangular mesh
TESTDATA_TRIANGLES = os.path.join(os.path.dirname(__file__),
                                  'testdata/triangles.csv')


class TestSuiteSpharaFilter(unittest.TestCase):
    """Class that contains the unit test for spharafilter.py.

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

    def test_spharaby_filter_constructor(self):
        """
        Class SpharaFilter, constructor, type exception

        Raise an exception if triangsampel has a wrong type.
        """
        with self.assertRaises(TypeError) as cm:
            sf.SpharaFilter([1, 2, 3, 4])
        err = cm.exception
        self.assertEqual(str(err), 'triangsamples is no instance of TriMesh')

    def test_filter_unit_allpass_simple(self):
        """Class SpharaFilter, mode='unit', allpass, simple mesh

        Apply a SPHARA spatial allpass filter with unit edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_unit_simple = sf.SpharaFilter(testtrimesh, mode='unit',
                                         specification=0)

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_unit_simple.basis()[0])])

        # apply SPHARA based spatial allpass filter
        data_filt_unit_simple = sf_unit_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_unit_simple,
                data))

    def test_filter_unit_dc_simple(self):
        """Class SpharaFilter, mode='unit', dc-pass, simple mesh

        Apply a SPHARA spatial dc-pass filter with unit edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_unit_simple = sf.SpharaFilter(testtrimesh, mode='unit',
                                         specification=1)

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_unit_simple.basis()[0])])

        # reference for filtered data
        data_filt_ref = data.copy()
        data_filt_ref[3] = [0., 0., 0.]
        data_filt_ref[4] = [0., 0., 0.]

        # apply SPHARA based spatial dc-pass filter
        data_filt_unit_simple = sf_unit_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_unit_simple,
                data_filt_ref))

    def test_filter_unit_low_simple(self):
        """Class SpharaFilter, mode='unit', lowpass, simple mesh

        Apply a SPHARA spatial lowpass filter with unit edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_unit_simple = sf.SpharaFilter(testtrimesh, mode='unit',
                                         specification=[1., 1., 0.])

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_unit_simple.basis()[0])])

        # reference for filtered data
        data_filt_ref = data.copy()
        data_filt_ref[4] = [0., 0., 0.]

        # apply SPHARA based spatial lowpass filter
        data_filt_unit_simple = sf_unit_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_unit_simple,
                data_filt_ref))

    def test_filter_ie_allpass_simple(self):
        """Class SpharaFilter, mode='inv_euclidean', allpass, simple mesh

        Apply a SPHARA spatial allpass filter with inv_euclidean edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_ie_simple = sf.SpharaFilter(testtrimesh, mode='inv_euclidean',
                                       specification=0)

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_ie_simple.basis()[0])])

        # apply SPHARA based spatial allpass filter
        data_filt_ie_simple = sf_ie_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_ie_simple,
                data))

    def test_filter_ie_dc_simple(self):
        """Class SpharaFilter, mode='inv_euclidean', dc-pass, simple mesh

        Apply a SPHARA spatial dc-pass filter with inv_euclidean edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_ie_simple = sf.SpharaFilter(testtrimesh, mode='inv_euclidean',
                                       specification=1)

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_ie_simple.basis()[0])])

        # reference for filtered data
        data_filt_ref = data.copy()
        data_filt_ref[3] = [0., 0., 0.]
        data_filt_ref[4] = [0., 0., 0.]

        # apply SPHARA based spatial dc-pass filter
        data_filt_ie_simple = sf_ie_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_ie_simple,
                data_filt_ref))

    def test_filter_ie_low_simple(self):
        """Class SpharaFilter, mode='inv_euclidean', lowpass, simple mesh

        Apply a SPHARA spatial lowpass filter with inv_euclidean edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_ie_simple = sf.SpharaFilter(testtrimesh, mode='inv_euclidean',
                                         specification=[1., 1., 0.])

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_ie_simple.basis()[0])])

        # reference for filtered data
        data_filt_ref = data.copy()
        data_filt_ref[4] = [0., 0., 0.]

        # apply SPHARA based spatial lowpass filter
        data_filt_ie_simple = sf_ie_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_ie_simple,
                data_filt_ref))

    def test_filter_fem_allpass_simple(self):
        """Class SpharaFilter, mode='fem', allpass, simple mesh

        Apply a SPHARA spatial allpass filter with fem edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_fem_simple = sf.SpharaFilter(testtrimesh, mode='fem',
                                        specification=0)

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_fem_simple.basis()[0])])

        # apply SPHARA based spatial allpass filter
        data_filt_fem_simple = sf_fem_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_fem_simple,
                data))

    def test_filter_fem_dc_simple(self):
        """Class SpharaFilter, mode='fem', dc-pass, simple mesh

        Apply a SPHARA spatial dc-pass filter with fem edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_fem_simple = sf.SpharaFilter(testtrimesh, mode='fem',
                                        specification=1)

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_fem_simple.basis()[0])])

        # reference for filtered data
        data_filt_ref = data.copy()
        data_filt_ref[3] = [0., 0., 0.]
        data_filt_ref[4] = [0., 0., 0.]

        # apply SPHARA based spatial dc-pass filter
        data_filt_fem_simple = sf_fem_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_fem_simple,
                data_filt_ref))

    def test_filter_fem_low_simple(self):
        """Class SpharaFilter, mode='fem', lowpass, simple mesh

        Apply a SPHARA spatial lowpass filter with fem edge weight to
        data sampled at a simple triangular mesh, 3 vertices, single
        triangle.

        """
        # define the simple test mesh
        testtrimesh = tm.TriMesh([[0, 1, 2]],
                                 [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        # create a SPHARA filter instance for the mesh
        sf_fem_simple = sf.SpharaFilter(testtrimesh, mode='fem',
                                        specification=[1., 1., 0.])

        # the data to filter
        data = np.concatenate([[[0., 0., 0.], [1., 1., 1.]],
                               np.transpose(sf_fem_simple.basis()[0])])

        # reference for filtered data
        data_filt_ref = data.copy()
        data_filt_ref[4] = [0., 0., 0.]

        # apply SPHARA based spatial lowpass filter
        data_filt_fem_simple = sf_fem_simple.filter(data)

        self.assertTrue(
            np.allclose(
                data_filt_fem_simple,
                data_filt_ref))


if __name__ == '__main__':
    unittest.main()
