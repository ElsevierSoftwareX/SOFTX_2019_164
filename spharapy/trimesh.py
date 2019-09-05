# -*- coding: utf-8 -*-
"""Triangular mesh data

This module provides a class for storing triangular meshes. Attributes
of the triangular mesh can be determined. In addition, methodes are available
to derive further information from the triangular grid.

"""

import numpy as np

__author__ = "Uwe Graichen"
__copyright__ = "Copyright 2018-2019, Uwe Graichen"
__credits__ = ["Uwe Graichen"]
__license__ = "BSD-3-Clause"
__version__ = "1.0.12"
__maintainer__ = "Uwe Graichen"
__email__ = "uwe.graichen@tu-ilmenau.de"
__status__ = "Release"


def area_triangle(vertex1, vertex2, vertex3):
    """Estimate the area of a triangle given by three vertices

    The area of the triangle given by three vertices is calculated by the half
    cross product formula.

    Parameters
    ----------
    vertex1 : array, shape (1, 3)
    vertex2 : array, shape (1, 3)
    vertex3 : array, shape (1, 3)


    Returns
    -------
    float
        Area of the triangle given by the three vertices.


    Examples
    --------

    >>> from spharapy import trimesh as tm
    >>> tm.area_triangle([1, 0, 0], [0, 1, 0], [0, 0, 1])
    0.8660254037844386

    """
    vertex1 = np.asarray(vertex1)
    vertex2 = np.asarray(vertex2)
    vertex3 = np.asarray(vertex3)

    trianglearea = (0.5 *
                    np.linalg.norm(np.cross(vertex2 - vertex1,
                                            vertex3 - vertex1)))
    return trianglearea


class TriMesh(object):
    """Triangular mesh class

    This class can be used to store data to define a triangular mesh and it
    provides atributes and methodes to derive further information about the
    triangular mesh.

    Parameters
    ----------
    trilist: array, shape (n_triangles, 3)
        List of triangles, each row of the array contains the edges of
        a triangle. The edges of the triangles are defined by the
        indices to the list of vertices. The index of the first vertex
        is 0. The number of triangles is n_triangles.
    vertlist: array, shape (n_points, 3)
        List of coordinates x, y, z which describes the positions of the
        vertices.

    Attributes
    ----------
    trilist: array, shape (n_triangles, 3)
        List of triangles of the mesh.
    vertlist: array, shape (n_points, 3)
        List of coordinates of the vertices
    """

    def __init__(self, trilist, vertlist):
        self.trilist = np.asarray(trilist)
        self.vertlist = np.asarray(vertlist)

    @property
    def trilist(self):
        """Get or set the list of triangles.

        Setting the list of triangles will simultaneously check if the
        triangle list is in the correct format.
        """

        return self._trilist

    @trilist.setter
    def trilist(self, trilist):
        if trilist.ndim != 2:
            raise ValueError('Triangle list has to be 2D!')
        elif trilist.shape[1] != 3:
            raise ValueError('Each entry of the triangle list has to consist '
                             'of three elements!')
        # pylint: disable=W0201
        self._trilist = np.asarray(trilist)

    @property
    def vertlist(self):
        """Get or set the list of vertices.

        Setting the list of triangles will simultaneously check if the
        vertice list is in the correct format.
        """

        return self._vertlist

    @vertlist.setter
    def vertlist(self, vertlist):
        if vertlist.ndim != 2:
            raise ValueError('Vertex list has to be 2D!')
        elif vertlist.shape[1] != 3:
            raise ValueError('Each entry of the vertex list has to consist '
                             'of three elements!')
        # pylint: disable=W0201
        self._vertlist = np.asarray(vertlist)

    def weightmatrix(self, mode='inv_euclidean'):
        """Compute a weight matrix for a triangular mesh

        The method creates a weighting matrix for the edges of a triangular
        mesh using different weighting function.

        Parameters
        ----------
        mode : {'unit', 'inv_euclidean', 'half_cotangent'}, optional
            The parameter `mode` specifies the method for determining
            the edge weights. Using the option 'unit' all edges of the
            mesh are weighted by unit weighting function, the result
            is an adjacency matrix. The option 'inv_euclidean' results
            in edge weights corresponding to the inverse Euclidean
            distance of the edge lengths. The option 'half_cotangent'
            uses the half of the cotangent of the two angles opposed
            to an edge as weighting function. the default weighting
            function is 'inv_euclidean'.

        Returns
        -------
        weightmatrix : array, shape (n_points, n_points)
            Symmetric matrix, which contains the weight of the edges
            between adjacent vertices. The number of vertices of the
            triangular mesh is n_points.

        Examples
        --------

        >>> from spharapy import trimesh as tm
        >>> testtrimesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])
        >>> testtrimesh.weightmatrix(mode='inv_euclidean')
        array([[ 0.        ,  0.4472136 ,  0.31622777],
               [ 0.4472136 ,  0.        ,  0.2773501 ],
               [ 0.31622777,  0.2773501 ,  0.        ]])

        """

        # plausibility test of option 'mode'
        if mode not in ('unit', 'inv_euclidean', 'half_cotangent'):
            raise ValueError("Unrecognized mode '%s'" % mode)

        # get the largest index from from triangle list
        maxindex = self.trilist.max()

        # fill the weight matrix with zeros, the size is (maxindex + 1)^2
        weightmatrix = np.zeros((maxindex + 1, maxindex + 1), dtype=float)

        if mode == 'unit':
            # iterate over triangle list an build weight matrix
            for x in self.trilist:

                weightmatrix[x[0], x[1]] = 1
                weightmatrix[x[1], x[0]] = 1

                weightmatrix[x[0], x[2]] = 1
                weightmatrix[x[2], x[0]] = 1

                weightmatrix[x[1], x[2]] = 1
                weightmatrix[x[2], x[1]] = 1

        elif mode == 'inv_euclidean':
            # iterate over triangle list an build weight matrix
            for x in self.trilist:
                # compute the three vectors of the triangle
                vec10 = self.vertlist[x[1]] - self.vertlist[x[0]]
                vec20 = self.vertlist[x[2]] - self.vertlist[x[0]]
                vec21 = self.vertlist[x[2]] - self.vertlist[x[1]]

                # fill in the weights in the weight matrix
                weightmatrix[x[0], x[1]] = (1 / np.linalg.norm(vec10))
                weightmatrix[x[1], x[0]] = (1 / np.linalg.norm(vec10))
                weightmatrix[x[0], x[2]] = (1 / np.linalg.norm(vec20))
                weightmatrix[x[2], x[0]] = (1 / np.linalg.norm(vec20))
                weightmatrix[x[2], x[1]] = (1 / np.linalg.norm(vec21))
                weightmatrix[x[1], x[2]] = (1 / np.linalg.norm(vec21))

        else:
            # iterate over triangle list an build weight matrix
            for x in self.trilist:
                # compute the directional vectors at the 1st vertex
                vec1 = self.vertlist[x[1]] - self.vertlist[x[0]]
                vec2 = self.vertlist[x[2]] - self.vertlist[x[0]]

                # compute the weight of the edge 0.5 * cot
                tempweight = 0.5 * (1.0 /
                                    np.tan(np.arccos(np.dot(vec1, vec2) /
                                                     (np.linalg.norm(vec1) *
                                                      np.linalg.norm(vec2)))))

                weightmatrix[x[1], x[2]] += tempweight
                weightmatrix[x[2], x[1]] += tempweight

                # compute the directional vectors at the 2nd vertex
                vec1 = self.vertlist[x[0]] - self.vertlist[x[1]]
                vec2 = self.vertlist[x[2]] - self.vertlist[x[1]]

                # compute the weight of the edge 0.5 * cot
                tempweight = 0.5 * (1.0 /
                                    np.tan(np.arccos(np.dot(vec1, vec2) /
                                                     (np.linalg.norm(vec1) *
                                                      np.linalg.norm(vec2)))))

                weightmatrix[x[0], x[2]] += tempweight
                weightmatrix[x[2], x[0]] += tempweight

                # compute the directional vectors at the 3rd vertex
                vec1 = self.vertlist[x[0]] - self.vertlist[x[2]]
                vec2 = self.vertlist[x[1]] - self.vertlist[x[2]]

                # compute the weight of the edge 0.5 * cot
                tempweight = 0.5 * (1.0 /
                                    np.tan(np.arccos(np.dot(vec1, vec2) /
                                                     (np.linalg.norm(vec1) *
                                                      np.linalg.norm(vec2)))))

                weightmatrix[x[0], x[1]] += tempweight
                weightmatrix[x[1], x[0]] += tempweight

        # return the weight matrix
        return weightmatrix

    def laplacianmatrix(self, mode='inv_euclidean'):
        """Compute a laplacian matrix for a triangular mesh

        The method creates a laplacian matrix for a triangular
        mesh using different weighting function.

        Parameters
        ----------
        mode : {'unit', 'inv_euclidean', 'half_cotangent'}, optional
            The methods for determining the edge weights. Using the option
            'unit' all edges of the mesh are weighted by unit weighting
            function, the result is an adjacency matrix. The option
            'inv_euclidean' results in edge weights corresponding to the
            inverse Euclidean distance of the edge lengths. The option
            'half_cotangent' uses the half of the cotangent of the two angles
            opposed to an edge as weighting function. the default weighting
            function is 'inv_euclidean'.

        Returns
        -------
        laplacianmatrix : array, shape (n_points, n_points)
            Matrix, which contains the discrete laplace operator for data
            defined at the vertices of a triangular mesh. The number of
            vertices of the triangular mesh is n_points.

        Examples
        --------

        >>> from spharapy import trimesh as tm
        >>> testtrimesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])
        >>> testtrimesh.laplacianmatrix(mode='inv_euclidean')
        array([[ 0.76344136, -0.4472136 , -0.31622777],
               [-0.4472136 ,  0.72456369, -0.2773501 ],
               [-0.31622777, -0.2773501 ,  0.59357786]])

        """

        # plausibility test of option 'mode'
        if mode not in ('unit', 'inv_euclidean', 'half_cotangent'):
            raise ValueError("Unrecognized mode '%s'" % mode)

        # determine the weight matrix with
        weightmatrix = self.weightmatrix(mode=mode)

        # compute the laplacian matrix
        laplacianmatrix = np.diag(weightmatrix.sum(axis=0)) - weightmatrix

        # return the laplacian matrix
        return laplacianmatrix

    def massmatrix(self, mode='normal'):
        """Mass matrix of a triangular mesh

        The method determines a mass matrix of a triangular mesh.

        Parameters
        ----------

        mode : {'normal', 'lumped'}, optional
            The `mode` parameter can be used to select whether a normal mass
            matrix or a lumped mass matrix is to be determined.

        Returns
        -------
        massmatrix : array, shape (n_points, n_points)
            Symmetric matrix, which contains the mass values for each edge and
            vertex for the FEM approch. The number of vertices of the
            triangular mesh is n_points.

        Examples
        --------

        >>> from spharapy import trimesh as tm
        >>> testtrimesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])
        >>> testtrimesh.massmatrix()
        array([[ 0.58333333,  0.29166667,  0.29166667],
               [ 0.29166667,  0.58333333,  0.29166667],
               [ 0.29166667,  0.29166667,  0.58333333]])

        References
        ----------
        :cite:`vallet07,dyer07,zhang07`

        """

        # plausibility test of option 'mode'
        if mode not in ('normal', 'lumped'):
            raise ValueError("Unrecognized mode '%s'" % mode)

        # get the largest index from from triangle list
        maxindex = self.trilist.max()

        # fill the weight matrix with zeros, the size is (maxindex + 1)^2
        massmatrix = np.zeros((maxindex + 1, maxindex + 1), dtype=float)

        if mode == 'lumped':
            # iterate over triangle list an build mass matrix
            for x in self.trilist:
                # compute the area of the triangle
                temparea = area_triangle(self.vertlist[x[0]],
                                         self.vertlist[x[1]],
                                         self.vertlist[x[2]])

                # add to every matrix element belonging to the vertex v(i) of
                # the triangle a 3rd of the triangle area
                massmatrix[x[0], x[0]] += temparea / 3
                massmatrix[x[1], x[1]] += temparea / 3
                massmatrix[x[2], x[2]] += temparea / 3
        else:
            # iterate over triangle list an build mass matrix
            for x in self.trilist:
                # compute the area of the triangle
                temparea = area_triangle(self.vertlist[x[0]],
                                         self.vertlist[x[1]],
                                         self.vertlist[x[2]])

                # add to every matrix element belonging to the edge e(i,j) of
                # the triangle a twelfth of the triangle area
                massmatrix[x[0], x[1]] += temparea / 12
                massmatrix[x[1], x[0]] = massmatrix[x[0], x[1]]
                massmatrix[x[0], x[2]] += temparea / 12
                massmatrix[x[2], x[0]] = massmatrix[x[0], x[2]]
                massmatrix[x[1], x[2]] += temparea / 12
                massmatrix[x[2], x[1]] = massmatrix[x[1], x[2]]

                # add to every matrix element belonging to the vertex v(i) of
                # the triangle a sixth of the triangle area
                massmatrix[x[0], x[0]] += temparea / 6
                massmatrix[x[1], x[1]] += temparea / 6
                massmatrix[x[2], x[2]] += temparea / 6

        # return the mass matrix
        return massmatrix

    def stiffnessmatrix(self):
        """Stiffness matrix of a triangular mesh

        The method determines a stiffness matrix of a triangular mesh.

        Returns
        -------
        stiffmatrix : array, shape (n_points, n_points)
            Symmetric matrix, which contains the stiffness values for each edge
            and vertex for the FEM approch. The number of vertices of the
            triangular mesh is n_points.

        Examples
        --------

        >>> from spharapy import trimesh as tm
        >>> testtrimesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])
        >>> testtrimesh.stiffnessmatrix()
        array([[-0.92857143,  0.64285714,  0.28571429],
               [ 0.64285714, -0.71428571,  0.07142857],
               [ 0.28571429,  0.07142857, -0.35714286]])

        References
        ----------
        :cite:`vallet07`

        """

        # get the largest index from from triangle list
        maxindex = self.trilist.max()

        # fill the weight matrix with zeros, the size is (maxindex + 1)^2
        stiffmatrix = np.zeros((maxindex + 1, maxindex + 1), dtype=float)

        # compute the cot weight matrix
        weightmatrix = self.weightmatrix(mode='half_cotangent')

        # compute and return the stiffness matrix
        stiffmatrix = -np.diag(weightmatrix.sum(axis=0)) + weightmatrix

        return stiffmatrix
