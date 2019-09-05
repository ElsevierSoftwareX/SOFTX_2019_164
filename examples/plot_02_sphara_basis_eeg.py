# -*- coding: utf-8 -*-
r""".. _sphara_basis_eeg:


Determination of the SPHARA basis functions for an EEG sensor setup
===================================================================

.. topic:: Section contents

   This tutorial introduces the steps necessary to determine a
   generalized spatial Fourier basis for an :term:`EEG` sensor setup
   using SpharaPy. The special properties of the different
   discretization approaches of the Laplace-Beltrami operator will be
   discussed.


Introduction
------------

A Fourier basis is a solution to Laplace's eigenvalue problem

.. math::

   L \vec{x} = \lambda \vec{x}\,,\qquad\qquad(1)

with the discrete :term:`Laplace-Beltrami operator` in matrix notation
:math:`L \in \mathbb{R}^{M \times N}`, the eigenvectors
:math:`\vec{x}` containing the harmonic functions and the eigenvalues
:math:`\lambda` the natural frequencies.

By solving a Laplace eigenvalue problem, it is also possible to
determine a basis for a spatial Fourier analysis. Often in practical
applications, a measured quantity to be subjected to a Fourier
analysis is only known at spatially discrete sampling points (the
sensor positions). An arbitrary arrangement of sample points on a
surface in three-dimensional space can be described by means of a
:term:`triangular mesh`. In the case of an :term:`EEG` system, the
sample positions (the vertices of the triangular mesh) are the
locations of the sensors arranged on the head surface. A SPHARA basis
is a solution of a Laplace eigenvalue problem for the given triangle
mesh, that can be obtained by discretizing a Laplace-Beltrami operator
for the mesh and solving the Laplace eigenvalue problem in equation
(1). The SpharaPy package provides three methods for the
discretization of the Laplace-Beltrami operator; unit weighting of the
edges and weighting with the inverse of the Euclidean distance of the
edges, and a FEM approach. For more detailed information please refer
to :ref:`introduction` and :ref:`eigensystems_of_lb_operators`.

"""

######################################################################
# At the beginning we import three modules of the SpharaPy package as
# well as several other packages and single functions of packages.

# Code source: Uwe Graichen
# License: BSD 3 clause

# import modules from spharapy package
import spharapy.trimesh as tm
import spharapy.spharabasis as sb
import spharapy.datasets as sd

# import additional modules used in this tutorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


######################################################################
# Specification of the spatial configuration of the EEG sensors
# -------------------------------------------------------------
#
# Import information about EEG sensor setup of the sample data set
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In this tutorial we will determine a SPHARA basis for a 256 channel
# :term:`EEG` system with equidistant layout. The data set is one of
# the example data sets contained in the SpharaPy toolbox, see
# :mod:`spharapy.datasets` and
# :func:`spharapy.datasets.load_eeg_256_channel_study`.

# loading the 256 channel EEG dataset from spharapy sample datasets
mesh_in = sd.load_eeg_256_channel_study()

######################################################################
# The dataset includes lists of vertices, triangles, and sensor
# labels, as well as :term:`EEG` data from previously performed
# experiment addressing the cortical activation related to
# somatosensory-evoked potentials (SEP).

print(mesh_in.keys())

######################################################################
# The triangulation of the :term:`EEG` sensor setup consists of 256
# vertices and 480 triangles.

vertlist = np.array(mesh_in['vertlist'])
trilist = np.array(mesh_in['trilist'])
print('vertices = ', vertlist.shape)
print('triangles = ', trilist.shape)

######################################################################

fig = plt.figure()
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
ax = fig.gca(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('The triangulated EEG sensor setup')
ax.view_init(elev=20., azim=80.)
ax.set_aspect('auto')
ax.plot_trisurf(vertlist[:, 0], vertlist[:, 1], vertlist[:, 2],
                triangles=trilist, color='lightblue', edgecolor='black',
                linewidth=0.5, shade=True)
plt.show()

######################################################################
# Create a SpharaPy TriMesh instance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the next step we create an instance of the class
# :class:`spharapy.trimesh.TriMesh` from the list of vertices and
# triangles.

# create an instance of the TriMesh class
mesh_eeg = tm.TriMesh(trilist, vertlist)

######################################################################
# The class :class:`spharapy.trimesh.TriMesh` provides a
# number of methods to determine certain properties of the triangle
# mesh required to generate the SPHARA basis, listed below:

# print all implemented methods of the TriMesh class
print([func for func in dir(tm.TriMesh) if not func.startswith('__')])

######################################################################
# Determining SPHARA bases using different discretisation approaches
# ------------------------------------------------------------------
# Computing the basis functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the final step of the tutorial we will calculate SPHARA bases for
# the given :term:`EEG` sensor setup. For this we create three
# instances of the class :class:`spharapy.spharabasis.SpharaBasis`. We
# use the three discretization approaches implemented in this class
# for the Laplace-Beltrami operator: unit weighting ('unit') and
# inverse Euclidean weigthing ('inv_euclidean') of the edges of the
# triangular mesh as well as the FEM discretization ('fem')

# 'unit' discretization
sphara_basis_unit = sb.SpharaBasis(mesh_eeg, 'unit')
basis_functions_unit, natural_frequencies_unit = sphara_basis_unit.basis()

# 'inv_euclidean' discretization
sphara_basis_ie = sb.SpharaBasis(mesh_eeg, 'inv_euclidean')
basis_functions_ie, natural_frequencies_ie = sphara_basis_ie.basis()

# 'fem' discretization
sphara_basis_fem = sb.SpharaBasis(mesh_eeg, 'fem')
basis_functions_fem, natural_frequencies_fem = sphara_basis_fem.basis()

######################################################################
# Visualization the basis functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The first 15 spatially low-frequency SPHARA basis functions are
# shown below, starting with DC at the top left.
#
# SPHARA basis using the discretization approache 'unit'
# """"""""""""""""""""""""""""""""""""""""""""""""""""""

# sphinx_gallery_thumbnail_number = 2
figsb1, axes1 = plt.subplots(nrows=5, ncols=3, figsize=(8, 12),
                             subplot_kw={'projection': '3d'})
for i in range(np.size(axes1)):
    colors = np.mean(basis_functions_unit[trilist, i + 0], axis=1)
    ax = axes1.flat[i]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=60., azim=80.)
    ax.set_aspect('auto')
    trisurfplot = ax.plot_trisurf(vertlist[:, 0], vertlist[:, 1],
                                  vertlist[:, 2], triangles=trilist,
                                  cmap=plt.cm.bwr,
                                  edgecolor='white', linewidth=0.)
    trisurfplot.set_array(colors)
    trisurfplot.set_clim(-0.15, 0.15)

cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.85,
                       orientation='horizontal', fraction=0.05, pad=0.05,
                       anchor=(0.5, -4.5))

plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
plt.show()

######################################################################
# SPHARA basis using the discretization approache 'inv_euclidean'
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

figsb1, axes1 = plt.subplots(nrows=5, ncols=3, figsize=(8, 12),
                             subplot_kw={'projection': '3d'})
for i in range(np.size(axes1)):
    colors = np.mean(basis_functions_ie[trilist, i + 0], axis=1)
    ax = axes1.flat[i]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=60., azim=80.)
    ax.set_aspect('auto')
    trisurfplot = ax.plot_trisurf(vertlist[:, 0], vertlist[:, 1],
                                  vertlist[:, 2], triangles=trilist,
                                  cmap=plt.cm.bwr,
                                  edgecolor='white', linewidth=0.)
    trisurfplot.set_array(colors)
    trisurfplot.set_clim(-0.18, 0.18)

cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.85,
                       orientation='horizontal', fraction=0.05, pad=0.05,
                       anchor=(0.5, -4.5))

plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
plt.show()

######################################################################
# SPHARA basis using the discretization approache 'fem'
# """""""""""""""""""""""""""""""""""""""""""""""""""""

figsb1, axes1 = plt.subplots(nrows=5, ncols=3, figsize=(8, 12),
                             subplot_kw={'projection': '3d'})
for i in range(np.size(axes1)):
    colors = np.mean(basis_functions_fem[trilist, i + 0], axis=1)
    ax = axes1.flat[i]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=60., azim=80.)
    ax.set_aspect('auto')
    trisurfplot = ax.plot_trisurf(vertlist[:, 0], vertlist[:, 1],
                                  vertlist[:, 2], triangles=trilist,
                                  cmap=plt.cm.bwr,
                                  edgecolor='white', linewidth=0.)
    trisurfplot.set_array(colors)
    trisurfplot.set_clim(-0.01, 0.01)

cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.85,
                       orientation='horizontal', fraction=0.05, pad=0.05,
                       anchor=(0.5, -4.5))

plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
plt.show()
