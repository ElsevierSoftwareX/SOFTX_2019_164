.. currentmodule:: spharapy

.. _glossary:

========================
Glossary of Common Terms
========================

.. glossary::

   Boundary condition
       In partial differential equations, boundary conditions (BC) are
       constraints of the solution function :math:`u` for a given
       domain :math:`D`. Thus, the values of the function are
       specified on the boundary (in the topological sense) of the
       considered domain :math:`D`. Neumann and Dirichlet boundary
       conditions are frequently used. The Python implementation of
       SPHARA uses the Neumann boundary condition in the solution
       of the Laplacian eigenvalue problem.
   
   EEG
       EEG is an electrophysiological method for measuring the
       electrical activity of the brain by recording potentials on the
       surface of the head.

   Finite Element Method
       The Finite Element Method (FEM) is a approach to solve (partial
       differential) equations, where continuous values are
       approximated as a set of values at discrete points. For the
       approximation nodal basis functions are used.

   Laplace-Beltrami operator
       The generalized Laplace operator, that can applied on functions
       defined on surfaces in Euclidean space and, more generally, on
       Riemannian and pseudo-Riemannian manifolds. For triangulated
       manifolds, there are several methods to discretize the
       Laplace-Beltrami operator.

   Triangular mesh
       A triangular mesh is  a piecewise planar approximation of a
       smooth surface in :math:`\mathbb{R}^3` using triangles. The
       triangles of the mesh are connected by their common edges or
       corners. The sample points used for the approximation are the
       verices :math:`\vec{c} \in V` with :math:`\vec{v}_i \in
       \mathbb{R}^3`. A triangle :math:`t` is defined by three indices
       to the list of vertices. Thus, a triangular grid is represented
       by a list of vertices and a list of triangles.
