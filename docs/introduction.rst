.. title:: The theoretical background in a nutshell: contents

.. _introduction:

==================================================
SPHARA -- The theoretical background in a nutshell
==================================================

.. toctree::
   :maxdepth: 2
   :numbered:


Motivation
==========

The discrete Fourier analysis of 2D data defined on a flat surface and
represented by a Cartesian or a regular grid is very common in digital
image processing and a fundamental tool in many applications. For such
data, the basis functions (BF) for the Fourier transformation are
usually implicitly specified in the transformation rule, compare
:cite:`rao10`.

However, in many applications the sensors for data acquisition are not
located on a flat surface and can not be represented by Cartesian or
regular grids. An example from the field of biomedical engineering for
non-regular sensor positions is the electroencephalography
(:term:`EEG`). In :term:`EEG`, the sensors are placed at predetermined
positions at the head surface, a surface in space
:math:`\mathbb{R}^3`. The positions of the sensors of these systems
can be described by means of triangular meshes. Because of the
particular sensor arrangement, the spatial analysis of multi-sensor
data can not be performed using the standard 2D Fourier
analysis. However, a spatial Fourier analysis can be also very useful
for spatially irregularly sampled data.

In this Python package we implement a new method for SPatial HARmonic
Analysis (SPHARA) of multisensor data using the eigenbasis of the
:term:`Laplace-Beltrami operator` of the meshed surface of sensor
positions. Using this approach, basis functions of spatial harmonics
for arbitrary arrangements of sensors can be generated. The recorded
multisensor data are decomposed by projection into the space of the
basis functions. For a much more detailed introduction of the
theoretical principles of SPHARA see also :cite:`graichen15`.

Discrete representation of surfaces
===================================

For the discrete case we assume that the sensors are located on an
arbitrary surface, which is represented by a triangular mesh in
:math:`\mathbb{R}^3`. The mesh :math:`M = \{V,E,T\}` consists of
vertices :math:`v \in V`, edges :math:`e \in E` and triangles :math:`t
\in T`. Each vertex :math:`v_i \in \mathbb{R}^3` represents a sensor
position. The number of vertices, edges and triangles of :math:`M` are
defined by :math:`|V|`, :math:`|E|` and :math:`|T|`, respectively. The
neighborhood :math:`i^{\star}` for a vertex :math:`v_i \in V` is
defined by :math:`i^{\star} = \{v_x \in V: e_{ix} \in E\}`, see
:numref:`fig1` (b). The number of neighbors of :math:`v_i` is
:math:`n_i = |i^{\star}|`. The angles :math:`\alpha_{ij}` and
:math:`\beta_{ij}` are located opposed to the edge :math:`e_{ij}`. The
triangles :math:`t_a` and :math:`t_b`, defined by the vertices
:math:`(v_i, v_j, v_o)` and :math:`(v_i, v_k, v_j)`, share the edge
:math:`e_{ij}`. The set of triangles sharing the vertex :math:`v_i` is
given by :math:`i^{\triangledown} = \{t_x \in T : v_i \in t_x\}`. The
area of a triangle :math:`t` is given by :math:`|t|`. An example for
these mesh components is illustrated in :numref:`fig1` (b).

.. _fig1:

.. figure:: _static/discretization_bary.*
   
   **The approximation of the Laplace-Beltrami operator.** (a)
   continuous representation; (b) discrete representation. The
   neighborhood :math:`i^{\star}` of vertex :math:`v_i` consists of
   the vertices :math:`\{v_x \in V: e_{ix} \in E\}`. Either the length
   of :math:`e_{ij}` or the size of the two angles :math:`\alpha_{ij}`
   and :math:`\beta_{ij}` opposed to the edge :math:`e_{ij}` are used
   to estimate the weight :math:`w(i,j)` for :math:`e_{ij}`. The two
   triangles :math:`t_a` and :math:`t_b` both share the edge
   :math:`e_{ij}`; (c) the area of the barycell
   :math:`A^{\mathrm{B}}_i` for the vertex :math:`v_i`.
	 
Discrete Laplace-Beltrami Operators
===================================

A function :math:`\vec{f}` is defined for all vertices :math:`v_i \in
V`, it applies :math:`\vec{f}: v_i \rightarrow \mathbb{R}` with
:math:`i = 1, \ldots, |V|`. A discretization :math:`\Delta_D` of the
:term:`Laplace-Beltrami operator` for :math:`\vec{f}` is given by

.. math::
   \Delta_D \vec{f}_i = b^{-1}_i\sum_{x \in i^{\star}} w(i, x)\left(\vec{f}_i -
   \vec{f}_x\right)\,,
   :label: eq:eq3

with the weighting function :math:`w(i, x)` for edges :math:`e_{ix}
\in E` and the normalization coefficient :math:`b_i` for the vertex
:math:`v_i`. For practical applications it is convenient to transform
equation :eq:`eq:eq3` into matrix notation. The elements of the
matrices :math:`B^{-1}` and :math:`S` are determined using the
coefficients :math:`b_i` and :math:`w(i, x)` of equation
:eq:`eq:eq3`. :math:`B^{-1}` is a diagonal matrix, the elements are

.. math::
   B^{-1}_{ij} = 
   \begin{cases}
   b^{-1}_i & \text{if}\quad i = j\\
   0 & \text{otherwise}\,,
   \end{cases}
   :label: eq:blaplacianmat
	
and the entries of :math:`S` are

.. math::
   S_{ij} = 
   \begin{cases}
   \sum_{x \in i^{\star}} w(i, x) & \text{if}\quad i = j\\
   -w(i, x) & \text{if}\quad e_{ix} \in E\\
   0 & \text{otherwise}\,.
   \end{cases}
   :label: eq:slaplacianmat

A Laplacian matrix :math:`L` can be expressed as product of a diagonal
matrix :math:`B^{-1}` and a matrix :math:`S`

.. math::
   L = B^{-1} \, S\,,
   :label: dislaplacianmatnotationprod

compare also :cite:`zhang10`. The size of the Laplacian
matrix :math:`L` for a mesh :math:`M` is :math:`n \times n`, with
:math:`n=|V|`. Using the Laplacian matrix `L`, :math:`\Delta_D`
applied to :math:`\vec{f}` can be written as

.. math::
   \Delta_D \vec{f} = - L \vec{f}\,.
   :label: dislaplacianmatnotation

In the following we present four different approaches to discretize
the :term:`Laplace-Beltrami operator`, a graph-theoretical approach and
three geometric approaches.

First, we look at the graph-theoretical approach, where the
coordinates of the positions of the vertices are not considered. The
topological Laplacian results from equation :eq:`eq:eq3` by using
:math:`w(i,x) = b^{-1}_i = 1`, see also
:cite:`taubin95,chung97,zhang07`. The graph-theoretical approach will
be referred to as TL later in the text.

Second, for inhomogeneous triangular meshes, where the distances
between vertices and the sizes of angles and triangles are different,
the weighting function :math:`w` has to be adapted according to the mesh
geometry. In these approaches, the positions of the vertices are also
considered. They are referred to as geometric approaches. There
are different approaches to treat inhomogeneous meshes.

The first possibility is to use the Euclidean distance of adjacent
vertices raised to the power of a value :math:`\alpha`. For equation
:eq:`eq:eq3` the coefficients :math:`b^{-1}_i = 1` and :math:`w(i, x) =
||e_{ix}||^{\alpha}` are chosen. A common choice is to use the inverse
of the Euclidean distance with :math:`\alpha = -1`
:cite:`taubin95,fujiwara95`. This approach will be referred to later
as IE.

The second approach for a geometric discretization of the
:term:`Laplace-Beltrami operator` is derived by minimizing the Dirichlet
energy for a triangulated mesh :cite:`pinkall93,polthier02`. It uses
cotangent weights with

.. math::
  w(i,x) = \frac{1}{2}\left(\cot(\alpha_{ix}) +
  \cot(\beta_{ix})\right)\,,
  :label: eq:weightfuncot

with the two angles :math:`\alpha_{ix}` and :math:`\beta_{ix}` opposed
to the edge :math:`e_{ix}`, see :numref:`fig1` (b). For edges on the
boundary of the mesh, the term :math:`\cot(\beta_{ix})` is omitted,
which leads to Neumann :term:`Boundary condition` (BC). A drawback of
using the cotangent weights is that the value representing the
integral of the Laplacian over a 1-ring neighborhood (area of the
:math:`i^{\star}`-neighborhood) is assigned to a point sample
:cite:`zhang07`. To resolve this issue and to guarantee the
correspondence between the continuous and the discrete approaches, the
weights in equation :eq:`eq:weightfuncot` are divided by the area
:math:`A^{\mathrm{B}}_i` of the barycell for the vertex :math:`v_i`
:cite:`meyer03`, resulting in

.. math::
   w(i,x) = \frac{1}{2 A^{\mathrm{B}}_i}\left(\cot(\alpha_{ix}) +
   \cot(\beta_{ix})\right)\,.
   :label: eq:weightfuncotbarycell

The barycell for a vertex :math:`v_i` is framed by a polygonal line
that connects the geometric centroids of triangles in
:math:`i^{\triangledown}` and the midpoints of the adjoined edges
:math:`e_{ix}`, see :numref:`fig1` (c). The area of the
:math:`i^{\star}`-neighborhood for a vertex :math:`v_i`, which is the
area of the triangles that are enclosed by the vertices :math:`v_x \in
i^{\star}`, is referred to as :math:`A^1_i`. Then
:math:`A^{\mathrm{B}}_i` can be determined by
:math:`A^{\mathrm{B}}_{i} = \frac{1}{3} A^1_i`. For the
discretizations using the cotangent weighted formulation, the
parameter :math:`b^{-1}_i` in equation :eq:`eq:eq3` is set to
:math:`b^{-1}_i = 1`. This approach, using cotangent weights will be
referred to as COT later in the manuscript.

The third geometric approach to discretize the Laplace-Beltrami
operator is the :term:`Finite Element Method` (FEM), which is related
to the approach using cotangent weights. Assuming that the function
:math:`f` is piecewise linear and defined by its values :math:`f_i` on
the vertices :math:`v_i` of a triangular mesh, :math:`f` can be
interpolated using nodal basis functions :math:`\psi_i`

.. math::
   f = \sum_{i = 1}^{|V|} f_i \, \psi_i\,.
   :label: eq:piecewise_linear

We use the hat function for :math:`\psi_i`, with

.. math::
   \psi_i(j) = 
   \begin{cases}
   1 & \text{if}\quad i = j\\
   0 & \text{otherwise}\,.
   \end{cases}
   :label: eq:psi_hat

For two functions :math:`f` and :math:`g` defined on :math:`M`, a
scalar product is given by

.. math::
   \int_{M} f g \; \mathrm{d} a =
   \sum_{i = 0}^{|V|} \sum_{j = 0}^{|V|}
   f_i g_i \int_{M} \psi_i \psi_j \;\mathrm{d} a = \left\langle
   \vec{f}, \vec{g} \right\rangle_B\,,
   :label: eq:scalarproductfem

with the area element :math:`\mathrm{d} a` on :math:`M` and the
mass matrix :math:`B`. The sparse mass matrix :math:`B` is given by

.. math::
   B_{ij} = \int_{M} \psi_i \psi_j \,\mathrm{d} a\,.
   :label: eq:defmassmatrix

For the FEM approach using hat functions, the elements of :math:`B`
can be calculated by

.. math::
   B_{ij} = 
   \begin{cases}
   \left( \sum_{t \in i^{\triangledown}} |t|\right) / 6 & \text{if}\quad i = j\\
   \left(|t_a| + |t_b| \right) / 12 & \text{if}\quad e_{ij} \in E\\
   0 & \text{otherwise}\,,
   \end{cases}
   :label: eq:massmat
	   
where :math:`t_a` and :math:`t_b` are the two triangles adjacent to
the edge :math:`e_{ij}`, see :numref:`fig1` (b). For the FEM
discretization of the :term:`Laplace-Beltrami operator` also a stiffness
matrix :math:`S` has to be calculated. The elements of :math:`S_{ij}`
can be estimated using the equations :eq:`eq:slaplacianmat`
and :eq:`eq:weightfuncot`, compare :cite:`dyer07,zhang07,vallet07`.


.. _eigensystems_of_lb_operators:

Eigensystems of Discrete Laplace-Beltrami Operators
===================================================

Desirable properties of the discrete Laplacian :math:`L` are symmetry,
positive weights, positive semi-definiteness, locality, linear
precision and convergence :cite:`wardetzky07`. The symmetry
:math:`L_{ij} = L_{ji}` leads to real eigenvalues and orthogonal
eigenvectors. Positive weights :math:`w(i,j) \ge 0` assure, together
with the symmetry, the positive semi-definiteness of :math:`L`. The
locality of the discrete :term:`Laplace-Beltrami operator` enables the
determination of weights :math:`w(i,j)` using the
:math:`i^{\star}`-neighborhood of a vertex :math:`v_i`, with
:math:`w(i,j) = 0`, if :math:`e_{ij} \notin E`. The linear precision
implies for a linear function :math:`f` defined on vertices
:math:`v_i` that :math:`\Delta_{D} \vec{f}_i = 0` applies, which
ensures the exact recovery of :math:`f` from the samples. The
convergence property provides the convergence from the discrete to the
continuous :term:`Laplace-Beltrami operator` :math:`\Delta_D \rightarrow
\Delta` for a sufficient refinement of the mesh.

The Laplacian matrix :math:`L` for the TL and the IE approach are positive
semi-definite, symmetric and use positive weights. The COT and the FEM
approach do not fulfill the positive weight property, if the mesh
contains triangles with interior angles in the interval :math:`(\pi/2,
\pi)`, for which the cotangent is negative. The TL approach is no
geometric discretization, because it violates the linear precision and
the convergence property. In contrast, the COT and the FEM approach are
geometric discretizations as they fulfill the linear precision and the
convergence property, but they violate the symmetry property. None of
the presented discretization methods fulfill all desirable properties,
see also :cite:`wardetzky07`.

The discrete Laplacian eigenvalue problem for a real and symmetric
Laplacian matrix is given by

.. math::
   L \, \vec{x}_i = \lambda_i \, \vec{x}_i\,,
   :label: eq:eigensystem

with eigenvectors :math:`\vec{x}_i` and eigenvalues :math:`\lambda_i`
of :math:`L`. The Laplacian matrix :math:`L` is real and symmetric for
the TL, the IE and the COT approach. Because :math:`L` is real and
symmetric, eigenvalues :math:`\lambda_i \in \mathbb{R}` with
:math:`\lambda_i \ge 0` are obtained. The eigenvectors
:math:`\vec{x}_i` are real-valued and form a harmonic orthonormal
basis. The corresponding eigenvalues :math:`\lambda_i` can be
considered as spatial frequencies. The eigenvectors :math:`\vec{x}_i`
can be used for a spectral analysis of functions defined on the mesh
:math:`M`. The projection of a discrete function :math:`\vec{f}`
defined on :math:`M` onto the basis of spatial harmonic functions is
performed by the inner product for Euclidean :math:`n`-spaces
:math:`\langle\vec{f}, \vec{x}_i\rangle`. For the matrix :math:`X`,
where the eigenvectors :math:`\vec{x}_i` represent the columns,

.. math::
   X = [\vec{x}_1 \; \vec{x}_2 \cdots \vec{x}_n]\,,

it applies

.. math::
   X^\top X = I\,,
   :label: eq:eigenortho

with the identity matrix :math:`I`.

For the FEM formulation, the BF :math:`\vec{y}_i` are computed
by solving the generalized symmetric definite eigenproblem

.. math::
   S \vec{y}_i = \lambda_i B \vec{y}_i\,.
   :label: eq:eigensystemgeneralized

Thus, the inversion of the mass matrix :math:`B` is avoided. Because
:math:`B^{-1}S` is not symmetric, the eigenvectors :math:`\vec{y}_i` are
real-valued, but not orthonormal with respect to the inner product for
Euclidean :math:`n`-spaces :math:`\left\langle . \right\rangle`. To use these
eigenvectors as BF, the inner product, defined in
equation~(\ref{eq:scalarproductfem}), has to be used

.. math::
   \left\langle \vec{f}, \vec{y}_i \right\rangle_B = \vec{f}^{\,\top} B \,
   \vec{y}_i\,,
   :label: eq:modifiedinnerproduct
	   
which assures the :math:`B`-orthogonality, compare also
\cite{vallet07}. The eigenvectors computed by the FEM approach can be
normalized by using the :math:`B`-relative norm

.. math::
   \vec{\tilde{y}}_i = \frac{\vec{y}_i}{\lVert\vec{y}_i\rVert_{B}}
   \quad \text{with} \quad
   \lVert\vec{y}_i\rVert_{B} = \sqrt{\left\langle \vec{y}_i, \vec{y}_i
   \right\rangle_B}\,.
   :label: eq:eigenvectorsnormalization
	
For a matrix :math:`\tilde{Y}`, where the normalized eigenvectors
:math:`\vec{\tilde{y}}_i` represent the columns

.. math::
   \tilde{Y} = [\vec{\tilde{y}}_1 \; \vec{\tilde{y}}_2 \cdots
   \vec{\tilde{y}}_n]\,,
   :label: eq:femeigenvektoren

it applies

.. math::
   \tilde{Y}^\top B \tilde{Y} = I\,.   
   :label: eq:femidentity

SPHARA as signal processing framework
=====================================

Requirements
------------

To use the eigenvectors of the discrete Laplacian Beltrami
operator in the context of a signal processing framework, it is
necessary that they exhibit certain properties. The eigenvectors have
to form a set of BF. An inner product for the decomposition and for
the reconstruction of the data has to be defined, which is used for
the transformation into the domain of the spatial frequencies and for
the back-transformation into the spatial domain. To be utilized
for practical applications, the transformation from the spatial domain
into the spatial frequency domain has to have linear properties and
should fulfill Parseval's theorem.

Basis functions
---------------

A complete set of linearly independent vectors can be used as
basis. For real and symmetric Laplacian matrices :math:`L` the
orthonormality of the eigenvectors is given inherently, see equations
:eq:`eq:eigensystem` and :eq:`eq:eigenortho`. For the FEM approach,
the orthonormality of the eigenvectors is assured explicitly, see
equations :eq:`eq:eigensystemgeneralized` to :eq:`eq:femidentity`. The
property of orthogonality includes the linear independence. To use the
eigenvectors as BF, they must further fulfill the property of
completeness. The completeness can be shown by the dimension theorem
for vector spaces. The dimensionality is equal for both the spatial
representation and the representation in the spatial frequency
domain. For a mesh with :math:`n` vertices, :math:`n` unit impulse
functions are used as BF for the spatial representation. For the same
mesh, we obtain :math:`n` discrete spatial harmonic functions
(eigenvectors) for the representation using spatial frequencies. The
calculated eigenvectors are orthonormal and complete; therefore, they
can be used as orthonormal BF.

.. _sphara_analysis_synthesis:

Anaylsis and synthesis
----------------------

For the anaylsis of discrete data defined on the vertices of the
triangular mesh, the inner product is used (transformation from
spatial domain to spatial frequency domain). For an anaylsis using the
eigenvectors of a symmetric Laplacian matrix :math:`L` (TL, IE and
COT), the vector space inner product is applied. The coefficient
:math:`c_i` for a single spatial harmonic BF :math:`\vec{x}_i` can be
determined by

.. math::
   c_i = \langle\vec{f}, \vec{x}_i\rangle\,.
   :label: eq:decomposition_standard

The transformation from the spatial into the spatial frequency domain
is computed by

.. math::
  \vec{c}^{\,\top} = \vec{f}^{\,\top} \, X\,.
  :label: eq:transformation_standard

For an analysis using eigenvectors computed by the FEM approach, the
inner product that assures the :math:`B`-orthogonality needs to be
applied

.. math::
   c_i =   \left\langle \vec{f}, \vec{\tilde{y}}_i \right\rangle_B =
   \vec{f}^{\,\top} B \, \vec{\tilde{y}}_i\,.
   :label: eq:decomposition_fem

The transformation from the spatial into the spatial frequency
domain is then be computed by

.. math::
   \vec{c}^{\,\top} = \vec{f}^{\,\top} B \, \tilde{Y}\,.
   :label: eq:transformation_fem

Discrete data are synthesized using the linear combination of the
coefficients :math:`c_i` and the corresponding BF :math:`\vec{x}_i` or
:math:`\vec{\tilde{y}}_i`

.. math::
   \vec{f} = \sum_{i = 1}^n c_i \, \vec{x}_i
   :label: eq:reconstruction

or

.. math::
   \vec{f}^{\,\top} = \vec{c}^{\,\top} \, \tilde{Y}^{\,\top}\,.
   :label: eq:reconstruction_mat

Spatial filtering using SPHARA
==============================

At the end of this short introduction we show the design of a spatial
filter as a practical application of SPHARA. The prerequisite for the
successful use of SPHARA-based filters is the separability of useful
signal and interference in the spatial SPHARA spectrum. This applies,
for example, to :term:`EEG`. In :term:`EEG`, the low-frequency SPHARA basis
functions provide the main contribution to signal power. In contrast,
single channel dropouts and spatially uncorrelated sensor noise
exhibit an almost equally distributed spatial SPHARA spectrum, compare
:cite:`graichen15` and tutorial :ref:`sphara_analysis_eeg`.

A filter matrix :math:`F` can be determined by
      
.. math::
   F = R \cdot X \cdot (R \cdot X)^{\intercal}\,.

The matrix :math:`X` contains columnwise the SPHARA basis functions
and the matrix :math:`R` is a selection matrix, that contains an 1 on
the main diagonal if the corresponding SPHARA basis function from
:math:`X` is chosen. All other elements of this matrix are 0.

If the Laplace-Beltrami Operator with FEM discretization is used to
calculate the SPHARA basis functions, the mass matrix :math:`B` must
be added to the equation to compute the filter matrix

.. math::
   F_{\mathrm{FEM}} = B \cdot R \cdot X \cdot (R \cdot X)^{\intercal}\,.

The spatial SPHARA filter is applied to the data by multiplying the
matrix containing the data :math:`D` by the filter matrix :math:`F`

.. math::
   \tilde{D} = D \cdot F\,.

The matrix :math:`D` contains data, time samples in rows and spatial
samples in columns and the matrix :math:`\tilde{D}` the filtered data,
see also tutorial :ref:`sphara_filtering_eeg`.
