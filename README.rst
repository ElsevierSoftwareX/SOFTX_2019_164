Sphara Implementation in Python
-------------------------------

SpharaPy is a Python implementation of the new approach for spatial
harmonic analysis (SPHARA) that extends the classical spatial Fourier
analysis to non-uniformly positioned samples on an arbitrary surface
in R^3, see also [graichen2015]. The basis functions used by SPHARA
are determined by eigenanalysis of the discrete Laplace-Beltrami
operator defined on a triangular mesh specified by the spatial
sampling points. The Python toolbox SpharaPy provides classes and
functions to determine the SPHARA basis functions, to perform data
analysis and synthesis (SPHARA transform) as well as classes to design
spatial filters using the SPHARA basis.

Requirements and installation
-----------------------------

Required software and packages:

- python3 (>=3.6)
- numpy (>=1.16.1)
- scipy (>=1.2.0)
- matplotlib (>=3.0.2)

To install, simply use: ``pip3 install spharapy``


Examples and Usage
------------------

Minimal examples are contained in the source code of the package. For
more detailed examples please have a look at the tutorials.
