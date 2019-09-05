.. spharapy documentation master file, created by
   sphinx-quickstart on Fri Nov  2 11:08:04 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================
Welcome to SpharaPy
===================

.. only:: html
	  
   What is SpharaPy?
   -----------------

   SpharaPy is a Python implementation of the new approach for spatial
   harmonic analysis (SPHARA) that extends the classical spatial
   Fourier analysis to non-uniformly positioned samples on an
   arbitrary surface in :math:`\mathbb{R}^3`, see also
   :cite:`graichen15`. The basis functions used by SPHARA are
   determined by eigenanalysis of the discrete :term:`Laplace-Beltrami
   operator` defined on a triangular mesh specified by the spatial
   sampling points. The Python toolbox SpharaPy provides classes and
   functions to determine the SPHARA basis functions, to perform data
   analysis and synthesis (SPHARA transform) as well as classes to
   design spatial filters using the SPHARA basis.

   Requirements and installation
   -----------------------------
   
   Required packages:
      * numpy (>=1.16.1)
      * scipy (>=1.2.0)
      * matplotlib (>=3.0.2)

   To install, simply use:
   
   .. code-block:: bash

      $ pip3 install spharapy


   Examples and Usage
   ------------------

   Minimal examples are contained in the source code of the
   package. For more detailed examples please have a look at the
   tutorials.

.. toctree::
   :maxdepth: 2
   :hidden:

   auto_examples/plot_01_quick_start
   introduction
   modules/classes
   auto_examples/index.rst
   glossary
   zzbibliography
    
