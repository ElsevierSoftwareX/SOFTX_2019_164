# -*- coding: utf-8 -*-

"""
The :mod:`spharapy.datasets`: module includes utilities to provide
sample datasets.
"""

from .base import load_minimal_triangular_mesh
from .base import load_simple_triangular_mesh
from .base import load_eeg_256_channel_study


__author__ = "Uwe Graichen"
__copyright__ = "Copyright 2018-2019, Uwe Graichen"
__credits__ = ["Uwe Graichen"]
__license__ = "BSD-3-Clause"
__version__ = "1.0.12"
__maintainer__ = "Uwe Graichen"
__email__ = "uwe.graichen@tu-ilmenau.de"
__status__ = "Release"


__all__ = ['load_minimal_triangular_mesh',
           'load_simple_triangular_mesh',
           'load_eeg_256_channel_study']
