# -*- coding: utf-8 -*-
"""
Base IO code to provide sample datasets
"""

import os
import csv

import numpy as np

__author__ = "Uwe Graichen"
__copyright__ = "Copyright 2018-2019, Uwe Graichen"
__credits__ = ["Uwe Graichen"]
__license__ = "BSD-3-Clause"
__version__ = "1.0.12"
__maintainer__ = "Uwe Graichen"
__email__ = "uwe.graichen@tu-ilmenau.de"
__status__ = "Release"


def load_minimal_triangular_mesh():
    """Returns the triangulation of a single triangle

    The data set consists of a list of three vertices at the unit vectors
    of vector space :math:`\mathbb{R}^3`and a list of a single
    triangle.

    ===================    =
    Number of vertices     3
    Number of triangles    1
    ===================    =

    Parameters
    ----------
    None

    Returns
    -------
    triangulation : dictionary
        Dictionary-like object containing the triangulation of a single
        triangle. The attributes are: 'vertlist', the list of vertices,
        'trilist', the list of triangles.
    """
    return {'vertlist': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'trilist': [[0, 1, 2]]}


def load_simple_triangular_mesh():
    """Returns the triangulation of a simple triangular mesh

    The data set consists of a triangulation of an unit hemisphere.

    ===================    ===
    Number of vertices     131
    Number of triangles    232
    ===================    ===

    Parameters
    ----------
    None

    Returns
    -------
    triangulation : dictionary
        Dictionary-like object containing the triangulation of a
        simple triangular mesh. The attributes are: 'vertlist', the
        list of vertices, 'trilist', the list of triangles.
    """
    # import vertices data
    path_to_vertdata = os.path.join(os.path.dirname(__file__),
                                    'data/simple_mesh_vert.csv')
    with open(path_to_vertdata, 'r') as f:
        reader = csv.reader(f)
        datavertices = [[float(y) for y in x]
                        for x in list(reader)]
        f.close()

    # import of triangle list
    path_to_tridata = os.path.join(os.path.dirname(__file__),
                                   'data/simple_mesh_tri.csv')
    with open(path_to_tridata, 'r') as f:
        reader = csv.reader(f)
        datatriangles = [[int(y) for y in x]
                         for x in list(reader)]
        f.close()

    return {'vertlist': datavertices,
            'trilist': datatriangles}


def load_eeg_256_channel_study():
    """Load sensor setup and measured EEG data

    The data set consists of a triangulation of a 256 channel
    equidistant EEG cap and EEG data from previously performed
    experiment addressing the cortical activation related to
    somatosensory-evoked potentials (SEP). During the experiment the
    median nerve of the right forearm was stimulated by bipolar
    electrodes (stimulation rate: 3.7 Hz, interstimulus interval: 270
    ms, stimulation strength: motor plus sensor threshold
    :cite:`mauguiere99,cruccu08`, constant current rectangular pulse
    wave impulses with a length of 50 \mu s, number of stimulations:
    6000). Data were sampled at 2048 Hz and software high-pass (24
    dB/oct, cutoff-frequency 2 Hz) and notch (50 Hz and two harmonics)
    filtered. All trials were manually checked for artifacts, the
    remaining trials were averaged, see also S1 data set in
    :cite:`graichen15`.

    ===================    ========================================
    Number of vertices     256
    Number of triangles    480
    SEP Data (EEG)         256 channels, 369 time samples
    Time range             50 ms before to 130 ms after stimulation
    Sampling frequency     2048 Hz
    ===================    ========================================

    Parameters
    ----------
    None

    Returns
    -------
    triangulation and EEG data: dictionary
        Dictionary-like object containing the triangulation of a
        simple triangular mesh. The attributes are: 'vertlist', the
        list of vertices, 'trilist', the list of triangles,
        'labellist' the list of labels of the EEG channels, 'eegdata',
        an array containing the EEG data.
    """
    # import vertices data
    path_to_vertdata = os.path.join(os.path.dirname(__file__),
                                    'data/eeg_256_channels_vert.csv')
    with open(path_to_vertdata, 'r') as f:
        reader = csv.reader(f)
        datavertices = [[float(y) for y in x]
                        for x in list(reader)]
        f.close()

    # import of triangle list
    path_to_tridata = os.path.join(os.path.dirname(__file__),
                                   'data/eeg_256_channels_tri.csv')
    with open(path_to_tridata, 'r') as f:
        reader = csv.reader(f)
        datatriangles = [[int(y) for y in x]
                         for x in list(reader)]
        f.close()

    # import the sensor labels
    path_to_labeldata = os.path.join(os.path.dirname(__file__),
                                     'data/eeg_256_channels_label.csv')
    with open(path_to_labeldata, 'r') as f:
        reader = csv.reader(f)
        datalabels = list(reader)
        f.close()

    # import SEP EEG data
    path_to_eegdata = os.path.join(os.path.dirname(__file__),
                                   'data/eeg_256_channels_sep_data.csv')
    with open(path_to_eegdata, 'r') as f:
        reader = csv.reader(f)
        eegdata = [[float(y) for y in x]
                   for x in list(reader)]
        eegdata = np.asarray(eegdata)
        f.close()

    return {'vertlist': datavertices,
            'trilist': datatriangles,
            'labellist': datalabels,
            'eegdata': eegdata}


if __name__ == "__main__":
    minimesh = load_minimal_triangular_mesh()
    print((minimesh['trilist']))
    print((minimesh['vertlist']))

    simplemesh = load_simple_triangular_mesh()
    print((simplemesh['trilist']))
    print((simplemesh['vertlist']))

    sepstudy = load_eeg_256_channel_study()
    print((sepstudy['trilist']))
    print((sepstudy['vertlist']))
    print((sepstudy['labellist']))
    print((sepstudy['eegdata'].shape))
