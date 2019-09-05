# -*- coding: utf-8 -*-
r""".. _sphara_filtering_eeg:


Spatial SPHARA filtering of EEG data
====================================

.. topic:: Section contents

   In this tutorial we show how to use the SPHARA basis functions to
   design a spatial low pass filter for application to EEG data. The
   FEM discretization of the Laplace-Beltrami operator is used to
   calculate the SPHARA basic functions that are used for the SPHARA
   low pass filter. The applicability of the filter is shown using an
   EEG data set that is disturbed by white noise in different noise
   levels.


Introduction
------------

The human head as a volume conductor exhibits spatial low-pass filter
properties. For this reason, the potential distribution of the EEG on
the scalp surface can be represented by a few low-frequency SPHARA
basis functions, compare :ref:`sphara_analysis_eeg`. In contrast,
single channel dropouts and spatially uncorrelated sensor noise
exhibit an almost equally distributed spatial SPHARA spectrum. This
fact can be exploited for the design of a spatial filter for the
suppression of uncorrelated sensor noise.

"""

######################################################################
# At the beginning we import three modules of the SpharaPy package as
# well as several other packages and single functions from
# packages.

# Code source: Uwe Graichen
# License: BSD 3 clause

# import modules from spharapy package
import spharapy.trimesh as tm
import spharapy.spharafilter as sf
import spharapy.datasets as sd

# import additional modules used in this tutorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


######################################################################
# Import the spatial configuration of the EEG sensors and the SEP data
# --------------------------------------------------------------------
# In this tutorial we will apply a spatial SPHARA filter to SEP data
# of a single subject recorded with a 256 channel EEG system with
# equidistant layout. The data set is one of the example data sets
# contained in the SpharaPy toolbox.

# loading the 256 channel EEG dataset from spharapy sample datasets
mesh_in = sd.load_eeg_256_channel_study()

######################################################################
# The dataset includes lists of vertices, triangles, and sensor
# labels, as well as EEG data from previously performed experiment
# addressing the cortical activation related to somatosensory-evoked
# potentials (SEP).

print(mesh_in.keys())

######################################################################
# The triangulation of the EEG sensor setup consists of 256 vertices
# and 480 triangles. The EEG data consists of 256 channels and 369
# time samples, 50 ms before to 130 ms after stimulation. The sampling
# frequency is 2048 Hz.

vertlist = np.array(mesh_in['vertlist'])
trilist = np.array(mesh_in['trilist'])
eegdata = np.array(mesh_in['eegdata'])
print('vertices = ', vertlist.shape)
print('triangles = ', trilist.shape)
print('eegdata = ', eegdata.shape)

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

x = np.arange(-50, 130, 1/2.048)
figeeg = plt.figure()
axeeg = figeeg.gca()
axeeg.plot(x, eegdata[:, :].transpose())
axeeg.set_xlabel('t/ms')
axeeg.set_ylabel('V/µV')
axeeg.set_title('SEP data')
axeeg.set_ylim(-3.5, 3.5)
axeeg.set_xlim(-50, 130)
axeeg.grid(True)
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
# SPHARA filter using FEM discretisation
# --------------------------------------
# Create a SpharaPy SpharaFilter instance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the following step of the tutorial we determine an instance of
# the class SpharaFilter, which is used to execute the spatial
# filtering. For the determination of the SPHARA basis we use a
# Laplace-Beltrami operator, which is discretized by the FEM approach.

sphara_filter_fem = sf.SpharaFilter(mesh_eeg, mode='fem',
                                    specification=20)
basis_functions_fem, natural_frequencies_fem = sphara_filter_fem.basis()

######################################################################
# Visualization the basis functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The first 15 spatially low-frequency SPHARA basis functions are
# shown below, starting with DC at the top left.
#

figsb1, axes1 = plt.subplots(nrows=5, ncols=3, figsize=(8, 12),
                             subplot_kw={'projection': '3d'})
for i in range(np.size(axes1)):
    colors = np.mean(basis_functions_fem[trilist, i + 0], axis=1)
    ax = axes1.flat[i]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=60., azim=80.)
    ax.set_aspect('equal')
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


######################################################################
# SPHARA filtering of the EEG data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the next step we perform the SPHARA filtering of the EEG
# data. As a result, the butterfly plots of all channels of the EEG
# with and without filtering is compared. For the marked time samples
# also topo plots are provided.

# perform the SPHARA filtering
sphara_filt_eegdata = sphara_filter_fem.filter(eegdata.transpose()).transpose()

figsteeg, (axsteeg1, axsteeg2) = plt.subplots(nrows=2, figsize=(8, 6.5))

axsteeg1.axvline(13, color='red')
axsteeg1.axvline(19, color='blue')
axsteeg1.axvline(30, color='green')
axsteeg1.plot(x, eegdata[:, :].transpose())
axsteeg1.set_title('Unfiltered EEG data')
axsteeg1.set_ylabel('V/µV')
axsteeg1.set_xlabel('t/ms')
axsteeg1.set_ylim(-2.5, 2.5)
axsteeg1.set_xlim(-50, 130)
axsteeg1.grid(True)

axsteeg2.axvline(13, color='red')
axsteeg2.axvline(19, color='blue')
axsteeg2.axvline(30, color='green')
axsteeg2.plot(x, sphara_filt_eegdata[:, :].transpose())
axsteeg2.set_title('SPHARA low-pass filtered EEG data, 20 BF, fem')
axsteeg2.set_ylabel('V/µV')
axsteeg2.set_xlabel('t/ms')
axsteeg2.set_ylim(-2.5, 2.5)
axsteeg2.set_xlim(-50, 130)
axsteeg2.grid(True)

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, hspace=0.3)
plt.show()

######################################################################

time_pts = [129, 141, 164]
figsf1, axessf1 = plt.subplots(nrows=2, ncols=3, figsize=(8, 5),
                               subplot_kw={'projection': '3d'})

for i in range(2):
    for j in range(3):
        if i == 0:
            colorssf1 = np.mean(eegdata[trilist, time_pts[j]], axis=1)
        else:
            colorssf1 = np.mean(sphara_filt_eegdata[trilist, time_pts[j]],
                                axis=1)
        ax = axes1.flat[i]
        axessf1[i, j].set_xlabel('x')
        axessf1[i, j].set_ylabel('y')
        axessf1[i, j].set_zlabel('z')
        axessf1[i, j].view_init(elev=60., azim=80.)
        axessf1[i, j].set_aspect('equal')

        trisurfplot = axessf1[i, j].plot_trisurf(vertlist[:, 0],
                                                 vertlist[:, 1],
                                                 vertlist[:, 2],
                                                 triangles=trilist,
                                                 cmap=plt.cm.bwr,
                                                 edgecolor='white',
                                                 linewidth=0.)
        trisurfplot.set_array(colorssf1)
        trisurfplot.set_clim(-2., 2)

cbar = figsb1.colorbar(trisurfplot, ax=axessf1.ravel().tolist(), shrink=0.85,
                       orientation='horizontal', fraction=0.05, pad=0.05,
                       anchor=(0.5, 0))

plt.subplots_adjust(left=0.0, right=1.0, bottom=0.2, top=1.0)
plt.show()


######################################################################
# Application of the the spatial SPHARA filter to data with artificial noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In a final step the EEG data are disturbed by white noise with
# different noise levels (3dB, 0dB and -3dB). A spatial low-pass
# SPHARA filter with 20 basis functions is applied to these data. The
# results of the filtering are shown below.

# vector with noise levels in dB
db_val_vec = [3, 0, -3]

# compute the power of the SEP data
power_sep = np.sum(np.square(np.absolute(eegdata))) / eegdata.size

# compute a vector with standard deviations of the noise in relation
# to signal power for the given noise levels
noise_sd_vec = list(map(lambda db_val:
                    np.sqrt(power_sep / (10 ** (db_val / 10))),
                        db_val_vec))

# add the noise to the EEG data
eegdata_noise = list(map(lambda noise_sd:
                         eegdata + np.random.normal(0, noise_sd, [256, 369]),
                         noise_sd_vec))

# filter the EEG data containing the artificial noise
eegdata_noise_filt = list(map(lambda eeg_noise:
                              (sphara_filter_fem.filter(eeg_noise.transpose()).
                               transpose()),
                              eegdata_noise))

######################################################################

figfilt, axesfilt = plt.subplots(nrows=4, ncols=2, figsize=(8, 10.5))

axesfilt[0, 0].plot(x, eegdata[:, :].transpose())
axesfilt[0, 0].set_title('EEG data')
axesfilt[0, 0].set_ylabel('V/µV')
axesfilt[0, 0].set_xlabel('t/ms')
axesfilt[0, 0].set_ylim(-2.5, 2.5)
axesfilt[0, 0].set_xlim(-50, 130)
axesfilt[0, 0].grid(True)

axesfilt[0, 1].plot(x, sphara_filt_eegdata[:, :].transpose())
axesfilt[0, 1].set_title('SPHARA low-pass filtered EEG data')
axesfilt[0, 1].set_ylabel('V/µV')
axesfilt[0, 1].set_xlabel('t/ms')
axesfilt[0, 1].set_ylim(-2.5, 2.5)
axesfilt[0, 1].set_xlim(-50, 130)
axesfilt[0, 1].grid(True)

for i in range(3):
    axesfilt[i + 1, 0].plot(x, eegdata_noise[i].transpose())
    axesfilt[i + 1, 0].set_title('EEG data + noise, SNR ' +
                                 str(db_val_vec[i]) + 'dB')
    axesfilt[i + 1, 0].set_ylabel('V/µV')
    axesfilt[i + 1, 0].set_xlabel('t/ms')
    axesfilt[i + 1, 0].set_ylim(-2.5, 2.5)
    axesfilt[i + 1, 0].set_xlim(-50, 130)
    axesfilt[i + 1, 0].grid(True)

    axesfilt[i + 1, 1].plot(x, eegdata_noise_filt[i].transpose())
    axesfilt[i + 1, 1].set_title('EEG data + noise, SNR ' +
                                 str(db_val_vec[i]) + 'dB, SPHARA filtered')
    axesfilt[i + 1, 1].set_ylabel('V/µV')
    axesfilt[i + 1, 1].set_xlabel('t/ms')
    axesfilt[i + 1, 1].set_ylim(-2.5, 2.5)
    axesfilt[i + 1, 1].set_xlim(-50, 130)
    axesfilt[i + 1, 1].grid(True)

plt.subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.95, hspace=0.45)
plt.show()
# sphinx_gallery_thumbnail_number = 6
