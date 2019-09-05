.. _eeg_256_channel_stud:


256 cannel EEG sensor setup and EEG dataset
-------------------------------------------

The data set consists of a triangulation of a 256 channel equidistant
EEG cap and EEG data from previously performed experiment addressing
the cortical activation related to somatosensory-evoked potentials
(SEP). During the experiment the median nerve of the right forearm was
stimulated by bipolar electrodes (stimulation rate: 3.7 Hz,
interstimulus interval: 270 ms, stimulation strength: motor plus
sensor threshold :cite:`mauguiere99,cruccu08`, constant current
rectangular pulse wave impulses with a length of 50 μs, number of
stimulations: 6000). Data were sampled at 2048 Hz and software
high-pass (24 dB/oct, cutoff-frequency 2 Hz) and notch (50 Hz and two
harmonics) filtered. All trials were manually checked for artifacts,
the remaining trials were averaged, see also S1 data set in
:cite:`graichen15`.


**Dataset characteristics:**

    :Number of vertices: 256
    :Number of triangles: 480
    :SEP Data (EEG): 128 channels, 369 time samples
    :Time range: 50 ms before to 130 ms after stimulation
    :Sampling frequency: 2048 Hz
    :Creator: Uwe Graichen
    :Date: November, 2018
