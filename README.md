# Holotorch

Holotorch is a Fourier Optics / Coherent Imaging framework developped in PyTorch and PyTorch lightning.

The main functions are found in the folders "holotorch".

Holotorch provides the following:

1. Simple setup of optical setups in simulation for forward modeling
    1. Optical propagators (e.g. ASM-Kernel, Optical fouriertransforms)
    2. Abstracted componenents like Source, Detector, SLMs, DoEs, etc.
2. Complex Wavefront objects which carry more information than just a "datatensor", such as spectral and spatial information
3. Automatic batching and saving/loading of SLM-states for more complex "Machine Learning" tasks for "Deep Optics"
4. Abstracted code for hardware often used in research holographic displays
    1. Cameras (Flir + Ximea)
    2. SLM ( when SLM is displayed as second screen. Based on python package slmpy )
5. Data aquisition pipelines to capture datasets that are e.g. useful for calibration procedures (e.g. Neural Holography)
6. PyTorch Lightning Modules tailored for Computational Holography 
    1. SLM-Lightning: Simple optimization algorithm based on gradient descent where we optimize for an SLM given an optical setup (e.g. Near-Eye or Far-Field)
    2. (Neural) Etendue Lightning: Joint optimization of a hologram and SLM-patterns

# Best Practice
see holotorch_and_visual_studio_code.md

# Example Notebook

Please navigate to SIGGRAPH_Tutorial and open "tutorial.ipynb"

The Siggraph tutorial notebook contains:

1. Example for ASM-propagator
2. Creating a Hologram using Double-Phase-Amplitude-Encoding (DPAC)
3. Near-Eye Hologram (ASM-propagation) optimization
4. Conventional Etendue Expansion with a random diffuser
5. Neural Etendue expansion (Deep Optics) with respect to an image dataset
6. Examples on how to save/load learned models

# Contact
Contact:
florianschiffers ( at ) gmail.com or florian.schiffers ( at ) u.northwestern.edu
ocossairt ( at ) fb.com
nathanmatsuda ( at ) fb.com

# LICENSE

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg