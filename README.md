# Analysis of Oxytocin Dynamics using Machine Learning

[![MATLAB Version](https://img.shields.io/badge/MATLAB-R2020a%2B-orange.svg)](https://www.mathworks.com/products/matlab.html)

Repository for the code used in the publication: **"Unraveling the dynamics of oxytocin in hypothalamic neurons"**.

**The code will be released upon final decision regarding the publication.**

## Overview

Oxytocin (OT) is a neuropeptide crucial for social behavior, and its release depends on the movement of the vesicles that contain it. This work combines live-cell imaging with computational analysis to investigate the mobility of these vesicles.

Using machine learning-based classifiers (a 1D Convolutional Neural Network and a Random Forest), we reveal that the majority of oxytocin compartments exhibit **subdiffusive motion**. This behavior suggests constraints imposed by the complex intracellular environment, such as interactions with the cytoskeleton.

This repository contains the necessary code to:
1.  Extract features from particle trajectories (Mean Squared Displacement, anomalous exponent).
2.  Generate synthetic data for different diffusion modes (normal, subdiffusive, superdiffusive).
3.  Train the CNN and Random Forest models to classify diffusion types.
4.  Apply the trained models to experimental data to predict vesicle behavior.

## Contributions

Contributions are welcome. If you find a bug or have a suggestion, please open an issue in this repository.
