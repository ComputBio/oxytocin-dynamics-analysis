# Unraveling the dynamics of oxytocin in hypothalamic neurons

[![MATLAB Version](https://img.shields.io/badge/MATLAB-R2021a%2B-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository provides a machine learning framework to classify the diffusion modes of neuropeptide-containing vesicles (LDCVs) in neurons. By combining a **1D Convolutional Neural Network (CNN)** and a **Random Forest (RF)** classifier, the pipeline identifies Normal, Subdiffusive, and Superdiffusive motion from single-particle tracking (SPT) data.

Repository for the code used in the publication: 
> Aznar-Escolano, B., Egorova, V., Villanueva, J., Gutiérrez, L. M., González-Vélez, V., Gil, A., & Jurado, S. (2025). Unraveling the dynamics of oxytocin in hypothalamic neurons. *Traffic*. DOI: 10.1111/tra.70034

## Overview

Oxytocin (OT) is a neuropeptide crucial for social behavior, and its release depends on the movement of the vesicles that contain it. This work combines live-cell imaging with computational analysis to investigate the mobility of these vesicles.

Using machine learning-based classifiers (a 1D Convolutional Neural Network and a Random Forest), we reveal that the majority of oxytocin compartments exhibit **subdiffusive motion**. This behavior suggests constraints imposed by the complex intracellular environment, such as interactions with the cytoskeleton.

This repository contains the necessary code to:
1.  Extract features from particle trajectories (Mean Squared Displacement, anomalous exponent).
2.  Generate synthetic data for different diffusion modes (normal, subdiffusive, superdiffusive).
3.  Train the CNN and Random Forest models to classify diffusion types.
4.  Apply the trained models to experimental data to predict vesicle behavior.

---

## Getting Started

### Prerequisites
- **MATLAB R2021a** or later.
- **Deep Learning Toolbox** (for the CNN).
- **Statistics and Machine Learning Toolbox** (for the Random Forest).

### Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ComputBio/oxytocin-dynamics-analysis.git
   cd oxytocin-dynamics-analysis
   ```
2. **Note on Data:** The experimental dataset used in the paper is not included in this repository. To use the pipeline, you must first create a `data/` folder and either generate synthetic data or provide your own trajectories.
   ```matlab
   mkdir data
   mkdir results
   mkdir models
   ```
To use the classification scripts on your own data, ensure your trajectories match these constraints:
- **Trajectory Length:** The models are optimized for 80 frames. The script will automatically pad or truncate data to this length.
- **Sampling Rate:** The default is 1 Hz ($dt = 1s$). Adjust `params.timeStep` in `main.m` if your acquisition rate differs.

---

### 1. Generate Training Data
Since the original training data is not provided, you must generate a synthetic dataset using fractional Brownian Motion (fBM) simulations.
*   Run `main.m`. This will automatically call `step_synthetic.m` to generate 100,000 trajectories based on the experimental parameters (80 steps, 1Hz) described in the paper.

### 2. Train the Models
The pipeline will then train both the **Random Forest** and the **1D-CNN**.
*   The CNN learns from the raw shape of the Min-Max normalized MSD curves.
*   The Random Forest learns from extracted features: $\alpha$, $\log_{10}(K_\alpha)$, and $R^2$.

### 3. Analyze Your Own Data
To classify your own experimental trajectories, format your data as a MATLAB table `T` with the following columns:
- `X`: Cell array of x-coordinates.
- `Y`: Cell array of y-coordinates.
- `TAMSD`: Cell array of calculated Time-Averaged Mean Squared Displacements.

Run `data_preparation.m` to convert your raw tracking files into this format.

---

##  Project Structure

```text
├── main.m                 # Entry point: runs the full simulation & training
├── data_preparation.m     # Script to format user data for analysis
├── step_synthetic.m       # Generates 100k fBM trajectories for training
├── step_CNN.m             # Defines and trains the 1D-CNN
├── step_classification.m  # Applies the trained CNN to experimental data
│   ├── MSD.m              # Calculates TAMSD, alpha, and K_alpha
│   └── generate_synthetic_trajectories.m  # fBM simulation engine
├── models/                # [Folder for saved .mat model files]
└── results/               # [Folder for generated plots and tables]
```

---





## Contributions

Contributions are welcome. If you find a bug or have a suggestion, please open an issue in this repository.

**Contact:** 
*   **S. Jurado** (sjurado@umh.es)
*   **A. Gil** (amparo.gil@unican.es)




