# Learning Degradation Dynamics from Incomplete Trajectories and Failure Statistics
## Code repository for the Torque 2026 Conference paper

This repository contains the code to reproduce all results reported in the 2026 Torque Conference report.
It includes scripts for data loading and optimization, as well as notebooks for running experiments and generating figures.
The code is intended primarily for reproducibility and inspection rather than as a standalone library.

**Note:** You will need to download the [Care2Compare (version 6)](https://zenodo.org/records/15846963) dataset into a `dataset` directory at the root of the repository.
Alternatively, if you have already downloaded the dataset on your Linux machine, set up a symlink via `ln -s /your/data/directory dataset`.

## Contents

### Core scripts

- `load_data.py`  
  Utilities for loading and preparing the raw data.

- `optimization.py`  
  Implementation of a simple evolution strategy to fit the absorption times of the HMM.

- `compute_residuals_lin_reg.py`  
  Residual computation for linear regression model, outputs are used as the observations.

- `plotting_functions.py`  
  Shared plotting utilities.

- `util.py`  
  General helper functions.

### Notebooks

- `joint_inference.ipynb`  
  Main experiment code used for results reported in the paper.

- `preprocessing_plots.ipynb`  
  Generates preprocessing plots used in the paper.

- `inspect_indicator.ipynb`  
  Additional exploratory plots for indicator signals (not used in the paper).

### Data and outputs

- `dataset/`  
  Place the Care2Compare dataset here.

- `indicator_data/`  
  Indicator signal data will be stored here.

- `plots/`  
  Generated figures and plots go here.

## Installation

Install the dependencies via `pip install -r requirements.txt`.

## Usage

There are two preparatory steps needed before you can reproduce the results from the paper:

 - Download the [Care2Compare (version 6)](https://zenodo.org/records/15846963) dataset into `dataset`
 - Run the script `compute_residuals_lin_reg.py` to generate the residuals

Run the Jupyter notebooks to reproduce the results.
In particular, `joint_inference.ipynb` reproduces the main experimental results.

Supporting scripts are imported directly by the notebooks and are not intended
to be run as standalone command-line tools.

## Notes

- Paths and data locations are assumed to follow the current directory structure.

