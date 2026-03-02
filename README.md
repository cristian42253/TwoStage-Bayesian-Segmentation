# Two-Stage Bayesian Segmentation with AR(1) Errors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)]([[https://doi.org/10.5281/zenodo.XXXXXX)]
## Overview

This repository contains the Python implementation of the **Two-Stage Bayesian Segmentation Algorithm** developed for detecting structural changes in time series with high serial correlation.

The method addresses the problem of **Weak Identification** and **Likelihood Ridges** (Gustafson, 2010; Florens & Simoni, 2021) inherent in the simultaneous estimation of change-points and autoregressive parameters. By adopting a **Sequential Identification Strategy**, the algorithm decouples structure learning from noise estimation, ensuring asymptotic consistency and valid Bayesian learning.

## Key Features

* **Stage 1 (Restricted Experiment):** Uses a robust pilot estimator ($\hat{\varphi}_0$) derived from LOESS residuals to filter the structure topology via a **Collapsed Gibbs Sampler**.
* **Stage 2 (Conditional Inference):** Refines the autoregressive parameter ($\varphi$) and regression coefficients conditional on the identified structure, resolving the confounding between level shifts and persistence.
* **Identifiability:** Guarantees a strictly convex likelihood surface for structural parameters.

## Repository Structure

* `src/`: Core functions implementing the Sequential Identification logic.
* `supplementary_material/`: detailed algorithmic schemes and theoretical proofs regarding the identification strategy.
* `data/`: Synthetic datasets for reproducibility testing.

## Algorithmic Scheme

The estimation process follows the theoretical pipeline described in the associated thesis/paper:

1.  **Robust Initialization:** Non-parametric trend removal $\to$ Pilot $\hat{\varphi}_0$.
2.  **Structure Learning:** Metropolis-Hastings with analytic integration of nuisance parameters.
3.  **Parameter Recovery:** Conditional GLS estimation on stationary residuals.

> **Full Diagram:** A high-resolution flowchart of the coupling between modules is available in [`supplementary_material/Algorithm_Scheme_Flowchart.pdf`](./supplementary_material/Algorithm_Scheme_Flowchart.pdf).

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/tu_usuario/TwoStage-Bayesian-Segmentation.git](https://github.com/tu_usuario/TwoStage-Bayesian-Segmentation.git)
