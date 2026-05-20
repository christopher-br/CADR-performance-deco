# A Data-Centric Decomposition of Estimator Performance in Continuous Treatment Effect Estimation


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


This repository provides code for our manuscript "A Data-Centric Decomposition of Estimator Performance in Continuous Treatment Effect Estimation".  

In our manuscript, we evaluate the impacts of different data-generating processes on data-driven methodologies for conditional average dose response (CADR) estimation. We provide source code to reproduce our experiments, including data generators, performance evaluators, and learning methods.


## Repository structure

This repository is structured as follows:

```bash
src-of-gain/
    |- src/               # Core library
        |- data/            # Data generators
        |- methods/         # Treatment effect estimators
        |- utils/           # Performance evaluation and other utils
    |- scripts/           # Executables
        |- exp/             # Reproduce experiments
        |- figures/         # Reproduce figures
        |- tables/          # Reproduce tables
    |- data/              # Data files
    |- config/            # Paramters for data loading and hyperparameter tuning
```

For reproducing experiments on TCGA datasets, download the necessary covariate matrices from [here](https://drive.google.com/file/d/1VNEZn_aeNzxPfMB9uf2P4ofYJdF9U90N/view?usp=sharing) and save the ```data/``` folder to the repository.


## Installation

All code provided was written for ```python 3.9.16```. To execute the code, please install the necessary packages to a newly created virtual environment by running:

```bash
pip install -r requirements.txt
pip install .
```

## Running experiments

All executables are in the ```scripts/``` folder. To execute them, simply run:

```bash
python scripts/[folder]/[script]
```

All results (performance metrics and plots) are saved to dedicated folders in the repository during execution.

---
 
# Supplementary Material
 
Due to the strict page limit of the manuscript, additional results and implementation details are documented below. The notation `random.` denotes the randomized scenario, `non-unif.` denotes non-uniformity, and `conf.` denotes confounding. **Bold** indicates the best result, *italics* the second-best.
 
## Results per dataset
 
In addition to the detailed discussion of model performance on the TCGA-2 dataset presented in the main manuscript (Section *Case Study*), we provide results for every other available benchmarking dataset below.
 
### IHDP-1 dataset
 
Performance decomposition (MISE) on the IHDP-1 dataset:
 
| Method        | random.            | *d* non-unif.       | *d* conf.           |
|---------------|--------------------|---------------------|---------------------|
| Lin. reg.     | 4.66 ± 0.03        | 4.76 ± 0.06         | 4.85 ± 0.07         |
| Reg. tree     | *1.33* ± 0.11      | 1.35 ± 0.21         | *1.24* ± 0.10       |
| GAM           | 1.67 ± 0.03        | 1.71 ± 0.03         | 1.95 ± 0.13         |
| Kernel ridge  | 1.81 ± 0.03        | 1.73 ± 0.02         | 1.77 ± 0.01         |
| xgboost       | **1.04** ± 0.10    | **1.08** ± 0.10     | **1.14** ± 0.13     |
| MLP           | 2.88 ± 0.26        | 3.10 ± 0.24         | 2.85 ± 0.24         |
| SCIGAN        | 6.86 ± 1.21        | 6.54 ± 1.07         | 5.88 ± 0.47         |
| DRNet         | 2.63 ± 0.15        | 2.39 ± 0.06         | 2.60 ± 0.11         |
| VCNet         | 1.38 ± 0.20        | *1.18* ± 0.20       | 1.43 ± 0.21         |
 
### News-3 dataset
 
Performance decomposition (MISE) on the News-3 dataset:
 
| Method        | random.            | *d* non-unif.       | *d* conf.           |
|---------------|--------------------|---------------------|---------------------|
| Lin. reg.     | 1.07 ± 0.10        | 1.09 ± 0.11         | 1.08 ± 0.10         |
| Reg. tree     | 1.26 ± 0.13        | 1.30 ± 0.10         | 1.29 ± 0.18         |
| GAM           | 1.11 ± 0.08        | 1.16 ± 0.08         | 1.12 ± 0.05         |
| Kernel ridge  | *0.93* ± 0.06      | *0.91* ± 0.06       | *0.91* ± 0.06       |
| xgboost       | 0.98 ± 0.06        | 0.98 ± 0.04         | *0.97* ± 0.05       |
| MLP           | 1.04 ± 0.08        | 1.03 ± 0.12         | 1.01 ± 0.11         |
| SCIGAN        | 1.57 ± 0.15        | 1.89 ± 0.22         | 2.32 ± 1.71         |
| DRNet         | 1.01 ± 0.10        | 0.99 ± 0.09         | 1.00 ± 0.08         |
| VCNet         | **0.91** ± 0.05    | **0.90** ± 0.04     | **0.78** ± 0.06     |
 
### Synth-1 dataset
 
Performance decomposition (MISE) on the Synth-1 dataset:
 
| Method        | random.            | *d* non-unif.       | *d* conf.           |
|---------------|--------------------|---------------------|---------------------|
| Lin. reg.     | 0.73 ± 0.03        | 0.73 ± 0.03         | 0.77 ± 0.03         |
| Reg. tree     | 0.50 ± 0.05        | 0.53 ± 0.12         | 0.57 ± 0.11         |
| GAM           | 0.44 ± 0.03        | 0.44 ± 0.03         | 0.48 ± 0.04         |
| Kernel ridge  | *0.32* ± 0.03      | *0.33* ± 0.04       | *0.37* ± 0.03       |
| xgboost       | 0.41 ± 0.03        | 0.41 ± 0.02         | 0.49 ± 0.04         |
| MLP           | *0.32* ± 0.02      | *0.32* ± 0.03       | *0.42* ± 0.05       |
| SCIGAN        | 0.58 ± 0.11        | 0.62 ± 0.09         | 1.09 ± 0.13         |
| DRNet         | 0.49 ± 0.03        | 0.49 ± 0.03         | 0.50 ± 0.03         |
| VCNet         | **0.31** ± 0.03    | **0.31** ± 0.03     | **0.37** ± 0.04     |
 
## Results for policy error
 
### Policy error on TCGA-2 dataset
 
| Method        | random.                | *t* non-unif.          | *t* conf.              | *d* non-unif.            | *d* conf.                |
|---------------|------------------------|------------------------|------------------------|--------------------------|--------------------------|
| Lin. reg.     | 514.47 ± 7.65          | 514.01 ± 8.27          | 514.01 ± 8.27          | 514.01 ± 8.27            | 514.01 ± 8.27            |
| Reg. tree     | 0.74 ± 0.27            | 0.56 ± 0.16            | 0.60 ± 0.13            | 0.46 ± 0.14              | 0.46 ± 0.09              |
| GAM           | 0.23 ± 0.12            | 0.01 ± 0.00            | 0.04 ± 0.06            | 265.32 ± 290.72          | 265.18 ± 290.57          |
| Kernel ridge  | **0.00** ± 0.00        | **0.00** ± 0.00        | **0.00** ± 0.00        | **0.00** ± 0.00          | **0.00** ± 0.00          |
| xgboost       | 0.21 ± 0.05            | 0.21 ± 0.26            | 0.15 ± 0.08            | 0.04 ± 0.02              | 0.05 ± 0.03              |
| MLP           | 0.20 ± 0.16            | 0.06 ± 0.07            | 0.14 ± 0.08            | 413.84 ± 205.22          | 411.41 ± 205.04          |
| SCIGAN        | 0.70 ± 0.73            | 0.02 ± 0.02            | 0.52 ± 1.03            | 30.22 ± 73.39            | 0.00 ± 0.00              |
| DRNet         | 0.29 ± 0.38            | 0.52 ± 0.41            | 0.42 ± 0.43            | 0.26 ± 0.40              | 0.26 ± 0.40              |
| VCNet         | **0.00** ± 0.00        | **0.00** ± 0.00        | **0.00** ± 0.00        | **0.00** ± 0.00          | **0.00** ± 0.00          |
 
### Dose policy error on IHDP-3 dataset
 
| Method        | Base                   | *d* non-unif.          | *d* conf.              |
|---------------|------------------------|------------------------|------------------------|
| Lin. reg.     | 401.45 ± 0.00          | 401.45 ± 0.00          | 401.45 ± 0.00          |
| Reg. tree     | 99.79 ± 53.19          | 197.42 ± 76.05         | 362.05 ± 17.93         |
| GAM           | 447.43 ± 318.78        | 623.00 ± 447.26        | 373.21 ± 30.14         |
| Kernel ridge  | 138.15 ± 61.42         | 187.38 ± 99.14         | 320.90 ± 1.35          |
| xgboost       | 51.45 ± 24.91          | 100.86 ± 46.60         | 306.24 ± 11.75         |
| MLP           | 16.54 ± 3.42           | *18.42* ± 5.79         | 304.08 ± 3.34          |
| SCIGAN        | 162.53 ± 148.43        | 172.43 ± 140.13        | 297.81 ± 66.66         |
| DRNet         | *15.80* ± 10.79        | 76.62 ± 123.90         | **254.17** ± 31.15     |
| VCNet         | **4.75** ± 4.16        | **11.23** ± 6.67       | *264.68* ± 30.55       |
 
## Implementation and hyperparameter optimization
 
All experiments were written in Python 3.9 and run on an Apple M2 Pro SoC with 10 CPU cores, 16 GPU cores, and 16 GB of shared memory. The system needs approximately two days for the iterative execution of all experiments.
 
For SCIGAN and VCNet, we use the original implementations provided by Bica et al. (2020) ([SCIGAN repository](https://github.com/ioanabica/SCIGAN)) and Nie et al. (2021) ([VCNet repository](https://github.com/lushleaf/varying-coefficient-net-with-functional-tr)). All remaining neural network architectures were implemented in PyTorch using Lightning. Xgboost is implemented using the `xgboost` library. GAMs were implemented using the `PyGAM` library. For Kernel Ridge Regression, following Singh et al. (2024) and the implementation of Raykov et al. (2025), the kernel factorizes as *K* = *K<sub>X</sub>* ⊙ *K<sub>T</sub>* ⊙ *K<sub>D</sub>*, with Radial Basis Function (RBF) kernels on covariates and dose, a delta kernel on the intervention, and bandwidths set by scaling the median pairwise distance of training inputs by *σ<sub>x</sub>* and *σ<sub>d</sub>*. All other methods were implemented using the `Scikit-Learn` library and the `statsmodels` library.
 
For TCGA-based datasets, linear regression models and GAMs were trained using the first 50 principal components of the covariate matrix to reduce computational complexity.
 
### Hyperparameter optimization
 
For all methods, we used a validation set for hyperparameter optimization and chose the best model in terms of validation set mean squared errors (MSE). We do so to ensure fair model comparison and isolate model performance from parameter selection procedures, as presented accompanying some estimators. We ran a random search over the hyperparameter ranges listed below per model. If not specified differently, the remaining hyperparameters are set to match the specifications of the original authors. Results are not to be compared to the original papers, as the optimization scheme and parameter search ranges differ from the original records.
 
#### Linear Regression
 
| Parameter | Values |
|-----------|--------|
| Penalty   | {Elastic net, None} |
 
#### Regression Tree
 
| Parameter              | Values |
|------------------------|--------|
| Max depth              | {5, 15, None} |
| Min sample split       | {2, 5, 20} |
| Min samples per leaf   | {1, 5, 10} |
| Max features per split | {None, √*p*(**x**)} |
| Splitting criterion    | {Gini} *(fixed)* |
 
#### GAM
 
| Parameter           | Values |
|---------------------|--------|
| Interaction type    | {Univariate} |
| Numb configurations | {20} |
 
#### Kernel Ridge
 
| Parameter   | Values |
|-------------|--------|
| *λ*         | {0.001, 0.01, 0.1, 1.0} |
| *σ<sub>x</sub>* | {0.5, 1.0, 2.0} |
| *σ<sub>d</sub>* | {0.5, 1.0, 2.0} |
 
#### xgboost
 
| Parameter                | Values |
|--------------------------|--------|
| Learning rate            | {0.01, 0.1, 0.2} |
| Max depth                | {3, 5, 7, 9} |
| Subsample                | {0.5, 0.7, 1.0} |
| Min child weight         | {1, 3, 5} |
| Gamma                    | {0.0, 0.1, 0.2} |
| Columns sampled per tree | {0.3, 0.5, 0.7} |
 
#### MLP
 
| Parameter         | Values |
|-------------------|--------|
| Learning rate     | {0.0001, 0.001} |
| L2 regularization | {0.0, 0.1} |
| Batch size        | {64, 128} |
| Hidden size       | {32, 48} |
| Num steps         | {5000} *(fixed)* |
| Num layers        | {2} *(fixed)* |
| Optimizer         | {Adam} *(fixed)* |
 
#### SCIGAN
 
| Parameter         | Values |
|-------------------|--------|
| Hidden size       | {32, 64, 128} |
| Batch size        | {128, 256} |
| Num head layers   | {2} *(fixed)* |
| Num dose samples  | {5} *(fixed)* |
| *λ*               | {1} *(fixed)* |
| Optimizer         | {Adam} *(fixed)* |
 
#### DRNet
 
| Parameter         | Values |
|-------------------|--------|
| Learning rate     | {0.0001, 0.001} |
| L2 regularization | {0.0, 0.1} |
| Batch size        | {64, 128} |
| Hidden size       | {32, 48} |
| Num dose strata   | {10} *(fixed)* |
| Num steps         | {5000} *(fixed)* |
| Num layers        | {2} *(fixed)* |
| Optimizer         | {Adam} *(fixed)* |
 
#### VCNet
 
| Parameter     | Values |
|---------------|--------|
| Learning rate | {0.001, 0.01} |
| Batch size    | {128, 256} |
| Hidden size   | {32} *(fixed)* |
| Num steps     | {5000} *(fixed)* |
| Optimizer     | {Adam} *(fixed)* |
 
