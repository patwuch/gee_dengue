# Chuang Lab — Taipei Medical University

Spatiotemporally-aware ML/DL dengue fever prediction by jointly modelling Taiwan and Southeast Asia trends and dynamics: integrating climate projections, reported infection cases, and Shared Socioeconomic Pathways (SSPs). 

Workflows managed with [Snakemake](https://snakemake.readthedocs.io/en/stable/) and environment with [Docker](https://docs.docker.com/desktop/setup/install/windows-install/). Note that if you build your own environment with "environment.yml" via Conda(https://anaconda.org/anaconda/conda), GPU compatibility issues may arise.

## Modules

**climate-projection-module** — Processes climate projection models of various SSPs and simulation models from NetCDF into merged NetCDF and easy-to-use TSV outputs.

**dengue-infection-module** — Cleans and standardises dengue case data from OpenDengue and Indonesia MoH sources.

**machine-learning-module** — Trains, evaluates, and create xAI artefacts for Random Forest, XGBoost, and ST-GAT models on remote sensing + climate projection + infection data.

## Specs

Tested on:
- OS: Ubuntu 24.04.4 LTS
- CPU: Intel Xeon W-2235 (6 cores / 12 threads @ 3.80GHz)
- RAM: 32GB
- GPU: NVIDIA GeForce GTX 1050 Ti (4GB VRAM, compute capability 6.1)
- CUDA: 12.2
- PyTorch: 2.5.1+cu121
- Docker image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

Note: GTX 1050 Ti (sm_61) is not explicitly compiled in the above image but 
falls back to sm_60 and runs correctly. Users with sm_75+ (Turing and newer) 
will get fully optimized builds.

## Notes

GPU acceleration is used only for the machine-learning-module for
(1) XGBoost tuning, training, and inference
(2) DL tuning, training, inference, and xAI