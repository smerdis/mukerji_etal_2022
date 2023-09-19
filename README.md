# mukerji_etal_2022
Code and data for Mukerji et al 2022 paper in Frontiers in Human Neuroscience.

This repository contains a runnable notebook that will regenerate the figures in the paper from the included data files.

To ensure reproducibility, this notebook can be run in containerized form (requires Docker Desktop) by using the following command from within this repository:
```
docker run -it --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work jupyter/scipy-notebook:2021-09-20
```
The JupyterLab interface is then available at http://127.0.0.1:10000/notebooks/work/Mukerji%20et%20al%2C%202022.ipynb. Running all cells will re-generate the plots. This may take some time due to the bootstrapping procedure.

Paper citation: Mukerji A, Byrne KN, Yang E, Levi DM, and Silver MA (2022). Visual cortical γ−aminobutyric acid and perceptual suppression in amblyopia. *Front. Hum. Neurosci.* 16:949395. doi: https://doi.org/10.3389/fnhum.2022.949395
