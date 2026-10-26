# Setup

## 1. Create the conda environment

```bash
mamba create -n schubness-cs6850 -y -c conda-forge \
  python=3.7.12 \
  numpy=1.19.5 \
  scipy=1.7.3 \
  pandas=1.3.5 \
  scikit-learn=0.21.3 \
  matplotlib=3.5.3 \
  seaborn \
  networkx=2.7 \
  tqdm \
  natsort \
  h5py \
  python-igraph=0.10.1 \
  leidenalg=0.9.0 \
  louvain=0.8.0 \
  ipykernel \
  joblib \
  numba=0.53.1 \
  llvmlite=0.36 \
  r-base=4.1.3 \
  "rpy2<3.5"
```

```bash
conda activate schubness-cs6850
```

## 2. Install Python requirements

From the repository root:

```bash
pip install -r requirements.txt
pip install --no-deps umap-learn==0.5.1 pynndescent==0.5.8 scikit-hubness==0.21.2
```

## 3. Install R packages

```bash
Rscript -e 'install.packages(
  c("RANN", "Seurat", "tidyverse", "dplyr", "assertthat", "testthat",
    "glue", "BBmisc", "Xmisc", "rstatix", "knn.covertree", "remotes"),
  repos = "https://cloud.r-project.org"
)'
Rscript -e 'remotes::install_github("dynverse/dyno")'
Rscript -e 'remotes::install_github("mojaveazure/seurat-disk")'
```

## 4. Install scEntropy

```bash
pip install --no-deps "git+https://github.com/jzlei/scEntropy.git"
```

## 5. Point the editor at the environment

```bash
conda run -n schubness-cs6850 python -c 'import sys; print(sys.executable)'
```

Set `python.defaultInterpreterPath` in `.vscode/settings.json` to that path, then reload the window.

```bash
python -m ipykernel install --user --name schubness-cs6850 --display-name "Python (schubness-cs6850)"
```
