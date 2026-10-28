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

From the repository root:

```bash
Rscript -e 'install.packages(
  c("RANN", "Seurat", "tidyverse", "dplyr", "assertthat", "testthat",
    "glue", "BBmisc", "Xmisc", "rstatix", "knn.covertree", "remotes", "BiocManager"),
  repos = "https://cloud.r-project.org"
)'
Rscript -e 'remotes::install_github("dynverse/dyno")'
Rscript -e 'remotes::install_github("mojaveazure/seurat-disk")'

makevars=$(mktemp)
printf '%s\n' 'CFLAGS += -std=gnu11 -Wno-error=incompatible-pointer-types' > "$makevars"
export R_MAKEVARS_USER="$makevars"
Rscript -e 'install.packages(
  c(
    "https://cran.r-project.org/src/contrib/Archive/dbplyr/dbplyr_2.3.4.tar.gz",
    "https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-5.tar.gz",
    "https://cran.r-project.org/src/contrib/Archive/MASS/MASS_7.3-60.0.1.tar.gz"
  ),
  repos = NULL,
  type = "source"
)'
Rscript -e 'BiocManager::install(
  c("DuoClustering2018", "SingleCellExperiment", "splatter"),
  ask = FALSE,
  update = FALSE
)'
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

## 6. Data

Download the dataset archives from [Zenodo record 4597151](https://zenodo.org/records/4597151) into `Scripts/python/data/`, then unzip them there. Unzipping creates the directories the notebooks read.

```bash
cd Scripts/python/data
unzip Duo_datasets.zip
unzip Freytag_Sun_datasets.zip
unzip Krzak_datasets.zip
unzip Cytotrace_datasets.zip
unzip TI_datasets.zip
```

| Archive | Directory |
|---|---|
| `Duo_datasets.zip` | `Scripts/python/data/DuoClustering2018/` |
| `Freytag_Sun_datasets.zip` | `Scripts/python/data/Freytag_datasets/` |
| `Krzak_datasets.zip` | `Scripts/python/data/benchmark_sensitivity_frontiers/` |
| `Cytotrace_datasets.zip` | `Scripts/python/data/Cytotrace_datasets/` |
| `TI_datasets.zip` | `Scripts/python/data/TI/` |

From the repository root, download the HVG10 RDS files into `Scripts/python/data/DuoClustering2018/sce_filteredHVG10/`:

```bash
Rscript -e '
outdir <- "Scripts/python/data/DuoClustering2018/sce_filteredHVG10"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
datasets <- c(
  "Koh", "KohTCC", "Kumar", "KumarTCC",
  "SimKumar4easy", "SimKumar4hard", "SimKumar8hard",
  "Trapnell", "TrapnellTCC",
  "Zhengmix4eq", "Zhengmix4uneq", "Zhengmix8eq"
)
for (name in datasets) {
  fn <- get(paste0("sce_filteredHVG10_", name), envir = asNamespace("DuoClustering2018"))
  sce <- fn()
  saveRDS(sce, file.path(outdir, paste0("sce_filteredHVG10_", name, ".rds")))
}
'
```
