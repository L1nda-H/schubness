# Export DuoClustering2018 sce_filteredHVG10 RDS files to MatrixMarket + labels
# without needing SingleCellExperiment/DuoClustering2018 installed.
#
# The RDS files can be fetched directly from ExperimentHub:
#   EH1502 Koh, EH1506 KohTCC, EH1510 Kumar, EH1514 KumarTCC, EH1518 Trapnell,
#   EH1522 TrapnellTCC, EH1526 Zhengmix4eq, EH1530 Zhengmix4uneq,
#   EH1534 Zhengmix8eq, EH1538 SimKumar4easy, EH1542 SimKumar4hard,
#   EH1546 SimKumar8hard
#   curl -L -o EH1510.rds https://experimenthub.bioconductor.org/fetch/1510
#
# Usage: Rscript export_duo_hvg10.R <rds_dir> <out_dir>
# Then:  python Scripts/python/Clustering/mtx_to_h5ad.py <out_dir> data/

ids <- c(
  Koh = 1502, KohTCC = 1506, Kumar = 1510, KumarTCC = 1514,
  Trapnell = 1518, TrapnellTCC = 1522, Zhengmix4eq = 1526,
  Zhengmix4uneq = 1530, Zhengmix8eq = 1534, SimKumar4easy = 1538,
  SimKumar4hard = 1542, SimKumar8hard = 1546
)
args <- commandArgs(trailingOnly = TRUE)
rds_dir <- args[1]
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(Matrix))

for (name in names(ids)) {
  candidates <- file.path(rds_dir, c(
    paste0("EH", ids[[name]], ".rds"),
    paste0("sce_filteredHVG10_", name, ".rds")
  ))
  path <- candidates[file.exists(candidates)][1]
  if (is.na(path)) {
    message("missing ", name)
    next
  }
  # Without the SingleCellExperiment package the object loads as an
  # unregistered S4 instance; its slots are still reachable as attributes.
  sce <- suppressWarnings(readRDS(path))
  assays <- get("data", attr(attr(sce, "assays"), ".xData"))
  counts <- attr(assays, "listData")$counts
  col_data <- attr(attr(sce, "colData"), "listData")
  stopifnot(length(col_data$phenoid) == ncol(counts))
  writeMM(Matrix(counts, sparse = TRUE), file.path(out_dir, paste0(name, ".mtx")))
  write.csv(
    data.frame(cell = colnames(counts), phenoid = as.character(col_data$phenoid)),
    file.path(out_dir, paste0(name, "_obs.csv")),
    row.names = FALSE
  )
  writeLines(rownames(counts), file.path(out_dir, paste0(name, "_genes.txt")))
  message(name, ": ", nrow(counts), " genes x ", ncol(counts), " cells")
}
