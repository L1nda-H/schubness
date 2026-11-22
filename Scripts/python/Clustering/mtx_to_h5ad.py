"""Convert export_duo_hvg10.R output into the H5AD layout used by
run_h5ad_gamma_experiment.py (cells x genes, layers['counts'], obs['phenoid']).

Usage: python mtx_to_h5ad.py <export_dir> <h5ad_dir>
"""

import sys
from pathlib import Path

import anndata
import pandas as pd
import scipy.io
import scipy.sparse as sp


def main(export_dir, h5ad_dir):
    export_dir, h5ad_dir = Path(export_dir), Path(h5ad_dir)
    h5ad_dir.mkdir(parents=True, exist_ok=True)
    for mtx in sorted(export_dir.glob("*.mtx")):
        name = mtx.stem
        counts = sp.csr_matrix(scipy.io.mmread(str(mtx)).T, dtype="float32")
        obs = pd.read_csv(export_dir / (name + "_obs.csv")).set_index("cell")
        obs.index = obs.index.astype(str)
        genes = (export_dir / (name + "_genes.txt")).read_text().split()
        adata = anndata.AnnData(
            X=counts.copy(),
            obs=obs,
            var=pd.DataFrame(index=genes),
            # Dense: run_h5ad_gamma_experiment.prepare_pca calls np.asarray
            # on the backed layer. HVG10 matrices are small.
            layers={"counts": counts.toarray()},
        )
        out = h5ad_dir / "sce_filteredHVG10_{}_gpu.h5ad".format(name)
        adata.write_h5ad(out)
        print(name, adata.shape, dict(adata.obs["phenoid"].value_counts()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
