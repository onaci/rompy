import intake
import pandas as pd

import rompy
import rompy.filters as filt
from rompy.swan import Swan_accessor


cat = intake.open_catalog("/source/csiro/rompy/rompy/catalogs/bathy.yaml")

# ds_filters = {
#     filt.rename_filter: {"elevation": "depth"},
# }

# # dset = cat["gebco"].to_dask()
# dset = cat.gebco(
#     urlpath="/static/glob/gebco20/GEBCO_2020_4km.nc",
#     ds_filters=ds_filters,
# ).to_dask()

# dset.swan.to_inpgrid(
#     output_file="test.bot",
#     grid=
# )
# import ipdb; ipdb.set_trace()


dset = cat.gebco_sel().to_dask()