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

dset = cat.gebco().to_dask()

from rompy.filters import crop_filter

ds = crop_filter(dset, lat=slice(0, 10), lon=slice(0, 10))

import ipdb; ipdb.set_trace()
