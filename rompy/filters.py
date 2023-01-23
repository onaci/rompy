#-----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO 
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------


def derived_filter(ds, derived_variables):
    """Add derived variable to Dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset to add derived variables to.
    derived_variables: dict
        Mapping {`derived_variable_name`: `derived_variable_definition`} where
        `derived_variable_definition` is a string to be evaluated defining some
        transformation based on existing variables in the input dataset `ds`.

    Returns
    -------
    ds: xarray.Dataset
        Input dataset with extra derived variables.

    Example
    -------
    >>> import xarray as xr
    >>> ds = xr.DataArray([-10, -11], coords={"x": [0, 1]}).to_dataset(name="elevation")
    >>> ds = derived_filter(ds, {"depth": "ds.elevation * -1"})

    """
    for var, expr in derived_variables.items():
        ds[var] = eval(expr)
    return ds


def sort_filter(ds,coords=None):
    for c in coords:
        if c in ds:
            ds = ds.sortby(c)
    return ds

def subset_filter(ds,data_vars=None):
    if data_vars is not None:
        ds = ds[data_vars]
    return ds

def crop_filter(ds,**data_slice):
    if data_slice is not None:
        this_crop = {k:data_slice[k] for k in data_slice.keys() if k in ds.dims.keys()}
        ds = ds.sel(this_crop)
        for k in data_slice.keys():
            if (k not in ds.dims.keys()) and (k in ds.coords.keys()):
                ds=ds.where(ds[k]>float(data_slice[k][0]),drop=True)
                ds=ds.where(ds[k]<float(data_slice[k][1]),drop=True)
    return ds

def timenorm_filter(ds,interval='hour',reftime=None):
    from pandas import to_timedelta, to_datetime
    dt = to_timedelta('1 ' + interval)
    if reftime is None:
        ds['init'] = ds['time'][0]
    else:
        ds['init'] = (('time',),to_datetime(ds[reftime].values))
    ds['lead'] = ((ds['time']-ds['init'])/dt).astype('int')
    ds['lead'].attrs['units'] = interval
    ds = ds.set_coords('init')
    ds = ds.swap_dims({'time':'lead'})
    return ds

def rename_filter(ds,**varmap):
    ds = ds.rename(varmap)
    return ds

def get_filter_fns():
    return {'sort': sort_filter,
            'subset': subset_filter,
            'crop': crop_filter,
            'timenorm': timenorm_filter,
            'rename': rename_filter}

def _open_preprocess(url,chunks,filters,xarray_kwargs):
    import xarray as xr
    ds = xr.open_dataset(url,chunks=chunks,**xarray_kwargs)
    filter_fns = get_filter_fns()
    for fn, params in filters.items():
        if isinstance(fn,str):
            fn = filter_fns[fn]
        ds = fn(ds,**params)
    return ds
