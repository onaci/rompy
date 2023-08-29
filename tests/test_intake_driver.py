import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import pytest
import yaml 
import random

import rompy
from rompy.core import BaseGrid, DataBlob, DataGrid, SourceIntake

# round now to the nearest 6 hours
cycle = datetime.utcnow().replace(
    hour=0, minute=0, second=0, microsecond=0
) - timedelta(days=2)


@pytest.fixture
def gfs():
    return DataGrid(
        id="gfs_wind",
        source = SourceIntake(
        catalog_uri=os.path.join(rompy.__path__[0], "catalogs", "gfs.yml"),
        dataset_id="gfs_glob05",
        kwargs = {"cycle": cycle},
        ),
        filter={
            "crop": {
                "time": slice(
                    cycle,
                    cycle + timedelta(days=1),
                ),  # Change dates to available dates
                "lat": slice(0, 10),
                "lon": slice(0, 10),
            },
            "subset": {"data_vars": ["ugrd10m", "vgrd10m"]},
        },
    )

@pytest.fixture
def csiro():
    return DataGrid(
        id="TODO",
        source = SourceIntake(
        catalog_uri=os.path.join(rompy.__path__[0], "catalogs", "gfs.yml"),
        dataset_id="gfs_glob05",
        kwargs = {"cycle": cycle},
        ),
        filter={
            "crop": {
                "time": slice(
                    cycle,
                    cycle + timedelta(days=1),
                ),  # Change dates to available dates
                "lat": slice(0, 10),
                "lon": slice(0, 10),
            },
            "subset": {"data_vars": ["ugrd10m", "vgrd10m"]},
        },
    )

def generate_synthetic_data_and_catalog(tmp_path_factory, failing):
    # Create sample coordinates
    lat = np.array([])
    lon = np.array([])

    # Create sample data variables
    init = pd.to_datetime('2021-03-01')
    lead = np.arange(0, 91)
    time = init + pd.to_timedelta(lead, unit='D')
    labels = [random.choice('abcdefghijklmnopqrstuvwxyz') for _ in lead]
    wnd_dir_data = np.random.rand(1, 91, len(lat), len(lon))
    wnd_spd_data = np.random.rand(1, 91, len(lat), len(lon))

    if failing:
        coord = 'time'
        coord_values = time
    else:
        coord = 'labels' 
        coord_values = labels 
         
    # Create synthetic Dataset
    dataset = xr.Dataset(
        data_vars={
            'wnd_dir': (['init', 'lead', 'lat', 'lon'], wnd_dir_data),
            'wnd_spd': (['init', 'lead', 'lat', 'lon'], wnd_spd_data)
        },
        coords={
            'lat': (['lat'], lat),
            'lon': (['lon'], lon),
            coord: (['lead'], coord_values),
            'init': (['init'], [init]),
            'lead': (['lead'], lead)
        },
        attrs={
            'Conventions': 'CF-1.5',
            'source': 'Australian Bureau of Meteorology',
            'expt_id': '0002',
            'NCO': '4.5.4',
            'nco_openmp_thread_number': 1,
            'history': 'Tue Mar 2 12:52:34 2021: /usr/bin/ncrcat /OSM...'
        }
    )

    # Save dataset as temporary file
    temp_dir = tmp_path_factory.mktemp('data')
    temp_file = temp_dir / "temp_dataset.nc"
    dataset.to_netcdf(temp_file)

    # Create temporary file Intake catalog
    temp_catalog_data = {
        'metadata': 
            {
                'version':1,
            },
        'sources': {
            'my_dataset': {
                'driver': 'netcdf',
                'args': {
                    'urlpath': temp_file.as_posix()
                },
                'metadata': {}
            }
        }
    }
    intake_temp_file = temp_dir / "temp_catalog.yaml"
    # Write the catalog to a YAML file
    with open(intake_temp_file, 'w') as f:
        yaml.dump(temp_catalog_data, f)
                
    if failing: 
        cropper =  {"time": slice(
                '2021-04-01',
                '2021-04-04',
            ) }
    else:
        cropper =  {"labels": slice(
                'd',
                'g',
            )}
        
    return DataGrid(
        id = "some_id",
        source=SourceIntake(
        catalog_uri=intake_temp_file.as_posix(),
        dataset_id="my_dataset",
        ),
        filter={
            "crop":  cropper,
            "subset": {"data_vars": ["wnd_dir", "wnd_spd"]},
        },
    )

@pytest.fixture()
def synthetic_catalog_and_data_bad(tmp_path_factory):
    return generate_synthetic_data_and_catalog(tmp_path_factory, failing=True)    

@pytest.fixture()
def synthetic_catalog_and_data_good(tmp_path_factory):
    return generate_synthetic_data_and_catalog(tmp_path_factory, failing=False)    

# mark as slow
@pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Only run when --run-slow is given",
)
def test_gfs(gfs):
    """Test that the GFS catalog works"""
    assert gfs.ds.lat.max() == 10
    assert gfs.ds.lat.min() == 0
    assert gfs.ds.lon.max() == 10
    assert gfs.ds.lon.min() == 0
    run_dir = tempfile.mkdtemp()
    downloaded_ds = xr.open_dataset(gfs.get(run_dir))
    assert downloaded_ds.lat.max() == 10
    assert downloaded_ds.lat.min() == 0
    assert downloaded_ds.lon.max() == 10
    assert downloaded_ds.lon.min() == 0
    shutil.rmtree(run_dir)


@pytest.mark.skip(reason="Not yet implemented - Ben to have a look")
def test_csiro(csiro):
    """Test that the CSIRO catalog works"""
    assert csiro.ds.lat.max() == 10
    assert csiro.ds.lat.min() == 0
    assert csiro.ds.lon.max() == 10
    assert csiro.ds.lon.min() == 0
    run_dir = tempfile.mkdtemp()
    downloaded_ds = xr.open_dataset(gfs.get(run_dir))
    # These may not be exact, may need to fine tune
    # assert downloaded.ds.lat.max() == 10
    # assert downloaded.ds.lat.min() == 0
    # assert downloaded.ds.lon.max() == 10
    # assert downloaded.ds.lon.min() == 0
    # shutil.rmtree(run_dir)

def test_non_dim_filter_time_fail(synthetic_catalog_and_data_bad):
    """Test that datetime filtering fails if you are trying it on non-dimensioned time coordinates"""
    with pytest.raises(TypeError):
        synthetic_catalog_and_data_bad.ds

def test_non_dim_filter(synthetic_catalog_and_data_good):
    """Test that datetime filtering works if you are trying it on non-dimensioned non-time coordinates"""
    assert synthetic_catalog_and_data_good.ds
