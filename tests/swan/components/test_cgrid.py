"""Test cgrid component."""
import pytest

from rompy.swan.components.cgrid import (
    CGRID,
    REGULAR,
    CURVILINEAR,
    UNSTRUCTURED,
)


@pytest.fixture(scope="module")
def curvilinear_kwargs():
    yield dict(mdc=36, flow=0.04, fhigh=0.4, mxc=10, myc=10, fname="grid_coord.txt")


def test_cgrid():
    CGRID(mdc=36, flow=0.04, fhigh=0.4, msc=28, dir1=None, dir2=None)


def test_cgrid_msc_gt_3():
    with pytest.raises(ValueError):
        CGRID(mdc=36, msc=2)


def test_cgrid_circle():
    cgrid = CGRID(mdc=36, flow=0.04, fhigh=0.4)
    assert cgrid.dir_sector == "CIRCLE"


def test_cgrid_sector():
    cgrid = CGRID(mdc=36, flow=0.04, fhigh=0.4, dir1=0.0, dir2=180.0)
    assert cgrid.dir_sector == "SECTOR 0.0 180.0"


def test_cgrid_dir1_and_dir2():
    with pytest.raises(ValueError):
        CGRID(mdc=36, flow=0.04, fhigh=0.4, dir1=45.0)


def test_cgrid_freq_args_at_least_two():
    with pytest.raises(ValueError):
        CGRID(mdc=36, flow=0.04)


def test_cgrid_flow_less_than_fhigh():
    with pytest.raises(ValueError):
        CGRID(mdc=36, flow=0.4, fhigh=0.04)


def test_regular_cgrid():
    cgrid = REGULAR(
        mdc=36,
        flow=0.04,
        fhigh=0.4,
        xpc=0.0,
        ypc=0.0,
        alpc=0.0,
        xlenc=100.0,
        ylenc=100.0,
        mxc=10,
        myc=10,
    )


def test_curvilinear_cgrid(curvilinear_kwargs):
    CURVILINEAR(**curvilinear_kwargs)


def test_curvilinear_cgrid_exception(curvilinear_kwargs):
    CURVILINEAR(xexc=-999.0, yexc=-999.0, **curvilinear_kwargs)
    with pytest.raises(ValueError):
        CURVILINEAR(xexc=-999.0, **curvilinear_kwargs)
    with pytest.raises(ValueError):
        CURVILINEAR(yexc=-999.0, **curvilinear_kwargs)


def test_read_grid_coord_free_or_fixed_or_unformatted_only(curvilinear_kwargs):
    CURVILINEAR(format="fixed", form="(10X,12F5.0)", **curvilinear_kwargs)
    CURVILINEAR(format="free", **curvilinear_kwargs)
    CURVILINEAR(format="unformatted", **curvilinear_kwargs)
    with pytest.raises(ValueError):
        CURVILINEAR(format="something_else", **curvilinear_kwargs)


def test_read_grid_coord_idfm_options(curvilinear_kwargs):
    CURVILINEAR(format="fixed", idfm=1, **curvilinear_kwargs)
    CURVILINEAR(format="fixed", idfm=5, **curvilinear_kwargs)
    CURVILINEAR(format="fixed", idfm=6, **curvilinear_kwargs)
    CURVILINEAR(format="fixed", idfm=8, **curvilinear_kwargs)
    with pytest.raises(ValueError):
        CURVILINEAR(format="fixed", idfm=9, **curvilinear_kwargs)


def test_read_grid_fixed_format_arguments(curvilinear_kwargs):
    CURVILINEAR(format="fixed", form="(10X,12F5.0)", **curvilinear_kwargs)
    CURVILINEAR(format="fixed", idfm=1, **curvilinear_kwargs)
    with pytest.raises(ValueError):
        CURVILINEAR(format="fixed", idfm=5, form="(10X,12F5.0)", **curvilinear_kwargs)
    with pytest.raises(ValueError):
        CURVILINEAR(format="fixed", **curvilinear_kwargs)


def test_unstructured_cgrid_adcirc():
    UNSTRUCTURED(mdc=36, flow=0.04, fhigh=0.4)


def test_unstructured_cgrid_triangle_easymesh():
    UNSTRUCTURED(mdc=36, flow=0.04, fhigh=0.4, grid_type="triangle", fname="mesh.txt")
    UNSTRUCTURED(mdc=36, flow=0.04, fhigh=0.4, grid_type="easymesh", fname="mesh.txt")


def test_unstructured_cgrid_grid_types():
    with pytest.raises(ValueError):
        UNSTRUCTURED(mdc=36, flow=0.04, fhigh=0.4, grid_type="something_else")