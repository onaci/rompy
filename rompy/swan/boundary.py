"""SWAN boundary classes."""
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import wavespectra
from pydantic import Field, model_validator

from rompy.core.filters import Filter
from rompy.core.data import (
    DataGrid,
    SourceBase,
    SourceDataset,
    SourceFile,
    SourceIntake,
    SourceDatamesh,
)
from rompy.swan.grid import SwanGrid


logger = logging.getLogger(__name__)


def find_minimum_distance(points: list[tuple[float, float]]) -> float:
    """Find the minimum distance between a set of points.

    Parameters
    ----------
    points: list[tuple[float, float]]
        List of points as (x, y) tuples.

    Returns
    -------
    min_distance: float
        Minimum distance between all points.

    """

    def calculate_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    n = len(points)
    if n <= 1:
        return float("inf")

    # Sort points by x-coordinate
    points.sort()

    # Recursive step
    if n == 2:
        return calculate_distance(*points[0], *points[1])

    mid = n // 2
    left_points = points[:mid]
    right_points = points[mid:]

    # Divide and conquer
    left_min = find_minimum_distance(left_points)
    right_min = find_minimum_distance(right_points)

    min_distance = min(left_min, right_min)

    # Find the closest pair across the dividing line
    strip = []
    for point in points:
        if abs(point[0] - points[mid][0]) < min_distance:
            strip.append(point)

    strip_min = min_distance
    strip_len = len(strip)
    for i in range(strip_len - 1):
        j = i + 1
        while j < strip_len and (strip[j][1] - strip[i][1]) < strip_min:
            distance = calculate_distance(*strip[i], *strip[j])
            if distance < strip_min:
                strip_min = distance
            j += 1

    return min(min_distance, strip_min)


class SourceWavespectra(SourceBase):
    """Wavespectra dataset from wavespectra reader."""

    model_type: Literal["wavespectra"] = Field(
        default="wavespectra",
        description="Model type discriminator",
    )
    uri: str | Path = Field(description="Path to the dataset")
    reader: str = Field(
        description="Name of the wavespectra reader to use, e.g., read_swan",
    )
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to wavespectra.read_swan",
    )

    def __str__(self) -> str:
        return f"SourceWavespectra(uri={self.uri}, reader={self.reader})"

    def _open(self):
        return getattr(wavespectra, self.reader)(self.uri, **self.kwargs)


BOUNDARY_SOURCE_TYPES = Union[
    SourceDataset,
    SourceFile,
    SourceIntake,
    SourceDatamesh,
    SourceWavespectra,
]


class DataBoundary(DataGrid):
    """SWAN BOUNDNEST1 NEST data class.

    Notes
    -----
    The `tolerance` behaves differently with sel_methods `idw` and `nearest`; in `idw`
    sites with no enough neighbours within `tolerance` are masked whereas in `nearest`
    an exception is raised (see wavespectra documentation for more details).

    Be aware that when using `idw` missing values will be returned for sites with less
    than 2 neighbours within `tolerance` in the original dataset. This is okay for land
    mask areas but could cause boundary issues when on an open boundary location. To
    avoid this either use `nearest` or increase `tolerance` to include more neighbours.

    TODO: Break this into a core and a swan specific class.
    TODO: Allow specifying resolutions along x and y instead of a single value.

    """

    id: str = Field(description="Unique identifier for this data source")
    source: BOUNDARY_SOURCE_TYPES = Field(
        description=(
            "Dataset source reader, must return a wavespectra-enabled "
            "xarray dataset in the open method"
        ),
        discriminator="model_type",
    )
    spacing: Optional[float] = Field(
        default=None,
        description=(
            "Spacing between boundary points, by default defined as the minimum "
            "distance between points in the dataset"
        ),
    )
    sel_method: Literal["idw", "nearest"] = Field(
        default="idw",
        description=(
            "Wavespectra method to use for selecting boundary points from the dataset"
        ),
    )
    tolerance: float = Field(
        default=1.0,
        description=(
            "Wavespectra tolerance for selecting boundary points from the dataset"
        ),
        ge=0,
    )
    rectangle: Literal["closed", "open"] = Field(
        default="closed",
        description=(
            "Defines whether boundary is defined over an closed or open rectangle"
        ),
    )
    filter: Optional[Filter] = Field(
        default_factory=Filter,
        description="Optional filter specification to apply to the dataset",
    )

    @model_validator(mode="after")
    def assert_has_wavespectra_accessor(self) -> "DataBoundary":
        dset = self.source.open()
        if not hasattr(dset, "spec"):
            raise ValueError(f"Wavespectra compatible source is required")
        return self

    @property
    def ds(self):
        """Return the filtered xarray dataset instance."""
        dset = super().ds
        if dset.efth.size == 0:
            raise ValueError(f"Empty dataset after applying filter {self.filter}")
        return dset

    def _boundary_resolutions(self, grid):
        """Boundary resolution based on the shortest distance between points.

        The boundary resolution should be based on the dataset resolution instead of
        the grid resolution to avoid creating points unecessarily. Here we find the
        minimum distance between points in the dataset and use that to define the
        boundary resolution ensuring the grid sizes are divisible by the resolution.

        """
        # Find out the minimum distance between points in the original dataset
        buffer = 2 * min(grid.dx, grid.dy)
        x0, y0, x1, y1 = grid.bbox(buffer=buffer)
        # Select dataset points just outside the actual grid to optimise the search
        ds = self.ds.spec.sel([x0, x1], [y0, y1], method="bbox")
        points = list(zip(ds.lon.values, ds.lat.values))
        min_distance = find_minimum_distance(points)
        # Calculate resolutions ensuring at least 3 points per side
        xlen = grid.maxx - grid.minx
        nx = max(xlen // min_distance, 3)
        dx = xlen / nx
        ylen = grid.maxy - grid.miny
        ny = max(ylen // min_distance, 3)
        dy = ylen / ny
        return dx, dy

    def _boundary_points(self, grid):
        """Coordinates of boundary points based on grid bbox and dataset resolution."""
        if self.spacing is None:
            dx, dy = self._boundary_resolutions(grid)
            spacing = min(dx, dy)
        else:
            spacing = self.spacing
        points = grid.points_along_boundary(spacing=spacing)
        if len(points.geoms) < 4:
            logger.warning(
                f"There are only {len(points)} boundary points (less than 1 point per grid side), "
                f"consider setting a smaller spacing (the current spacing is {spacing})"
            )
        xbnd = np.array([p.x for p in points.geoms])
        ybnd = np.array([p.y for p in points.geoms])
        return xbnd, ybnd

    def _filter_grid(self, grid, buffer=0.1):
        """Required by SwanForcing"""
        pass

    def get(self, stage_dir: str, grid: SwanGrid) -> str:
        """Write the data source to a new location"""
        xbnd, ybnd = self._boundary_points(grid)
        ds = self.ds.spec.sel(
            lons=xbnd,
            lats=ybnd,
            method=self.sel_method,
            tolerance=self.tolerance,
        )
        filename = f"{self.id}.bnd"
        filepath = Path(stage_dir) / filename
        ds.spec.to_swan(filepath)
        cmd = f"BOUNDNEST1 NEST '{filename}' {self.rectangle.upper()}"
        return cmd

    def plot(self, model_grid=None, cmap="turbo", fscale=10, ax=None, **kwargs):
        return scatter_plot(
            self, model_grid=model_grid, cmap=cmap, fscale=fscale, ax=ax, **kwargs
        )

    def plot_boundary(
        self, model_grid=None, cmap="turbo", fscale=10, ax=None, **kwargs
    ):
        """Plot the boundary points on a map."""

        xbnd, ybnd = self._boundary_points(model_grid)
        ds = self.ds.spec.sel(
            lons=xbnd,
            lats=ybnd,
            method=self.sel_method,
            tolerance=self.tolerance,
        )
        fig, ax = model_grid.plot(ax=ax, fscale=fscale, **kwargs)
        return scatter_plot(
            self,
            ds=ds,
            model_grid=model_grid,
            cmap=cmap,
            fscale=fscale,
            ax=ax,
            **kwargs,
        )


def scatter_plot(
    bnd, ds=None, model_grid=None, cmap="turbo", fscale=10, ax=None, **kwargs
):
    """Plot the grid"""

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    if ds is None:
        ds = bnd.ds

    # First set some plot parameters:
    minLon, minLat, maxLon, maxLat = (
        ds[bnd.lonname].values[0],
        ds[bnd.latname].values[0],
        ds[bnd.lonname].values[-1],
        ds[bnd.latname].values[-1],
    )
    extents = [minLon, maxLon, minLat, maxLat]

    if ax is None:
        # create figure and plot/map
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(fscale, fscale * (maxLat - minLat) / (maxLon - minLon)),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        # ax.set_extent(extents, crs=ccrs.PlateCarree())

        coastline = cfeature.GSHHSFeature(
            scale="auto", edgecolor="black", facecolor=cfeature.COLORS["land"]
        )

        ax.add_feature(coastline)
        ax.add_feature(cfeature.BORDERS, linewidth=2)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=2,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    ax.scatter(ds[bnd.lonname], ds[bnd.latname], transform=ccrs.PlateCarree())
