#-----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO 
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from .core import BaseModel, BaseGrid
import logging


logger = logging.getLogger(__file__)

FILL_VALUE = -99.0


class SwanModel(BaseModel):

    def __init__(self, run_id='run_0001', template=None, checkout=None, settings=None, output_dir=None):

        super().__init__(model="SWAN", run_id=run_id, template=template, checkout=checkout,settings=settings, output_dir=output_dir)

    def _get_grid(self, key=None):
        from intake.source.utils import reverse_format

        if key is not None and key in self.settings.keys():
            grid_spec = self.settings[key]
        else:
            grid_spec = self.default_context['bottom_grid']
        fmt = '{gridtype} {x0:f} {y0:f} {rot:f} {nx:d} {ny:d} {dx:f} {dy:f}'
        if 'EXC' in grid_spec: # append excluded value string
            fmt += ' EXC {exc:f}'
        grid_params = reverse_format(fmt, grid_spec)
        return SwanGrid(**grid_params)

        
class SwanGrid(BaseGrid):

    def __init__(self,gridtype=None,x0=None,y0=None,rot=None,dx=None,dy=None,nx=None,ny=None,exc=None,gridfile=None):

        super(BaseGrid, self).__init__()

        if gridtype == "REG":
            mandatory_args = ['x0','y0','rot','dx','dy','nx','ny']
        elif gridtype == "CURV":
            mandatory_args = ['gridfile',]
        else:
            raise ValueError("Unknown SwanGrid type = " + str(gridtype))

        self.__gridtype = gridtype
        self.__x0 = x0
        self.__y0 = y0
        self.__rot = rot
        self.__dx = dx
        self.__dy = dy
        self.__nx = nx
        self.__ny = ny
        self.__exc = exc
        self.__gridfile = gridfile
        
        if any([self.__dict__['_SwanGrid__' + k] is None for k in mandatory_args]):
            raise ValueError(f"SwanGrid object of type = {gridtype} require values for the following arguments " + str(mandatory_args))

        self.__regen_grid()

    def __regen_grid(self):
        if self.gridtype == "REG": 
             _x, _y = self.__gen_reg_cgrid()
        elif self.gridtype == "CURV":
             _x, _y  = self.__gen_curv_cgrid()
        self.x = _x
        self.y = _y

    def __gen_reg_cgrid(self):
        
        # Grid at origin
        i = np.arange(0.,self.dx*(self.nx+0.1),self.dx)
        j = np.arange(0.,self.dy*(self.ny+0.1),self.dy)
        ii, jj = np.meshgrid(i, j)

        # Rotation
        alpha = -self.rot * np.pi / 180.
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        gg = np.dot(np.vstack([ii.ravel(), jj.ravel()]).T, R)

        # Translation
        x = gg[:, 0] + self.x0
        y = gg[:, 1] + self.y0

        x = np.reshape(x,ii.shape)
        y = np.reshape(y,ii.shape)

        return x, y

    def __gen_curv_cgrid(self):
        ''' loads a SWAN curvilinear grid and returns cgrid lat/lons and
        command to be used in SWAN contol file. The Default grid is one I made using
        Deltares' RGFGrid tool and converted to a SWAN-friendly formate using Deltares
        OpenEarth code "swan_io_grd.m"

        '''
        # number of grid cells in the 'x' and 'y' directions:
        # (you can get this from d3d_qp.m or other Deltares OpenEarth code)
        nX = self.nx
        nY = self.ny

        grid_Data = open(self.gridpath).readlines()
        ix = grid_Data.index("x-coordinates\n")
        iy = grid_Data.index("y-coordinates\n")
        lons=[]
        lats=[]
        for idx in np.arange(ix+1,iy):
            lons.append(re.sub('\n','',grid_Data[idx]).split())
        for idx in np.arange(iy+1,len(grid_Data)):
            lats.append(re.sub('\n','',grid_Data[idx]).split())
        flatten = lambda l: [item for sublist in l for item in sublist]
        lons = np.array(flatten(lons)).astype(np.float)
        lats = np.array(flatten(lats)).astype(np.float)

        x = np.reshape(lats, (nX, nY))
        y = np.reshape(lons, (nX, nY))

        return x, y

    @property
    def gridtype(self):
        return self.__gridtype

    # The folling properties and setters are necessary
    # to ensure the grid is regnerated when any one of them changes
    @property
    def x0(self):
        return self.__x0
    
    @x0.setter
    def x0(self, x0):
        self.__x0 = x0
        self.__regen_grid()

    @property
    def y0(self):
        return self.__y0
    
    @y0.setter
    def y0(self, y0):
        self.__y0 = y0
        self.__regen_grid()

    @property
    def rot(self):
        return self.__rot
    
    @rot.setter
    def rot(self, rot):
        self.__rot = rot
        self.__regen_grid()

    @property
    def dx(self):
        return self.__dx
    
    @dx.setter
    def dx(self, dx):
        self.__dx = dx
        self.__regen_grid()

    @property
    def dy(self):
        return self.__dy
    
    @dy.setter
    def dy(self, dy):
        self.__dy = dy
        self.__regen_grid()

    @property
    def nx(self):
        return self.__nx
    
    @nx.setter
    def nx(self, nx):
        self.__nx = nx
        self.__regen_grid()

    @property
    def ny(self):
        return self.__ny
    
    @ny.setter
    def ny(self, ny):
        self.__ny = ny
        self.__regen_grid()

    @property
    def exc(self):
        return self.__exc
    
    @exc.setter
    def exc(self, exc):
        self.__exc = exc

    @property
    def inpgrid(self):
        if self.gridtype == "REG":
            inpstr = f'REG {self.x0} {self.y0} {self.rot} {self.nx-1:0.0f} {self.ny-1:0.0f} {self.dx} {self.dy}'
            if self.exc is not None:
                inpstr += f' EXC {self.exc}'
            return inpstr
        elif self.gridtype == "CURV":
            raise NotImplementedError('Curvilinear grids not supported yet')
            # return f'CURVilinear {self.nx-1:0.0f} {self.ny-1:0.0f} \nREADGRID COOR 1 \'{os.path.basename(self.gridpath)}\' 1 0 1 FREE' 

    @property
    def cgrid(self):
        if self.gridtype == "REG":
            return f'REG {self.x0} {self.y0} {self.rot} {self.dx*self.nx} {self.dy*self.ny} {self.nx-1:0.0f} {self.ny-1:0.0f}'
        elif self.gridtype == "CURV":
            raise NotImplementedError('Curvilinear grids not supported yet')
            # return (f'CURVilinear {self.nx-1:0.0f} {self.ny-1:0.0f}',f'READGRID COOR 1 \'{os.path.basename(self.gridpath)}\' 1 0 1 FREE')

    def nearby_spectra(self,ds_spec,dist_thres=0.05,plot=True):
        """Find points nearby and project to the boundary

        Parameters
        ----------
        ds_spec: xarray.Dataset
            an XArray dataset of wave spectra at a number of points. 
            Dataset variable names standardised using wavespectra.read_* 
            functions. 
            
            See https://wavespectra.readthedocs.io/en/latest/api.html#input-functions
        dist_thres: float, optional [Default: 0.05]
            Maximum distance to translate the input spectra to the grid boundary
        plot: boolean, optional [Default: True]
            Generate a plot that shows the applied projections

        Returns
        -------
        xarray.Dataset
            A subset of ds_spec with lat and lon coordinates projected to the boundary
        """

        bbox = self.bbox(buffer=dist_thres)
        minLon, minLat, maxLon, maxLat = bbox

        inds = np.where((ds_spec.lon > minLon) & (ds_spec.lon < maxLon) & (ds_spec.lat > minLat) & (ds_spec.lat < maxLat))[0]
        ds_spec=ds_spec.isel(site=inds)

        #Work out the closest spectral points
        def _nearestPointOnLine(p1, p2, p3):
            # calculate the distance of p3 from the line between p1 and p2 and return 
            # the closest point on the line

            from math import sqrt, fabs
            a = p2[1] - p1[1]
            b = -1. * (p2[0] - p1[0])
            c = p2[0] * p1[1] - p2[1] * p1[0]

            dist = fabs(a * p3[0] + b * p3[1] + c) / sqrt(a ** 2 + b ** 2)
            x = (b * (b * p3[0] - a * p3[1]) - a * c) / (a ** 2 + b ** 2)
            y = (a * (-b * p3[0] + a * p3[1]) - b * c) / (a ** 2 + b ** 2)

            return dist, x, y

        bx, by = self.boundary_points()
        pol = np.stack([bx,by])

        #Spectra points
        ds_spec.lon.load()
        ds_spec.lat.load()
        ds_spec['lon_original']=ds_spec['lon']
        ds_spec['lat_original']=ds_spec['lat']
        p3s = list(zip(ds_spec.lon.values,ds_spec.lat.values))

        if plot:
            fig,ax = self.plot()
            ax.scatter(ds_spec.lon,ds_spec.lat)

        specPoints = []
        specPointCoords = []
        for i in range(pol.shape[1]-1):
            p1 = pol[:,i]
            p2 = pol[:,i+1]
            line = np.stack((p1, p2))
            output = np.array(list(map(lambda xi: _nearestPointOnLine(p1, p2, xi), p3s)))
            dists = output[:, 0]
            segmentPoints = output[:, 1:]
            inds = np.where((dists < dist_thres))[0]
            
            # Loop through the points projected onto the line
            for ind in inds:
                specPoint=ds_spec.isel(site=ind)

                segLon = segmentPoints[ind, 0]
                segLat = segmentPoints[ind, 1]
                
                if plot:
                    ax.plot([segLon, specPoint.lon],[segLat, specPoint.lat],color='r',lw=2)
                    ax.scatter(specPoint.lon,specPoint.lat,marker='o',color='b')
                    ax.scatter(segLon,segLat,marker='x',color='g')
                
                specPoint['lon'] = segLon
                specPoint['lat'] = segLat
                specPoints.append(specPoint)
                
            logger.debug(f"Segment {i} - Indices {inds}")

        if plot: 
            fig.show()

        ds_boundary = xr.concat(specPoints,dim='site')
        return ds_boundary


def dset_to_swan(
    dset: xr.Dataset,
    output_file: str,
    variables: list,
    fmt: str="%4.2f",
    fill_value: float=FILL_VALUE,
    time_dim="time",
):
    """Convert netcdf into SWAN ASCII file.

    Parameters
    ----------
    dset: xr.Dataset
        Dataset to write in SWAN ASCII format.
    output_file: str
        Local file name for the ascii output file.
    variables: list
        Variables to write to ascii.
    fmt: str
        String float formatter.
    fill_value: float
        Fill value.
    time_dim: str
        Name of the time dimension if available in the dataset.

    """
    # Input checking
    for data_var in variables:
        if data_var not in dset.data_vars:
            raise ValueError(f"Variable {data_var} not in {dset}")
        if dset[data_var].ndim not in (2, 3):
            raise NotImplementedError(
                "Only 2D and 3D datasets are supported but "
                f"dset.{data_var} has {dset[data_var].ndim} dims"
            )

    # Ensure time is a dimension so iteration will work
    if time_dim not in dset.dims:
        dset = dset.expand_dims(time_dim, 0)

    # Write to ascii
    logger.info(f"Writing SWAN ASCII file: {output_file}")
    with open(output_file, "w") as stream:
        for time in dset[time_dim]:
            logger.debug(f"Appending Time {pd.to_datetime(time.values)} to {output_file}")
            for data_var in variables:
                logger.debug(f"Appending Variable {data_var} to {output_file}")
                data = dset[data_var].sel(time=time).fillna(fill_value).values
                np.savetxt(fname=stream, X=data, fmt=fmt, delimiter="\t")

    return output_file


@xr.register_dataset_accessor("swan")
class Swan_accessor(object):

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def grid(self, x="lon", y="lat", exc=FILL_VALUE):
        """SWAN Grid object for this dataset.

        TODO: Perhaps we could assume y, x are the last 2 axes.

        """
        return SwanGrid(
            gridtype="REG",
            x0=float(self._obj[x].min()),
            y0=float(self._obj[y].min()),
            dx=float(np.diff(self._obj[x]).mean()),
            dy=float(np.diff(self._obj[y]).mean()),
            nx=len(self._obj[x]),
            ny=len(self._obj[y]),
            rot=0,
            exc=exc,
        )

    def to_bottom_grid(
        self,
        output_file,
        fmt="%4.2f",
        x="lon",
        y="lat",
        z="depth",
        vmin=0.0,
        fill_value=FILL_VALUE,
        fac=1.0
    ):
        """Write SWAN inpgrid BOTTOM file.

        Parameters
        ----------
        output_file: str
            Local file name for the ascii output file.
        fmt: str
            String float formatter.
        x: str
            Name of the x-axis in the dataset.
        y: str
            Name of the y-axis in the dataset.
        z: str
            Name of the bottom variable in the dataset.
        vmin: float
            Minimum value below which depths are masked.
        fill_value: float
            Fill value.
        fac: float
            Multiplying factor, only necessary when the data are not in meters.

        Returns
        -------
        inpgrid: str
            SWAN INPgrid command instruction.
        readinp: str
            SWAN READinp command instruction.

        TODO: Merge this method with `to_inpgrid`.
        TODO: Change cookiecutter so readinp is used.

        """
        dset_to_swan(
            dset=self._obj.where(self._obj > vmin).transpose(..., y, x),
            output_file=output_file,
            fmt=fmt,
            variables=[z],
            fill_value=fill_value,
        )
        grid = self.grid(x=x, y=y)
        inpgrid = f"{grid.inpgrid}"
        readinp = f"{fac} '{Path(output_file).name}' 3 FREE"
        return inpgrid, readinp

    def to_inpgrid(self,output_file,grid=None,var='WIND',
                   fmt='%.2f',x='lon',y='lat',z1='u10',z2=None,time='time'):
        ''' This function writes to a SWAN inpgrid format file (i.e. WIND)
            
            Args:
            TBD
        '''
        import numpy as np
        import pandas as pd
        import os

        ds = self._obj
        
        if grid is None:
            # If no grid passed in assume it is a REG grid
            if len(ds[x].shape) == 1:
                grid = SwanGrid(gridtype="REG",
                                x0=float(ds[x].min()),
                                y0=float(ds[y].min()),
                                dx=float(np.diff(ds[x]).mean()),
                                dy=float(np.diff(ds[y]).mean()),
                                nx=len(ds[x]),
                                ny=len(ds[y]),
                                rot=0
                )
            else:
                raise ValueError('No grid specified for output and number of dims for x-coordinate > 1')
        # else:
        #     raise NotImplementedError('Specifying an alternative output grid is currently not implemented. Only regular grids supported.')

        # ds = ds.transpose((time,) + ds[x].dims)
        dt = np.diff(ds.time.values).mean()/pd.to_timedelta(1,'H')

        inptimes = []
        with open(output_file, 'wt') as f:
            # iterate through time
            for ti, windtime in enumerate(ds.time.values):

                time_str = pd.to_datetime(windtime).strftime("%Y%m%d.%H%M%S")
                logger.debug(time_str)

                # write SWAN time header to file:
                f.write(f'{time_str}\n')

                # Write first component to file
                z1t = np.squeeze(ds[z1].isel(time=ti).values)
                np.savetxt(f,z1t,fmt=fmt)

                if z2 is not None:
                    z2t = np.squeeze(ds[z2].isel(time=ti).values)
                    np.savetxt(f,z2t,fmt=fmt)    

                inptimes.append(time_str)

        if len(inptimes)<1:
            import os
            os.remove(output_file)
            raise ValueError(f'***Error! No times written to {output_file}\n. Check the input data!')

        input_strings = (f"{grid.inpgrid} NONSTATION {inptimes[0]} {dt} HR",f"1 '{os.path.basename(output_file)}' 3 0 1 0 FREE")

        return input_strings

    def to_tpar_boundary(self,dest_path,
                              boundary,
                              interval,
                              x_var='lon',
                              y_var='lat',
                              hs_var='sig_wav_ht',
                              per_var='pk_wav_per',
                              dir_var='pk_wav_dir',
                              dir_spread=20.):
        ''' This function writes parametric boundary forcing to a set of 
            TPAR files at a given distance based on gridded wave output. It returns the string to be included in the Swan INPUT file.

            At present simple nearest neighbour point lookup is used.
            
            Args:
            TBD
        '''
        from shapely.ops import substring

        bound_string = "BOUNDSPEC SEGM XY "
        point_string = "&\n {xp:0.8f} {yp:0.8f} "
        file_string = "&\n {len:0.8f} '{fname}' 1 "

        for xp, yp in boundary.exterior.coords:
            bound_string += point_string.format(xp=xp,yp=yp)

        bound_string += "&\n VAR FILE "

        n_pts = int((boundary.length)/interval)
        splits = np.linspace(0,1.,n_pts)
        boundary_points = []
        j=0
        for i in range(len(splits)-1):
            segment=substring(boundary.exterior, splits[i], splits[i+1],normalized=True)
            xp = segment.coords[1][0]
            yp = segment.coords[1][1]
            logger.debug(f'Extracting point: {xp},{yp}')
            ds_point = self._obj.sel(indexers={x_var:xp,y_var:yp},method='nearest',tolerance=interval)
            if len(ds_point.time)==len(self._obj.time):
                if not np.any(np.isnan(ds_point[hs_var])):
                    with open(f'{dest_path}/{j}.TPAR', 'wt') as f:
                        f.write('TPAR\n')
                        for t in range(len(ds_point.time)):
                            ds_row=ds_point.isel(time=t)
                            lf = '{tt} {hs:0.2f} {per:0.2f} {dirn:0.1f} {spr:0.2f}\n'
                            f.write(lf.format(tt=str(ds_row['time'].dt.strftime('%Y%m%d.%H%M%S').values),
                                            hs=float(ds_row[hs_var]),
                                            per=float(ds_row[per_var]),
                                            dirn=float(ds_row[dir_var]),
                                            spr=dir_spread))
                    bound_string += file_string.format(len=splits[i+1]*boundary.length,
                                                       fname=f'{j}.TPAR')
                    j+=1

        return bound_string
