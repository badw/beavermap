import numpy as np 
import h5py
import pyFAI
import fabio
import os
from queue import Empty
import tqdm 
from pathos.helpers import mp as pmp
import psutil 
import tqdm_pathos
class BeaverMap:
    def __init__(
        self,
            h5_file,
            poni_file,
            mask_file,
            chunk_size,
            location,
            #ncpus,
            **kws
    ):
        '''
        Code for easy paralellisation and analysis of Synchrotron Data from ESRF 
        '''
        
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

        self.h5_file = h5_file
        self.poni_file = poni_file
        self.mask_file = mask_file
        self.chunk_size = chunk_size
        self.location = location
        self.default_region = (0,45)
        self.max_sum_results = None 
        self.integrate_results = None
        try:
            self.mask_data = np.array(fabio.open(self.mask_file).data)
        except Exception as e:
            raise RuntimeError(e)

        try:
            self.ai = pyFAI.load(self.poni_file)
            self.integrate_function = self.ai.integrate1d
        except Exception as e:
            raise RuntimeError(e)
        
        ''' define some useful parameters'''
        with h5py.File(h5_file,'r') as f:
            self.n_images = f[self.location].shape[0]
            self.dimensions = f[self.location].shape[1:]
            self.dim0 = f['/1.1/technique/dim0'][()]
            self.dim1 = f['/1.1/technique/dim1'][()]

            self.mean = np.mean(f['1.1/measurement/ct34'])
            self.median = np.median(f['1.1/measurement/ct34'])
            self.max_val = np.max(f['1.1/measurement/ct34'])
            self.min_val = np.min(f['1.1/measurement/ct34'])
            '''
            separating into "chunks" for memory efficiency default = 100
            '''

            self.chunks = self.data_chunker(self.n_images,self.chunk_size)

    
    @staticmethod        
    def data_chunker(length,chunksize):
        try:
            chunks = np.reshape(
                np.arange(0, length, 1),
                (int(length/chunksize), chunksize)
            )
            return(chunks)
        except ValueError as error:
            print(error)

            
    @property
    def default_integrate_args(self):
        self.integrate_args = {'npt': 10000,
                               'correctSolidAngle': False,
                               'error_model': 'poisson',
                               'azimuth_range': None,
                               'radial_range': (0, 45),  # (16.70, 17.05),
                               'polarization_factor': 1,
                               'method': 'full',
                               'unit': '2th_deg',
                               'normalization_factor': 1,
                               'safe':False
                               }
    

    def max_sum_worker(self,image_chunk):
        with h5py.File(self.h5_file,'r') as f:
            maximum = np.max(f[self.location][image_chunk],axis=0)
        return(maximum)
    
    def max_sum(self,ncpus:int=-1,tqdm_kwargs={}):
        _tqdm_kwargs = {'ncols':80,'desc':"performing max summation"}
        _tqdm_kwargs.update(tqdm_kwargs)

        results = tqdm_pathos.map(
            self.max_sum_worker,
            self.chunks,
            n_cpus = ncpus if ncpus != -1 else psutil.cpu_count(),
            tqdm_kwargs=_tqdm_kwargs)
        
        self.max_sum_results = np.array(results).max(axis=0)
        return(np.array(results).max(axis=0))

    def integrate_worker(
            self,
            image_chunk,
            args,
            regions,
    ):
        #with h5py.File(self.h5_file, "r") as f:
        #    h5filedata = f[self.location][image_chunk]

        full_data = np.zeros((len(regions), self.dim0, self.dim1))

        for image in image_chunk:
            i0 = int(np.floor(image / self.dim1))  # check these...
            i1 = image - self.dim1 * int(np.floor(image / self.dim1))
            with h5py.File(self.h5_file, "r") as f:
                data = f[self.location][image]
                integrated = np.asarray(
                    self.integrate_function(data=data, mask=self.mask_data, **args)
                )[0:2]

            for i, r in enumerate(regions):
                _arrmask = (integrated[0] >= r[0]) & (integrated[0] <= r[1])
                full_data[i][i0, i1] = np.sum(integrated[1][_arrmask])

        return(full_data)
    
    def integrate(
            self,
            ncpus:int = -1,
            regions=None,
            integrate_args=None,
            tqdm_kwargs = {}
            ):
        
        _tqdm_kwargs = {'ncols':80,'desc':"performing integration"}
        _tqdm_kwargs.update(tqdm_kwargs)

        if not integrate_args:
            self.default_integrate_args
        else:
            self.integrate_args = integrate_args

        if regions is not None:
            self.regions = regions
        else:
            raise AttributeError('no 2theta regions supplied')

        #try:
        #    images = np.arange(self.n_images).reshape((-1,chunk_size))
        #except Exception as e:
        #    raise ValueError(e)
        
        results = tqdm_pathos.map(
            self.integrate_worker,
            self.chunks,
            self.integrate_args,
            regions,
            n_cpus = ncpus if ncpus != -1 else psutil.cpu_count(),
            tqdm_kwargs=_tqdm_kwargs)
        
        self.integrate_results = np.array(results).sum(axis=0)
        return(np.array(results).sum(axis=0))
    
    def save_integration(self,filename='beavermap_integration_data.npz'):
        metadata = self.__dict__
        if self.integrate_results is not None:
            np.savez(filename, features=self.integrate_results, metadata=metadata)
        else:
            raise AttributeError('no integration results found.')

    def save_max_sum(self,filename='beavermap_max_sum_data.npz'):
        metadata = self.__dict__
        if self.max_sum_results is not None:
            np.savez(filename, features=self.max_sum_results, metadata=metadata)
        else:
            raise AttributeError('no max sum results found.')