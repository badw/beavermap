import numpy as np 
import h5py
import pyFAI
import fabio
import os
import psutil 
import tqdm_pathos
import itertools as it 
from typing import Union,Optional
class BeaverMap:
    def __init__(
        self,
            h5_file:str,
            poni:Optional[Union[dict,str]],
            mask:Optional[Union[np.ndarray,str]],
            chunk_size:int = 100,
    ):
        '''
        BeaverMap - Beam Weaver Mapping 

        code for multiprocessing ESRF synchrotron data and running basic analysis and plotting 

        Args: 
        h5_file:str - location of h5_file from esrf data 
        poni: Union(dict,str) - location of poni file or dict to be read into pyFAI.load
        mask: Union(str,np.array) - location of mask_file or mask_file_data in a np.array that is used in the max_summation and integration 
        chunk_size: int - size of the data chunks (for memory efficiency) - default = 100 
        '''
        
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

        self.h5_file = h5_file
        self.poni = poni
        self.mask = mask
        self.chunk_size = chunk_size
        self.location = '1.1/measurement/eiger'
        self.default_region = (0,45)
        self.max_sum_results = None 
        self.integrate_results = None
        try:
            if isinstance(self.mask,dict):
                self.mask_data = self.mask
            else:
                self.mask_data = np.array(fabio.open(self.mask).data)
        except Exception as e:
            raise RuntimeError(e)

        try:
            self.ai = pyFAI.load(self.poni)
            self.integrate_function = self.ai.integrate1d # may not be necessary - but this function has a large memory toll when opened multiple times
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

            self.chunks = list(
                self.data_chunker(np.arange(self.n_images), self.chunk_size)
            )

    
    @staticmethod        
    def data_chunker(
        iterable_list:list,
        chunk_size:int
        ):
        """
        given a list, returns "chunks" of a given "chunksize" (allows for asymmetrical chunks)
        Args: 
        iterable_list: a list of ints, or floats
        chunksize: an integer number in which to divide the iterable_list i.e. 100 = a chunk of size 100 

        Returns:
        iterable list 
        """
        its = iter(iterable_list)
        return iter(lambda: tuple(it.islice(its, chunk_size)), ())
        
    @property
    def default_integrate_args(self):
        """
        generates a set of default integration arguments for use when integrating the data 
        """
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
    

    def max_sum_worker(self,image_chunk:list):
        """
        function that generates the max summation of an array 
        Args: 
        image_chunk: list 
        returns: 
        np.max(image_chunk)
        """
        with h5py.File(self.h5_file,'r') as f:
            maximum = np.max(f[self.location][np.array(image_chunk)],axis=0)
        return(maximum)
    
    def max_sum(self,ncpus:int=-1,tqdm_kwargs:dict={}):
        """
        runs the max summation for the given data 
        Args:
        ncpus:int = -1 ; number of cpus used to multiprocess the data (default = -1 = max number of cpus available)
        tqdm_kwargs:dict = {} ; kwargs to pass to pathos_tqdm

        Returns:
        np.array
        """
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
            image_chunk:list,
            integrate_args:dict,
            regions:list,
    ):
        """
        function that integrates the given data 
        Args: 
        image_chunk : list - list of images to be integrated 
        integrate_args: dict - integration args to be passed to pyFAI.integrate1d
        regions: list - 2theta regions to be sampled 
        Returns:
        np.array
        """

        full_data = np.zeros((len(regions), self.dim0, self.dim1))

        for image in image_chunk:
            i0 = int(np.floor(image / self.dim1))  # check these...
            i1 = image - self.dim1 * int(np.floor(image / self.dim1))
            with h5py.File(self.h5_file, "r") as f:
                data = f[self.location][image]
                integrated = np.asarray(
                    self.integrate_function(data=data, mask=self.mask_data, **integrate_args)
                )[0:2]

            for i, r in enumerate(regions):
                _arrmask = (integrated[0] >= r[0]) & (integrated[0] <= r[1])
                full_data[i][i0, i1] = np.sum(integrated[1][_arrmask])

        return(full_data)
    
    def integrate(
            self,
            ncpus:int = -1,
            regions:list = None,
            integrate_args:dict = None,
            tqdm_kwargs:dict = {}
            ):
        """
        multiprocessed integration runner 
        Args: 
        ncpus:int = -1 ; number of cpus used to multiprocess the data (default = -1 = max number of cpus available)
        regions:list ; two theta regions to be sampled 
        integrate_args: dict ; integration kwargs to be passed to pyFAI.integrate1d
        tqdm_kwargs: dict ; pathos_tqdm kwargs

        Returns:
        np.array

        """
        
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
        
        results = tqdm_pathos.map(
            self.integrate_worker,
            self.chunks,
            self.integrate_args,
            regions,
            n_cpus = ncpus if ncpus != -1 else psutil.cpu_count(),
            tqdm_kwargs=_tqdm_kwargs)
        
        self.integrate_results = np.array(results).sum(axis=0)

        return(np.array(results).sum(axis=0))
    
    def save_integration(
            self,filename:str ='beavermap_integration_data.npz'
            ):
        """
        save the integration data to np.npz
        Args: 
        filename:str ; default = 'beavermap_integration_data.npz'
        """
        metadata = self.__dict__
        if self.integrate_results is not None:
            np.savez(filename, features=self.integrate_results, metadata=metadata)
        else:
            raise AttributeError('no integration results found.')

    def save_max_sum(self,filename='beavermap_max_sum_data.npz'):
        """
        save the integration data to np.npz
        Args: 
        filename:str ; default = 'beavermap_max_sum_data.npz'
        """
        metadata = self.__dict__
        if self.max_sum_results is not None:
            np.savez(filename, features=self.max_sum_results, metadata=metadata)
        else:
            raise AttributeError('no max sum results found.')