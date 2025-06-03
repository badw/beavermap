import numpy as np 
import h5py
import pyFAI
import fabio
import os
from queue import Empty, Full
import tqdm 
from pathos.helpers import mp as pmp
import time 
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
            nworkers,
            max_memory:int = 4000,# MB
            reserve_memory:int=500, # MB,
            **kws
    ):
        '''
        Code for easy paralellisation and analysis of Synchrotron Data from ESRF 
        '''
        
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        #os.environ['OMP_NUM_THREADS'] = '1'

        self.h5_file = h5_file
        self.poni_file = poni_file
        self.mask_file = mask_file
        self.chunk_size = chunk_size
        self.location = location
        #cpus
        self.nworkers = nworkers if nworkers != -1 else pmp.cpu_count()
        #memory 
        self.estimated_memory = 1000*1024*1024 # 1gb estimated memory # need to make this depend on numpy 
        self.total_memory = int(psutil.virtual_memory().total) # in MB 
        self.reserve_memory = int(500 * 1024 * 1024)  # 500 MB reserve memory
        if self.reserve_memory/1024/1024 + max_memory < self.total_memory/1024/1024:
            self.max_memory = max_memory * 1024 * 1024  # MB -> bytes
        else:
            raise AttributeError(
                f"max_memory ({max_memory}MB) + reserve_memory ({self.reserve_memory/1024/1024}MB) > total_memory ({self.total_memory/1024/1024}MB) "
            )
        self.in_queue = None
        self.out_queue = None
        self.workers = None
        self.default_region = (0,45)
        try:
            self.mask_data = np.array(fabio.open(self.mask_file).data)
        except Exception as e:
            raise RuntimeError(e)

        try:
            self.ai = pyFAI.load(self.poni_file)
        except Exception as e:
            raise RuntimeError(e)
        
        ''' define some useful parameters'''
        with h5py.File(h5_file,'r') as f:
            self.n_images = f[self.location].shape[0]
            self.dimensions = f[self.location].shape[1:]
            self.dim0 = f['/1.1/technique/dim0'][()]
            self.dim1 = f['/1.1/technique/dim1'][()]
            '''
            separating into "chunks" for memory efficiency default = 100
            '''

            #self.chunks = np.reshape(
            #    np.arange(0,self.n_images,1),
            #    (int(self.n_images/self.chunk_size),self.chunk_size)
            #    )

            self.chunks = self.data_chunker(self.n_images,self.chunk_size)

    def available_memory(self):
        available = psutil.virtual_memory().available
        return(available - self.reserve_memory > self.estimated_memory)
    
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
                               }

    def h5file_data(self):
        return(h5py.File(self.h5_file,'r'))
    
    def safe_put(self,in_q,item,timeout=1.0):
        while True: 
            if not self.available_memory():
                time.sleep(0.5)
                continue
            try:
                in_q.put(item,block=True,timeout=timeout)
                return
            except Full:
                print('currently_full')
                time.sleep(0.5)
    
    def terminate_workers(self):
        if self.workers is not None:
            for i in range(self.nworkers):
                #self.safe_put(self.in_queue,None)
                self.in_queue.put(None)
            
            for w in self.workers:
                w.terminate()
                w.join(0)
            
            self.in_queue.close()
            self.out_queue.close()
            self.workers = None 

    def workers_alive(self):
        return all([worker.is_alive() for worker in self.workers])
    
    def get_result_from_queue(self,pbar=None,ii=None):
        # handle exception gracefully to avoid hanging rocesses
        try:
            result = self.out_queue.get(timeout=20)
            if pbar and ii:
                pbar.update(ii)
        
        except Empty:
            # didn't receive anything for 20 seconds; this could be OK or it could
            # the processes have been killed
            if not self.workers_alive():
                self.terminate_workers()
                raise RuntimeError(
                    "Some subprocessess were killed unexpectedly."
                )
            else:
                return self.get_result_from_queue(pbar,ii)

        if isinstance(result[0], Exception):
            self.terminate_workers()
            raise result[0]
        return result
    
    def max_sum_worker(self,in_q,out_q):
        while True:
            chk = in_q.get()
            with h5py.File(self.h5_file,'r') as f:
                maximum = np.max(f[self.location][chk],axis=0)
                out_q.put(maximum)
          
    def max_sum(self):
        '''
        a lot of the multiprocessing was adapted from AMSET - thanks Alex
        '''

        self.terminate_workers()
        ctx = pmp.get_context('spawn')
        self.in_queue = ctx.Queue()# play around with this value
        self.out_queue = ctx.Queue()

        for chunk in self.chunks: 
            self.in_queue.put(chunk)           

        self.workers = [] 
        for _ in range(self.nworkers):
            self.workers.append(
                ctx.Process(
                    target=self.max_sum_worker,
                    args=(self.in_queue, self.out_queue,)
                )
            )
                
        for w in self.workers:
            w.start()

        _bar_format = "{desc} {n_fmt}/{total_fmt}|{percentage:3.0f}%|{bar}| {elapsed}<{remaining}{postfix}"
        with tqdm.tqdm(
            total=self.n_images,
            desc='performing max summation. images:',
            bar_format=_bar_format,
            ncols=80
            ) as pbar:    
            results = []
            for i in self.chunks:
                results.append(
                    self.get_result_from_queue(pbar,self.chunks.shape[1])
                    )

        total = np.array(results).max(axis=0)

        self.terminate_workers()

        return(total)

    def integrate_worker(
            self,
            in_q,
            out_q,
            args,
            regions
    ):
        while True:
            ### queue memory checker here?
            image = in_q.get()
            with h5py.File(self.h5_file, "r") as f:
                res_map = np.zeros((self.dim0, self.dim1, 2, args["npt"]))
                i0 = int(np.floor(image / self.dim1))  # check these...
                i1 = image - self.dim1 * int(np.floor(image / self.dim1))
                res_map[i0, i1] = np.array(
                    self.ai.integrate1d(
                        data=f[self.location][image], mask=self.mask_data, **args
                    )[0:2]
                )

            full_data = np.zeros((len(regions), self.dim0, self.dim1))

            for i, r in enumerate(regions):
                _arrmask = (res_map[i0, i1][0] >= r[0]) & (res_map[i0, i1][0] <= r[1])
                full_data[i][i0, i1] = np.sum(res_map[i0, i1][1][_arrmask])
            res_map = None
            out_q.put(full_data)

    def integrate(
            self, 
            integrate_args=None, 
            regions=[[0, 100]], 
            ):
        
        if not integrate_args:
            self.default_integrate_args
        else:
            self.integrate_args = integrate_args

        final_results = []

        ctx = pmp.get_context("spawn")

        self.in_queue = ctx.Queue()# can play around with this value
        self.out_queue = ctx.Queue()  # can we combine these into one     queue?
        image_range = np.arange(self.n_images)

        for image in image_range:
            #self.safe_put(self.in_queue,image)
            self.in_queue.put(image)

        self.workers = []
        for _ in range(self.nworkers):
            self.workers.append(
                ctx.Process(
                    target=self.integrate_worker,
                    args=(self.in_queue, self.out_queue, self.integrate_args, regions),
                )
            )
        for w in self.workers:
            w.start()

        _bar_format = "{desc} {n_fmt}/{total_fmt}|{percentage:3.0f}%|{bar}| {elapsed}<    {remaining}{postfix}"
        total = int(len(image_range) * len(image_range) / 2 - len(image_range) / 2)
        divider = len(image_range) / total

        with tqdm.tqdm(
            total=len(image_range),
            desc=f"performing integration",
            bar_format=_bar_format,
            ncols=80,
        ) as pbar:
            results = []
            for ii, image in enumerate(image_range):
                results.append(self.get_result_from_queue(pbar, np.round(divider * ii)))

        self.terminate_workers()

        final_results.extend([np.array(results).sum(axis=0)])

        self.terminate_workers()

        return np.array(final_results).sum(axis=0)
    

    def _pathos_tqdm_integrate_worker(
            self,
            image,
            args,
            regions
    ):
            ### queue memory checker here?
        with h5py.File(self.h5_file,'r') as f:
            res_map = np.zeros((self.dim0, self.dim1, 2, args["npt"]))

            i0 = int(np.floor(image / self.dim1))  # check these...
            i1 = image - self.dim1 * int(np.floor(image / self.dim1))

            res_map[i0, i1] = np.array(
                self.ai.integrate1d(
                    data=f[self.location][image], mask=self.mask_data, **args
                )[0:2]
            )

        full_data = np.zeros((len(regions), self.dim0, self.dim1))

        for i, r in enumerate(regions):
            _arrmask = (res_map[i0, i1][0] >= r[0]) & (res_map[i0, i1][0] <= r[1])
            full_data[i][i0, i1] = np.sum(res_map[i0, i1][1][_arrmask])
        return full_data
    
    def _pathos_tqdm_integrate(self,integrate_args,regions):
        pool = pmp.Pool(maxtasksperchild=10)
        results = tqdm_pathos.map(
            self._pathos_tqdm_integrate_worker,
            iterable=np.arange(self.n_images),
            args=integrate_args,
            regions=regions,
            n_cpus=self.nworkers,
            pool=pool
            
        )