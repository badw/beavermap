import numpy as np 
from beavermap.generate import BeaverMap
import pyFAI 
import fabio
from scipy.signal import find_peaks
from typing import Optional, Union 
""" work in progress"""
class BeaverMapPeakFinder:

    def __init__(self,BeaverMap):
        self.max_sum = BeaverMap.max_sum_results
        self.metadata = BeaverMap.__dict__

        if 'integrate_args' not in self.metadata:
            BeaverMap.default_integrate_args
            self.metadata = BeaverMap.__dict__

        try:
            self.mask_data = np.array(fabio.open(self.metadata['mask']).data)
        except Exception as e:
            raise RuntimeError(e)

    def integrate_max_sum(self):
        result = pyFAI.load(self.metadata['poni']).integrate1d(
            self.max_sum,
            mask = self.mask_data,
            **self.metadata['integrate_args']) 
        
        self.integrated_max_sum = result
        
        return(result)
    
    def peak_finder(
            self,
            integrated_data:Optional[None]=None,
            **scipy_kwargs):
        if integrated_data is None:
            integrated_data = self.integrate_max_sum()

        peaks = find_peaks(integrated_data[1],**scipy_kwargs)

        self.peaks = peaks

        return(integrated_data[0][peaks[0]])