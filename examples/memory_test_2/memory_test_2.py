from memory_profiler import profile 
from beavermap import BeaverMap
import os 
import numpy as np 
import h5py
import copy 
import time 
import pyFAI
import gc 
ranges = [(9, 13),     # XLPE/EVA
    (18.957, 19, 357),
    (13.879, 14.103),  # Graphite
    (11.017, 11.068),  # Mica
    (12.226, 12.396),
    (13.426, 13.603),
    (16.484, 16.660),
    (14.393, 14.793),  # NaCl
    (16.663, 17.063),
    (23.733, 24.133),
    (27.933, 28.333),
    (29.213, 29.613),
    (33.950, 34.250),
    (37.043, 37.443),
    (38.048, 38.448),
    (41.940, 42.200),
    (44.630, 44.859),
    (14.993, 15.173),  # KCl
    (21.303, 21.484),
    (25.169, 25.324),
    (26.124, 26.472),
    (15.522, 15.803),  # CaCO3
    (19.054, 19.154),
    (20.720, 20.948),
    (22.690, 22.815),
    (24.650, 24.812),
    (25.328, 25.596),
    (29.687, 29.917),
    (31.276, 31.910),
    (33.147, 33.501),
    (33.753, 33.946),
    (19.154, 19.368),  # ZnO
    (18.193, 18.464),
    (24.873, 25.117),
    (32.292, 32.617),
    (34.783, 35.007),
    (35.227, 35.568),
    (38.717, 39.275),
    (44.330, 44.574),
    (14.45, 14.70),    #Unknowns
    (28.3, 28.5),
    (26.55, 26.85),
    (13, 13.2),
    (17.75, 17.95),
    (18.3, 18.7),
    (24.26, 24.80),
    (28.9, 29.10)
              ]

@profile
def func2():
    image = 100
    i0 = int(np.floor(image/bm.dim1))
    i1 = image - bm.dim1 * int(np.floor(image / bm.dim1))    

    with h5py.File(bm.h5_file,'r') as f:
        data = f[bm.location][image]

    integrate = list(bm.ai.integrate1d(data=data,mask=bm.mask_data,**bm.integrate_args))[0:2]

    np.save('temp.npy',integrate)
    del bm.ai 
@profile
def func3():
    integrate = np.load('temp.npy')
    image = 100
    i0 = int(np.floor(image/bm.dim1))
    i1 = image - bm.dim1 * int(np.floor(image / bm.dim1))    
    
    full_data_2 = np.zeros((len(ranges), bm.dim0, bm.    dim1))
    for i, r in enumerate(ranges):
        _arrmask = (integrate[0] >= r[0]) & (integrate[0]     <= r[1])
        full_data_2[i][i0, i1] = np.sum(integrate[1]    [_arrmask])    

    print(full_data_2.nbytes/1024/1024)
if __name__ == '__main__':
    bm = BeaverMap(
        h5_file = '../data/al2o3_m330p0/al2o3_m330p0.h5',
        poni_file = '../data/m330p0.poni',
        mask_file = '../data/mask.edf',
        chunk_size = 100,
        location = '1.1/measurement/eiger',
        nworkers = 1,
    )
    bm.default_integrate_args
    func2()
    time.sleep(2)
    func3()
