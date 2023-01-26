import pandas as pd
from pathlib import Path
import nmrglue as ng

def read_topspin_data(data_path, dataset, expno, procno, rowno=None, window=None):

    # get complete data path
    full_path = Path(data_path, dataset, expno, 'pdata', procno)

    # read processed data
    dic, data = ng.bruker.read_pdata(str(full_path), read_procs=True, read_acqus=False, scale_data=True, all_components=False)
    data = pd.DataFrame(data)

    # get universal dictionnary
    udic = ng.bruker.guess_udic(dic,data)

    # exp dimensions
    ndim = udic['ndim']

    #2D example
    uc_F2 = ng.fileiobase.uc_from_udic(udic, ndim-1)
    ppm_F2 = pd.Series(uc_F2.ppm_scale())

    uc_F1 = ng.fileiobase.uc_from_udic(udic, ndim-2)
    ppm_F1 = pd.Series(uc_F1.ppm_scale())

    # filter selected window
    if window is not None:
        mask_F2 = (ppm_F2 >= window[3]) & (ppm_F2 <= window[2])
        mask_F1 = (ppm_F1 >= window[1]) & (ppm_F1 <= window[0])

        ppm_F2 = ppm_F2[mask_F2]
        ppm_F1 = ppm_F1[mask_F1]

        data = data.loc[mask_F1,mask_F2]
        print(data)
    return ppm_F2


data_path = '/Users/cyrilcharlier/Documents/Research/nmrData'
dataset = '8Carbios_ICCG_exchange'
expno = '1005'
procno = '1'

test = read_topspin_data(data_path, dataset, expno, procno, rowno=None, window=[140,130,12,11])
# print(test)