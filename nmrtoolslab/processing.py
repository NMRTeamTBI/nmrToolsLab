from pathlib import Path
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from pybaselines import Baseline
# baseline from
# https://pybaselines.readthedocs.io/en/latest/examples/index.html#spline-baseline-examples

def read_topspin_data(data_path, dataset, expno, procno):
    """
    Loading bruker NMR data

    Parameters
    __________
    data_path : path
        path to the data folder.
    dataset : folder 
        name of the data folder
    expno : float
        Experimental number
    procno : float
        Process number

    Return
    __________
    data : 2D ndarray
        Array NMR data
    dic : dictionnary   
        dictionnary
: universal dictionnary
        universal dictionnary
    """

    # get complete data path
    full_path = Path(data_path, dataset, str(expno), 'pdata', str(procno))

    # read processed data
    dic, data = ng.bruker.read_pdata(str(full_path), 
        read_procs=True, 
        read_acqus=False, 
        scale_data=False, 
        all_components=False
        )

    # get universal dictionnary
    udic = ng.bruker.guess_udic(dic,data)

    return data, dic, udic

def write_topspin_pdata(data_path, dataset, expno, procno, data, dic):
   
    # get complete data path
    full_path = Path(data_path, dataset, str(expno))

    ng.fileio.bruker.write_pdata(
        full_path,
        dic,
        data,
        overwrite=True,
        pdata_folder=procno,       
        write_procs=True
        )

def shifting_sum(data, udic, n_spec_wdw = 16):
    """
    Adding subsequent 1Ds from a pseudo-2D spectrum

    Parameters
    __________
    data : 2D ndarray 
        Array pseudo-2D NMR data.
    udic : universal dictionnary 
        Universal dictionnary 
    n_spec_wdw : float, optionnal
        Number of spectra added. Default is 16

    Return
    __________
    data_sum : 2D ndarray
        Array of co-added NMR data. Last rows are only zeros

    """

    data_sum = np.zeros([udic[0]['size'],udic[1]['size']])
    for i in range(udic[0]['size']-n_spec_wdw):
        selected_data = data[i:n_spec_wdw+i,:]
        sum_selected_data = np.sum(selected_data,axis=0)
        data_sum[i,:] = sum_selected_data

    return data_sum

class baseline_correction(object):
    
    def __init__(self,data, spec_wdw,spec_list=False, base_type=False,poly_order=False):
        self.data       = data
        self.spec_wdw   = spec_wdw
        self.spec_list  = spec_list
        self.base_type  = base_type
        self.poly_order = poly_order
        self.data_bl    = False
        self.baseline   = {}

        self.initialize_spec_list()

        self.n_dim = self.data.ndim

        if base_type == 'lin':
            self.poly_order = 1


        self.polynomial_baseline_correction()

    def initialize_spec_list(self):
        if self.spec_list is False:   
            self.spec_list = np.arange(0,self.data.shape[0])
        elif type(self.spec_list) is list :
            pass
        elif type(self.spec_list) is int:
            self.spec_list = [self.spec_list]

    def polynomial_baseline_correction(self):
        """
        Loading bruker NMR data

        Parameters
        __________
        data : 1D or 2D ndarray 
            Array pseudo-2D NMR data.

        Return
        __________
        data_sum : 2D ndarray
            Array of base line corrected NMR data.

        """
        self.data_bl = self.data.copy()

        x = np.arange(min(self.spec_wdw),max(self.spec_wdw),1)
        baseline_fitter = Baseline(x_data=x)
        self.baseline['x'] = x

        if self.n_dim == 2:
            for s in self.spec_list:
                data_region = self.data[s,min(self.spec_wdw):max(self.spec_wdw)]

                baseline = baseline_fitter.imodpoly(data_region, poly_order=self.poly_order, num_std=0.7)[0]
                self.baseline[s] = baseline
                self.data_bl[s,min(self.spec_wdw):max(self.spec_wdw)] = data_region - baseline
        if self.n_dim == 1:
                data_region = self.data[min(self.spec_wdw):max(self.spec_wdw)]

                baseline = baseline_fitter.imodpoly(data_region, poly_order=self.poly_order, num_std=0.7)[0]
                self.baseline[1] = baseline
                self.data_bl[min(self.spec_wdw):max(self.spec_wdw)] = data_region - baseline

    def visualize_baseline_correction(self,s, baseline_correction=False):
        # ,show_all=False
        fig, ax = plt.subplots(1,1, figsize=(10,4))
        x_plot = self.baseline['x']

        if self.n_dim == 2:
            ax.plot(x_plot,self.data[s,min(self.spec_wdw):max(self.spec_wdw)],c='b',label='raw data')
            if baseline_correction is True:
                ax.plot(x_plot,self.baseline[s],c='g',label='baseline')
                ax.plot(x_plot,self.data_bl[s,min(self.spec_wdw):max(self.spec_wdw)],c='r',label='corrected')
        if self.n_dim == 1:
            ax.plot(x_plot,self.data[min(self.spec_wdw):max(self.spec_wdw)],c='b',label='raw data')
            if baseline_correction is True:
                ax.plot(x_plot,self.baseline[1],c='g',label='baseline')
                ax.plot(x_plot,self.data_bl[min(self.spec_wdw):max(self.spec_wdw)],c='r',label='corrected')


        ax.legend()
        ax.grid(axis = 'y')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('pts')

# def linear_baseline_correction(data,nl,spec_list=False):
#     """
#     Loading bruker NMR data

#     Parameters
#     __________
#     data : 1D or 2D ndarray 
#         Array pseudo-2D NMR data.
#     nl: list
#         List of baseline nodes.

#     Return
#     __________
#     data_sum : 2D ndarray
#         Array of base line corrected NMR data.

#     """

#     if spec_list is False:     
#         data_bl = ng.process.proc_bl.base(data,nl)
#     else:
#         data_bl = data
#         if type(spec_list) is int:
#             s = spec_list
#             data_bl[s] = ng.process.proc_bl.base(data[s],nl)

#         if type(spec_list) is list and len(spec_list) == 1:
#             s = spec_list[0]
#             data_bl[s] = ng.process.proc_bl.base(data[s],nl)

#         if type(spec_list) is list and len(spec_list) != 1:
#             s = spec_list
#             data_bl[s,:] = ng.process.proc_bl.base(data[s,:],nl)

#     return data_bl





