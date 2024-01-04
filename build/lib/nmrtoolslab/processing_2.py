from pathlib import Path
import nmrglue as ng
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybaselines import Baseline
# baseline from
# https://pybaselines.readthedocs.io/en/latest/examples/index.html#spline-baseline-examples

def read_topspin_data(data_path, dataset, expno, procno, scale_data=True):
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
        scale_data=scale_data, 
        all_components=False
        )

    # get universal dictionnary
    udic = ng.bruker.guess_udic(dic,data)

    return data, dic, udic

def experiment_label(udic):
    label_info = {}
    for k in range(udic['ndim']):
        label_info[k] = [int(udic[k]['label'][:-1]),udic[k]['label'][-1]]             
    return label_info

def get_ppm_list(udic):
    
    # define dimensions of the experiment
    ndim = udic['ndim']

    # create empty nested dictionnary for the ppm window
    ppm_window = {}
    for k in range(ndim):
        ppm_window[k] = {}

        uc_F = ng.fileiobase.uc_from_udic(udic, k)
        ppm_window[k]['ppm'] = pd.Series(uc_F.ppm_scale())

    return ppm_window

def reduce_spectral_window(intensity,ppm_window,spec_lim):
        
    # put data into dataframe
    intensity = pd.DataFrame(intensity)
    
    # create masks based on ppm values and select ppm windows
    for k in list(ppm_window.keys()):
        mask = (ppm_window[k]['ppm'] >= min(spec_lim[k][0],spec_lim[k][1])) & (ppm_window[k]['ppm'] <= max(spec_lim[k][0],spec_lim[k][1]))
        ppm_window[k]['mask'] = mask
        ppm_window[k]['ppm'] = ppm_window[k]['ppm'][mask]

    # select data based on ppm selection
    if len(list(ppm_window.keys())) == 2:
        intensity = intensity.loc[ppm_window[0]['mask'],ppm_window[1]['mask']] 
    if len(list(ppm_window.keys())) == 1:
        intensity = intensity.loc[ppm_window[0]['mask']] 

    # drop masks information
    for k in list(ppm_window.keys()):
        del ppm_window[k]['mask'] 

    return intensity, ppm_window

# def shifting_sum(data, udic, n_spec_wdw = 16):
#     """
#     Adding subsequent 1Ds from a pseudo-2D spectrum

#     Parameters
#     __________
#     data : 2D ndarray 
#         Array pseudo-2D NMR data.
#     udic : universal dictionnary 
#         Universal dictionnary 
#     n_spec_wdw : float, optionnal
#         Number of spectra added. Default is 16

#     Return
#     __________
#     data_sum : 2D ndarray
#         Array of co-added NMR data. Last rows are only zeros

#     """

#     data_sum = np.zeros([udic[0]['size'],udic[1]['size']])
#     for i in range(udic[0]['size']-n_spec_wdw):
#         selected_data = data[i:n_spec_wdw+i,:]
#         sum_selected_data = np.sum(selected_data,axis=0)
#         data_sum[i,:] = sum_selected_data

#     return data_sum








class analysis_1d(object):
    def __init__(self,data,udic,spec_wdw=False):
        self.data       = data 
        self.udic       = udic
        self.spec_wdw   = spec_wdw
        self.sel_data   = False
        self.sel_ppm    = False
        self.ref_ppm    = False #Data calibrated with TSP @ 0 ppm
        self.data_selection_1d()

    def data_selection_1d(self):
        uc_F = ng.fileiobase.uc_from_udic(self.udic, 0)
        ppm = pd.Series(uc_F.ppm_scale())

        if self.spec_wdw is False:
            self.sel_data = self.data
            self.sel_ppm = ppm

        else:
            mask = (ppm >= min(self.spec_wdw[0],self.spec_wdw[1])) & (ppm <= max(self.spec_wdw[0],self.spec_wdw[1]))
            ppm = ppm[mask]
            self.sel_ppm = ppm
            self.sel_data = self.data[mask]

    def find_maximum_pts(self):
        max_idx             = list(self.sel_data).index(max(list(self.sel_data)))
        ppm_max, signal_max = self.sel_ppm[max_idx-3:max_idx+4], self.sel_data[max_idx-3:max_idx+4]
        coeff             = np.polyfit(ppm_max, signal_max, 3)
    
        ppm_fit = np.linspace(min(ppm_max), max(ppm_max), 100000)
        model = np.poly1d(coeff)
        sig_fit = model(ppm_fit)

        max_idx = np.where(sig_fit == sig_fit.max())
        local_max = ppm_fit[max_idx] ## SR value
        
        self.ref_ppm = self.sel_ppm-local_max

        plt.plot(self.sel_ppm,self.sel_data,ls='none',marker='o')
        plt.plot(self.ref_ppm,self.sel_data,ls='none',marker='o')

        plt.plot(ppm_max,signal_max,ls='none',marker='o')
        plt.plot(ppm_fit,sig_fit)
        plt.show()

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

    def visualize_baseline_correction(self,s=False, baseline_correction=False):
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







