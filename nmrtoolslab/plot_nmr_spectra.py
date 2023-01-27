import pandas as pd
from pathlib import Path
import nmrglue as ng
import matplotlib.pyplot as plt

def read_topspin_data(
    data_path, 
    dataset, 
    expno, 
    procno,
    lowest_contour,
    contour_factor,
    n_contour,
    linewitdh_plot,
    plot_name=None, 
    plot_color=None, 
    spec_lim=None,

    ):

    # get complete data path
    full_path = Path(data_path, dataset, expno, 'pdata', procno)

    # read processed data
    dic, data = ng.bruker.read_pdata(str(full_path), 
        read_procs=True, 
        read_acqus=False, 
        #scale_data=True, 
        #all_components=False
        )

    # get universal dictionnary
    udic = ng.bruker.guess_udic(dic,data)

    # put data into dataframe
    data = pd.DataFrame(data)

    # exp dimensions
    ndim = udic['ndim']

    # filter selected window
    ppm_window = {}
    for k in range(ndim):
        ppm_window[k] = {}

    for k in range(ndim):
        uc_F = ng.fileiobase.uc_from_udic(udic, k)
        ppm = pd.Series(uc_F.ppm_scale())
    
        if spec_lim is not None:
            mask = (ppm >= min(spec_lim[k][0],spec_lim[k][1])) & (ppm <= max(spec_lim[k][0],spec_lim[k][1]))
            ppm = ppm[mask]
            ppm_window[k]['mask'] = mask
            ppm_window[k]['ppm'] = ppm

        label_info = [int(udic[k]['label'][:-1]),udic[k]['label'][-1]]             
        ppm_window[k]['label'] = label_info

    if ndim == 2:
        data = data.loc[ppm_window[0]['mask'],ppm_window[1]['mask']]
    if ndim == 1:
        data = data.loc[ppm_window[0]['mask']]

    # plot
    if plot_name is None:
        fig, ax = plt.subplots()
        plot_name = ax
    if plot_color is None:
        plot_color = 'red'
    
    if ndim == 2:
        cl = [lowest_contour * contour_factor ** x for x in range(n_contour)]    

        plot_name.contour(
            data,
            cl,
            colors = plot_color,
            linewidths=linewitdh_plot,
            extent=(
                max(ppm_window[1]['ppm']),
                min(ppm_window[1]['ppm']),
                max(ppm_window[0]['ppm']),
                min(ppm_window[0]['ppm'])
            )
        )

        plot_name.set_ylabel(r'$^{'+str(ppm_window[0]['label'][0])+'}$'+str(ppm_window[0]['label'][1])+ ' (ppm)')        
        plot_name.set_xlabel(r'$^{'+str(ppm_window[1]['label'][0])+'}$'+str(ppm_window[1]['label'][1])+ ' (ppm)')        

        plot_name.set_xlim(                                                      
            left    = max(ppm_window[1]['ppm']),                                      
            right   = min(ppm_window[1]['ppm'])                                       
            )
        plot_name.set_ylim(                                                      
            bottom    = max(ppm_window[0]['ppm']),                                      
            top   = min(ppm_window[0]['ppm'])                                       
            )

    if ndim == 1:
        plot_name.plot(
        ppm_window[0]['ppm'],
        data,
        color = plot_color,
        linewidth=linewitdh_plot
        )

        plot_name.set_xlim(                                                      
            left    = max(ppm_window[0]['ppm']),                                      
            right   = min(ppm_window[0]['ppm'])                                       
            )
        plot_name.set_xlabel(r'$^{'+str(ppm_window[0]['label'][0])+'}$'+str(ppm_window[0]['label'][1])+ ' (ppm)')        


data_path = '/Users/cyrilcharlier/Documents/Research/nmrData'
dataset = '8Carbios_ICCG_exchange'
expno = '1005'
procno = '1'

test_2d = read_topspin_data(
    data_path, 
    dataset, 
    expno, 
    procno,
    lowest_contour = 2e8,
    contour_factor = 1.4,
    n_contour = 30, 
    linewitdh_plot = 1.2,
    spec_lim=[[129.5,123.5],[9.78,9.37]],
    plot_color= 'blue'
    )

# test_1d = read_topspin_data(
#     data_path, 
#     dataset, 
#     expno, 
#     procno, 
#     lowest_contour = None,
#     contour_factor = None,
#     n_contour = None, 
#     linewitdh_plot = 1.2,
#     # plot_name=None,
#     plot_color='blue',    
#     spec_lim=[[1,-0.4]]
#     )

plt.show()
