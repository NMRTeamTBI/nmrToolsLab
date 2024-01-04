import pandas as pd
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
def plot_nmr_spectra(
    data,
    udic,
    lowest_contour  =   False,
    contour_factor  =   False,
    n_contour       =   False,
    linewitdh_plot  =   False,
    plot_name       =   None, 
    plot_color      =   None, 
    spec_lim        =   None,
    rotate          =   False,
    pseudo2D        =   False,
    detla_time      =   False,
    dot_plot        =   False,
    marker_size     =   False
    ):
    #
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

        if pseudo2D is True:
            if k == 0:
                y_0 =  min(spec_lim[k][0],spec_lim[k][1])
                y_1 =  max(spec_lim[k][0],spec_lim[k][1])
                time_scale = np.arange(0,len(ppm)+1,1)*detla_time
            if k == 1: #31P direct dimension
                mask = (ppm >= min(spec_lim[k][0],spec_lim[k][1])) & (ppm <= max(spec_lim[k][0],spec_lim[k][1]))
                ppm = ppm[mask]
                ppm_window[k]['mask'] = mask
                ppm_window[k]['ppm'] = ppm
        else:
            if spec_lim is not None:
                mask = (ppm >= min(spec_lim[k][0],spec_lim[k][1])) & (ppm <= max(spec_lim[k][0],spec_lim[k][1]))
                ppm = ppm[mask]
                ppm_window[k]['mask'] = mask
                ppm_window[k]['ppm'] = ppm
            else:
                ppm_window[k]['ppm'] = ppm

        label_info = [int(udic[k]['label'][:-1]),udic[k]['label'][-1]]             
        ppm_window[k]['label'] = label_info

    if pseudo2D is True:
        data = data.loc[y_0:y_1,ppm_window[1]['mask']] 
    else:
        if spec_lim is not None:
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
                max(ppm_window[0]['ppm']) if pseudo2D is False else time_scale[y_0],
                min(ppm_window[0]['ppm']) if pseudo2D is False else time_scale[y_1]
            )
        )
        if pseudo2D is True:
            plot_name.set_ylabel('time')
        else:
            plot_name.set_ylabel(r'$^{'+str(ppm_window[0]['label'][0])+'}$'+str(ppm_window[0]['label'][1])+ ' (ppm)')        
        plot_name.set_xlabel(r'$^{'+str(ppm_window[1]['label'][0])+'}$'+str(ppm_window[1]['label'][1])+ ' (ppm)')        

        plot_name.set_xlim(                                                      
            left    = max(ppm_window[1]['ppm']),                                      
            right   = min(ppm_window[1]['ppm'])                                       
            )
        plot_name.set_ylim(                                                      
            bottom    =     max(ppm_window[0]['ppm']) if pseudo2D is False else time_scale[y_0],                                      
            top       =     min(ppm_window[0]['ppm'])  if pseudo2D is False else time_scale[y_1]                                      
            )

    if ndim == 1:
        if rotate == False:
            plot_name.plot(
            ppm_window[0]['ppm'],
            data,
            color = plot_color,
            linewidth= 0 if dot_plot is True else linewitdh_plot, 
            marker = 'o' if dot_plot is True else None,
            ms = marker_size if dot_plot is True else None,
            )

            plot_name.set_xlim(                                                      
                left    = max(ppm_window[0]['ppm']),                                      
                right   = min(ppm_window[0]['ppm'])                                       
                )
            plot_name.set_xlabel(r'$^{'+str(ppm_window[0]['label'][0])+'}$'+str(ppm_window[0]['label'][1])+ ' (ppm)')
        if rotate == True:
            plot_name.plot(
            data,
            ppm_window[0]['ppm'],
            color = plot_color,
            linewidth=linewitdh_plot
            )

            plot_name.set_ylim(                                                      
                bottom    = max(ppm_window[0]['ppm']),                                      
                top   = min(ppm_window[0]['ppm'])                                       
                )
            plot_name.set_ylabel(r'$^{'+str(ppm_window[0]['label'][0])+'}$'+str(ppm_window[0]['label'][1])+ ' (ppm)')