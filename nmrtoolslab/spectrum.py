import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import nmrtoolslab.processing_2 as proc


class Spectrum(object):
    """This class is responsible for most of the work on nmrdata:
        * data loading
    """

    def __init__(self, dataset : dict = None):
        # self.data = data 
        # self.udic = udic 
        # self.selected_data = False

        # set spectrum-related attributes
        self.data_path = dataset["data_path"]
        self.dataset = dataset["dataset"]
        self.expno = dataset["expno"]
        self.procno = dataset["procno"]
        # self.pseudo2D = False 

        # load NMR data with user selected window
        self.intensity, dic, self.udic = proc.read_topspin_data(self.data_path,self.dataset,self.expno,self.procno)
        
        # calculate ppms if no spec_lim is provided
        self.ppm_window = proc.get_ppm_list(self.udic)

        # #select data if needed
        if 'spec_lim' in dataset:        
            spec_lim = dataset['spec_lim'] #if 'spec_lim' in dataset else None
            self.reduce_spectral_window(spec_lim=spec_lim)
        else:
            spec_lim = None

    def reduce_spectral_window(self,spec_lim):
            
        # put data into dataframe
        self.intensity = pd.DataFrame(self.intensity)
        
        # create masks based on ppm values and select ppm windows
        for k in list(self.ppm_window.keys()):
            mask = (self.ppm_window[k]['ppm'] >= min(spec_lim[k][0],spec_lim[k][1])) & (self.ppm_window[k]['ppm'] <= max(spec_lim[k][0],spec_lim[k][1]))
            self.ppm_window[k]['mask'] = mask
            self.ppm_window[k]['ppm'] = self.ppm_window[k]['ppm'][mask]

        # select data based on ppm selection
        if len(list(self.ppm_window.keys())) == 2:
            self.intensity = self.intensity.loc[self.ppm_window[0]['mask'],self.ppm_window[1]['mask']] 
        if len(list(self.ppm_window.keys())) == 1:
            self.intensity = self.intensity.loc[self.ppm_window[0]['mask']] 

        # drop masks information
        for k in list(self.ppm_window.keys()):
            del self.ppm_window[k]['mask'] 

       
        # # check if the provides information about the pseudo2D
        # self.pseudo2D = dataset['pseudo2D'] if 'pseudo2D' in dataset else False
        # self.detla_time = dataset['delta_time'] if 'delta_time' in dataset else 1

        # if self.pseudo2D is True:
        #     self.time_scale = np.arange(0,len(self.ppm_window)+1,1)*self.detla_time
        # else: 
        #     self.time_scale = False

    def plot_matplotlib(self, plot : bool = False, rotate : bool = False, linewidth : float = None, lowest_contour : float = None, contour_factor : float = None, n_contour : float = None, color = None, marker = None, marker_size = None, intensity_offset = None, ppm_offset = None):
        """
        XX
        """
        # create label axis from universal dictionnary
        label_info = proc.experiment_label(self.udic)
        
        # check if plot exists otherwise create it
        if plot:
            plot_name = plot            
        else:
            fig, ax = plt.subplots()
            plot_name = ax
        
        # define plot color and linewidt
        plot_color = 'blue' if color is None else color
        linewidth = 0.5 if linewidth is None else linewidth

        # define experimental dimensions
        ndim = len(list(self.ppm_window))

        if ndim == 2:
            lowest_contour = 1e9 if lowest_contour is None else lowest_contour
            contour_factor = 1.5 if contour_factor is None  else contour_factor
            n_contour = 10 if n_contour is None  else n_contour

            cl = [lowest_contour * contour_factor ** x for x in range(n_contour)]    

            plot_name.contour(
                self.intensity,
                cl,
                colors = plot_color,
                linewidths=linewidth,
                extent=(max(self.ppm_window[1]['ppm']),min(self.ppm_window[1]['ppm']),max(self.ppm_window[0]['ppm']),min(self.ppm_window[0]['ppm']),)
            )
            plot_name.set_ylabel(r'$^{'+str(label_info[0][0])+'}$'+str(label_info[0][1])+ ' (ppm)')        
            plot_name.set_xlabel(r'$^{'+str(label_info[1][0])+'}$'+str(label_info[1][1])+ ' (ppm)')        

            plot_name.set_xlim(left = max(self.ppm_window[1]['ppm']), right = min(self.ppm_window[1]['ppm']))
            plot_name.set_ylim(bottom = max(self.ppm_window[0]['ppm']), top = min(self.ppm_window[0]['ppm']))
                    
        if ndim == 1:
            intensity = self.intensity if intensity_offset is None else self.intensity+intensity_offset
            ppm_scale = self.ppm_window[0]['ppm'] if ppm_offset is None else self.ppm_window[0]['ppm']+ppm_offset
           
            plot_name.plot(
                ppm_scale if rotate is False else intensity,
                intensity if rotate is False else ppm_scale,
                c = plot_color,
                lw = linewidth,
                marker = marker if marker else None,
                ms = marker_size if marker_size else None,
            )
            
            if rotate is False:
                plot_name.set_xlim(left = max(self.ppm_window[0]['ppm']), right = min(self.ppm_window[0]['ppm']))
                plot_name.set_xlabel(r'$^{'+str(label_info[0][0])+'}$'+str(label_info[0][1])+ ' (ppm)')
                plot_name.spines[['top','left','right']].set_visible(False)
                plot_name.tick_params(labelleft=False,left=False)

            else:
                plot_name.set_ylim(bottom = max(self.ppm_window[0]['ppm']), top = min(self.ppm_window[0]['ppm']))             
                plot_name.set_ylabel(r'$^{'+str(label_info[0][0])+'}$'+str(label_info[0][1])+ ' (ppm)')
    
    def plot_plotly(self, plot : bool = False, rotate : bool = False, linewidth : float = None, a : bool = False, lowest_contour : float = None, contour_factor : float = None, n_contour : float = None, color = None, marker = None, marker_size = None , plot_legend = None,  intensity_offset = None, ppm_offset = None):
 
        # create label axis from universal dictionnary
        label_info = proc.experiment_label(self.udic)

        # define experimental dimensions
        ndim = len(list(self.ppm_window))

        # define plot color and linewidt
        plot_color = 'blue' if color is None else color
        linewidth = 0.5 if linewidth is None else linewidth

        if plot is False:
            fig = make_subplots(rows=1, cols=1)
        else:
            fig = plot

        # only for plotly
        if isinstance(self.intensity, pd.DataFrame):
            intensity_2_plot = self.intensity.iloc[:,0].to_numpy()
        else:
            intensity_2_plot = self.intensity



        if ndim ==1:

            intensity = intensity_2_plot if intensity_offset is None else intensity_2_plot+intensity_offset
            ppm_scale = self.ppm_window[0]['ppm'] if ppm_offset is None else self.ppm_window[0]['ppm']+ppm_offset

            mode = 'markers' if marker else 'lines'
            fig_exp = go.Scatter(
                x=ppm_scale if rotate is False else intensity, 
                y=intensity if rotate is False else ppm_scale, 
                mode=mode, 
                name = None if plot_legend is None else plot_legend, 
                showlegend=False if plot_legend is None else True,
                marker_color=plot_color
                )
            fig.update_layout(xaxis=dict(title='chemical shift (ppm)'))
            fig.update_layout(yaxis=dict(showgrid=False,showline=False,showticklabels=False))

            fig.add_trace(fig_exp, row=1, col=1)
            fig.update_xaxes(autorange="reversed", ticks="outside")
            fig.update_yaxes(exponentformat="power", showexponent="last")
            fig.update_layout(plot_bgcolor="white", xaxis=dict(linecolor="black"), yaxis=dict(linecolor="black"))
            # fig.update_xaxes(autorange=False, range=[np.max(self.ppm), np.min(self.ppm)])
        fig.update_traces(line={'width': linewidth})

        return fig