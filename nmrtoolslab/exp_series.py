import pandas as pd
import os
from pathlib import Path 
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Spectra_Series_Analysis(object):

    def __init__(self,dic_data, normalization=False, time_conversion=False, species_info=False, time_step=False):
        # initialize data
        self.data = dic_data
        self.params = pd.DataFrame()
        self.results = []
        self.integrals = pd.DataFrame()
        self.time_step = time_step

        ##
        self.initialize_data(self.data)
        self.load_data()

        self.normalized_concentration = pd.DataFrame(columns=['row_id','species','integral'])
        self.species_info = species_info

        if normalization is True:
            self.concentration_normalization()

        if time_conversion is True:
            self.time_conversion()

    def initialize_data(self,dic):
        row_to_append = []

        file_list = []
        
        if Path(dic['path']).exists() is False:
            raise ValueError('This path do not exist: {}'.format(Path(dic['path'])))  
        else:
            for file in os.listdir(dic['path']):
                if file.endswith('_fit.txt'):
                    file_list.append(file)

            # get file name for each species
            for sp in dic['species']:
                file = list(filter(lambda x: sp in x, file_list))
                if len(file) == 1:
                    # if information need to be added to the df...do it here and add name to the column names of the panda
                    row_to_append.append([id,sp,dic['path'],file[0]])
                else: 
                    raise ValueError('Several files for the same species: {}'.format(sp))

        self.params = pd.concat([self.params, pd.DataFrame(row_to_append)], axis=1)
        self.params.columns = ['name','species','path','file_name']

    def load_data(self):
        for line in range(len(self.params)):
            try: 
                data_path = Path(self.params.iloc[line].path,self.params.iloc[line].file_name)
            except:
                raise ValueError('The data do not exist')
            self.results.append({'raw data':pd.read_csv(data_path,sep='\t')}.copy())        
        self.results = dict(zip(self.params.species.tolist(),self.results)) #contains all the data 

        integrals = []
        for id in self.results.keys():
            for line in range(len(self.results[id]['raw data'])):
                integrals.append([id,
                    self.results[id]['raw data'].row_id.iloc[line],
                    self.results[id]['raw data'].integral.iloc[line]
                    ])
        self.integrals = pd.concat([self.integrals,pd.DataFrame(integrals,columns=['species','row_id','integral'])],axis=1)

    def concentration_normalization(self):
        conc = []
        for index, row in self.integrals.iterrows():
            integral = row['integral']

            TSP_data = self.integrals[self.integrals.row_id == row['row_id'] ]
            TSP_integral = float(TSP_data[TSP_data.species=='TSP'].integral)

            n_protons = self.species_info[row['species']]['n_protons']

            conc.append(((integral/TSP_integral)*9/n_protons)*1.8)
        self.integrals['concentration'] = conc

    def time_conversion(self):
        self.integrals['time'] = self.integrals['row_id']*self.time_step/3600

    def plot_single_dataset(self,rawdata=True, species=False, multiple_plot=True, concentration=False,time=False):
        
        if species:
            sp_list = species
        else:
            sp_list = self.params.species.tolist()

        if multiple_plot is True:
            n_row = len(sp_list)
            fig_full = make_subplots(rows=n_row, cols=1)
            fig_full.update_layout(autosize=False, width=900, height=len(sp_list)*300)

        if multiple_plot is False:
            fig_full = make_subplots(rows=1, cols=1)
            fig_full.update_layout(autosize=False, width=900, height=300)

        for s in range(len(sp_list)):
            sp = sp_list[s]

            if time is False:
                x_plot = self.integrals[self.integrals.species==sp].row_id
            else:
                x_plot = self.integrals[self.integrals.species==sp].time

            if concentration is True:
                y_plot = self.integrals[self.integrals.species==sp].concentration
            else:
                y_plot = self.integrals[self.integrals.species==sp].integral

            fig_exp = go.Scatter(x=x_plot, y=y_plot, mode='markers', name=sp)
            fig_full.add_trace(fig_exp, row=s+1 if multiple_plot is True else 1, col=1)

        fig_full.update_yaxes(exponentformat="power", showexponent="last")
        # fig_full['layout']['xaxis']['title']='id'
        # fig_full['layout']['yaxis']['title']='intensity'
        
        return fig_full