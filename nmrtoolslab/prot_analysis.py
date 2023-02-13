import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
from scipy.optimize import curve_fit, minimize
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class nmrpipe_file(object):
    def __init__(self,path,file_name,time_step,Exp=False):
        self.path = path
        self.file_name = file_name 
        self.time_step = time_step

        self.experimental_data = False

        self.Exp = Exp
        if self.Exp is True:
            self.decay_curves()
    def load_series_data(self):
        #reads nlin.tab file for a series of experiments
        # data path
        full_path = Path(self.path, self.file_name)

        # reads in data
        data = pd.read_table(full_path,header=6,sep='\s+')

        # clean and organize the data
        data =  data.drop([0,1,2])

        # get column names
        column_names   =  data.columns.values.tolist()[1:]

        # fix columns
        data =  data.iloc[:, :-1]
        data.columns = column_names
        
        return data 

    def decay_curves(self):
        data = self.load_series_data()

        # columns with intensities
        intensity_column_names = [s for s in data.columns.values.tolist()[1:] if "Z_" in s]
        #intensity_column_idx = [i for i, j in enumerate(data.columns.values.tolist()[1:]) if j in intensity_column_names]        

        # get list of residue   
        reslist = data.loc[:,'ASS'].values.tolist()

        #consolidated data
        consolidated_data = []
        for r in reslist:
            res_data = data[data.loc[:,'ASS']==r].loc[:,intensity_column_names].iloc[0]
            for t in range(len(intensity_column_names)):
                if self.time_step is False:
                    pass
                else:
                    time = t * self.time_step

                consolidated_data.append([int(r),intensity_column_names[t],time,float(res_data.loc[intensity_column_names[t]])])
                # exit()
        # print(data.loc[:,intensity_column_names].head())
        consolidated_data = pd.DataFrame(consolidated_data,columns=['ass','spec','delay','norm_int'])
        self.experimental_data = consolidated_data
        
class sparky_list(object):

    def __init__(self,path,list_name,peak_label,dimension):
        
        self.path = path
        self.list_name = list_name
        self.peak_label = peak_label
        self.dimension = dimension

        self.peak_list = False

        if self.dimension == '2D':
            self.read_peaklist_2D()

    def read_peaklist_2D(self):

        full_path = Path(self.path, str(self.list_name)+'.list')
        
        self.peak_list = pd.read_table(
        full_path,
        sep='\s+'
        )

        # Clear values with ?
        self.peak_list = self.peak_list[~self.peak_list.Assignment.str.contains("?", regex=False)]
        self.peak_list = self.peak_list[~self.peak_list.Assignment.str.contains("X", regex=False)]

        # Estimate the number of residues
        n_res = self.peak_list.shape[0]

        # Obtain the residue list
        amino_acid_assignment = [int("".join(self.split(self.peak_list.Assignment.iloc[i].replace(self.peak_label,''))[1:])) for i in range(n_res) ]

        # Obtain the residue type for each residue
        amino_acid_type = ["".join(self.split(self.peak_list.Assignment.iloc[i])[0]) for i in range(n_res) ]

        self.peak_list['Assignment']  = amino_acid_assignment
        self.peak_list['Res_Type']    = amino_acid_type

        exp_dimensions = self.split(self.peak_label)

        self.peak_list.rename(columns={
            'Assignment':'Ass',
            'Res_Type':'res_type',
            'w1':'w_'+str(exp_dimensions[0]),
            'w2':'w_'+str(exp_dimensions[2]),

            },
              inplace=True)

        self.peak_list = self.peak_list.reset_index(drop=True).set_index(self.peak_list['Ass'])

        if 'Data' in self.peak_list.columns:
            self.peak_list.drop(['Height'],axis=1,inplace=True)
            self.peak_list.rename(columns={'Data':'Height'},inplace=True)

    def split(self,word): 
        return [char for char in word] 

class csp_calculation(object):

    def __init__(self, peak_list_1, peak_list_2):
        self.peak_list_1 = peak_list_1
        self.peak_list_2 = peak_list_2

        self.gamma = {
            'H' :  1.0,                #'H1' 
            'C' :  0.251449530,        #'C13'
            'N' :  0.101329118,        #'N15'
            'P' :  0.404808636,        #'P31'
            'F' :  0.940746805,        #'F19'
            }

        self.csp_values = False
        self.csp_calculation()

    def split(self,word): 
        return [char for char in word] 

    def csp_calculation(self):
        """Calculate Detla_omegas and CSP between two peak lists.
        
        Returns:
            dataframe: ass, dw_w1, dw_W2, csp
        """

        # read both column names to obtain nuclei
        self.exp_dim_1 = list(self.peak_list_1.keys())[1:3]
        self.exp_dim_2 = list(self.peak_list_2.keys())[1:3]
        
        # Check that both list contains the same nuclei
        if self.exp_dim_1 == self.exp_dim_2:
            pass
        else:
            print('hello')
        # get gamma value according to the nucleus
        if self.exp_dim_1[0] == 'w_N':
            alpha = self.gamma['N']
        if self.exp_dim_1[0] == 'w_C':
            alpha = self.gamma['C']

        # select residues that are present in both peak list
        common_peak_list = set(list(self.peak_list_1.Ass)).intersection(list(self.peak_list_2.Ass))

        peak_list_1_sel = self.peak_list_1.loc[self.peak_list_1['Ass'].isin(common_peak_list)]
        peak_list_2_sel = self.peak_list_2.loc[self.peak_list_2['Ass'].isin(common_peak_list)]

        delta_w1 = peak_list_1_sel.loc[:,self.exp_dim_1[0]]-peak_list_2_sel.loc[:,self.exp_dim_1[0]]
        delta_w2 = peak_list_1_sel.loc[:,self.exp_dim_1[1]]-peak_list_2_sel.loc[:,self.exp_dim_1[1]]

        CSP = np.sqrt((delta_w2)**2+alpha*(delta_w1)**2)
        
        self.csp_values = pd.DataFrame()
        self.csp_values['delta_'+str(self.exp_dim_1[0])] = delta_w1
        self.csp_values['dela_'+str(self.exp_dim_1[1])] = delta_w2
        self.csp_values['csp'] = CSP
        self.csp_values['res_type'] = peak_list_1_sel.res_type

    def plot(self, title=False):
        """Plot experimental CSP
        Args:

        Returns: 
            go.Figure: plotly figure
        """
        
        fig = go.Figure(data=go.Bar(
            x=self.csp_values.index,
            y=self.csp_values.csp
            ))
        fig.update_layout(
            title="Combined CSP {%s,%s}" % (self.exp_dim_1[0],self.exp_dim_1[1]) if title is False else title,
            xaxis_title="residue number",
            yaxis_title="CSP",
            legend_title="Legend Title",
            )
        fig.update_layout(xaxis_range=[min(self.csp_values.index.values)-2,max(self.csp_values.index.values)+2])

        return fig

class data_consolidation(object):
    """This class 

    Args: (dic): input data as dictionnary that contains the data
    """
    def __init__(self,data,dim_data,data_type):
        self.data = data
        self.dim_data = dim_data
        self.data_type = data_type
        self.res_list = []
          

        self.consolidated_data = False

        # if data_type == 'pH':
        if data_type == 'time':
             self.data_initialisaion_time()
        else:
            self.data_initialisaion()

    def split(self,word): 
        return [char for char in word] 

    def data_initialisaion(self):
        # Load a series of peak lists 
        for i in self.data.keys():
            pk_list = sparky_list(
                self.data[i]['path'],
                self.data[i]['exp'],
                self.data[i]['peak_label'],
                self.dim_data
                )
            self.data[i]['data'] = pk_list.peak_list
            self.res_list.append(self.data[i]['data'].Ass.tolist())
        self.res_list = list(set(list(itertools.chain(*self.res_list))))

        general_table = []
        for res in self.res_list:
            for i in self.data.keys():
                dim = list(self.data[i]['data'][self.data[i]['data'].Ass==res])[1:3]
                for d in dim :
                    try:
                        
                        peak_info = list(self.data[i]['data'][self.data[i]['data'].Ass==res].loc[:,['Ass',d,'res_type']].values[0])
                        peak_info.insert(1,self.data[i][self.data_type])
                        peak_info.insert(2,self.split(d)[2])
                        general_table.append(peak_info)
                    except:
                        pass
        self.consolidated_data = pd.DataFrame(general_table,columns=['ass',self.data_type,'nucleus','shift','res_type'])

    def data_initialisaion_time(self):
        # Load a series of peak lists 
        for i in self.data.keys():
            pk_list = sparky_list(
                self.data[i]['path'],
                self.data[i]['exp'],
                self.data[i]['peak_label'],
                self.dim_data
                )
            self.data[i]['data'] = pk_list.peak_list
            self.res_list.append(self.data[i]['data'].Ass.tolist())
        self.res_list = list(set(list(itertools.chain(*self.res_list))))

        general_table = []
        for res in self.res_list:
            for i in self.data.keys():
                try:
                    peak_info = list(self.data[i]['data'][self.data[i]['data'].Ass==res].loc[:,['Ass','Height','res_type']].values[0])
                    peak_info.insert(1,self.data[i][self.data_type])
                    # peak_info.insert(2,self.split(d)[2])
                    general_table.append(peak_info)
                except:
                    pass

        self.consolidated_data = pd.DataFrame(general_table,columns=['ass',self.data_type,'Height','res_type'])

class time_fitting(object):
    def __init__(self,data,res, normalized=False):
        self.data = data 
        self.res = res
        self.normalized = normalized

        self.params = []

        self.intensities = []
        self.time_pts = []

        self.data_selection()


    def data_selection(self):
        selected_data = self.data[self.data.ass==self.res]

        self.time_pts =  selected_data.loc[:,'time']
        
        if self.normalized is True:
            self.intensities = selected_data.loc[:,'Height']/selected_data.loc[:,'Height'].iloc[0]
        else:
            self.intensities = selected_data.loc[:,'Height']

    def exp_model(self,a,b,c,t):
        return a * np.exp(-b * t) +c

    def fit(self):

        popt, pcov = curve_fit(lambda t, a, b, c: self.exp_model(a,b,c,t), self.time_pts, self.intensities)
        self.params = popt

    def plot(self, fit: bool = False):
        
        fig = plt.figure()

        #Experimental data
        plt.plot(
            self.time_pts,
            self.intensities, 
            ls='none',
            marker='o',
            color="b"
            )

        #Fitted Curve
        if fit:
            time_sim = np.linspace(0,max(self.time_pts)+1,100)    
            simulated_curve = self.exp_model(self.params[0],self.params[1],self.params[2],time_sim)

            plt.plot(
                time_sim, 
                simulated_curve, 
                ls='-', 
                color="r"
                )
        plt.ylabel(r'I/I$_{0}$')
        plt.xlabel('time (hours)')
        plt.title(self.res)
        plt.ylim(0,1.3)

        return fig



class pKA_fitting(object):

    def __init__(self,data,res, nuclei, model,new_params=False):
        self.data = data        
        self.res = res
        self.fit_res = {}
        self.nuclei = nuclei
        self.model = model
        self.new_params = new_params
        self.params = False

        self.data_selection()

    def data_selection(self):
        #select data to fit
        self.selected_data = self.data[self.data.ass.isin(self.res)]
        # self.selected_data = self.data[self.data.ass == self.res]

        self.selected_data = self.selected_data[self.selected_data.nucleus.isin(self.nuclei)]
        self.selected_data.reset_index(inplace=True,drop=True)

        self.res_list = list(self.selected_data.loc[:,'ass'].unique())
        self.nuclei_list = list(self.selected_data.loc[:,'nucleus'].unique())

    def build_models(self):
        if self.model == 1:
            self.model_name = 'model_1pKa'
            self.description = 'equation Henderson Hasselbach with 1 pKa' 
            self.params_name = ['pKa','d_low','d_high']
            self.params_type = ['pKa','shift','shift']

        if self.model == 2:
            self.model_name = 'model_2pKa'
            self.description = 'equation Henderson Hasselbach with 2 pKa' 
            self.params_name = ['d_low','d_high_1','pKa_1','d_high_2','pKa_2']
            self.params_type = ['shift','shift','pKa','shift','pKa']

    def build_params(self):
        #add option is user set params

        base_params = {   
            'pKa': {'ini':4, 'lb': 2, 'ub': 10},
            'shift':{
                'H':{'ini':7, 'lb': 2, 'ub': 12},
                'N':{'ini':120, 'lb': 100, 'ub': 140},
                'C':{'ini':140, 'lb': 130, 'ub': 150},
            }
        }


        tmp_params = []

        pKa_indexes = [i for i, x in enumerate(self.params_type) if x == 'pKa']
        for i in pKa_indexes:
            tmp_params.append([
                self.params_name[i],
                self.params_type[i],
                '',
                '',
                base_params[self.params_type[i]]['ini'],
                base_params[self.params_type[i]]['lb'],
                base_params[self.params_type[i]]['ub']
                ])

        for res in self.res_list:

            for nucleus in self.nuclei_list:
                shift_indexes = [i for i, x in enumerate(self.params_type) if x == 'shift']
                for s in shift_indexes:
                    tmp_params.append([
                        self.params_name[s],
                        self.params_type[s],
                        res,
                        nucleus,
                        base_params[self.params_type[s]][nucleus]['ini'],
                        base_params[self.params_type[s]][nucleus]['lb'],
                        base_params[self.params_type[s]][nucleus]['ub']
                        ])
        
        self.params = pd.DataFrame(tmp_params,columns=['par','par_type','ass','nucleus','ini','lb','ub'])

        # add par index in model vector        
        par_list = []
        for par in self.params.par:
            par_list.append(self.params_name.index(par))
        self.params['model_idx'] = par_list 
    
        # add column to build model         
        self.params['par_idx'] = [i for i in range(len(self.params))]

    def update_params(self, new_params):
        if self.new_params:
            for i in self.new_params.keys():
                for k in self.new_params[i].keys():
                    self.params.loc[self.params.par==i,k] = self.new_params[i][k]

    def func(self,pH,params):
        #Henderson Hasselback with 1pkA
        if self.model == 1:
            d = params[1]+(params[2]-params[1])/(1+10**(params[0]-pH)) 
        #Henderson Hasselback with 2pkA        
        if self.model == 2:
            d = params[0]+(params[1]-params[0])/(1+10**(pH-params[2]))+(params[3]-params[0])/(1+10**(pH-params[4]))                 
        return d

    def unpack_fit_parameters(self,params):
        self.fit_par = {}

        
        #pKa independant of res and nucleus
        pKa_idx = self.params[self.params['par_type']=='pKa'].par_idx.values
        pKa_model_idx = self.params[self.params['par_idx'].isin(pKa_idx)][['par','model_idx','par_idx']]

        for res in self.res_list:
            nuc_dict = {}
            nuc_res =self.params[self.params.ass==res]
            for s in self.nuclei_list:
                nuc_params =nuc_res[nuc_res.nucleus==s]
                # print(nuc_params)
                test = [None]*len(self.params_name)
                for p in range(len(pKa_model_idx.model_idx.values)):
                    test[pKa_model_idx.model_idx.values[p]] = pKa_model_idx.par_idx.values[p]
                for k in range(len(nuc_params)):
                    test[nuc_params.model_idx.values[k]] = nuc_params.par_idx.values[k]

                test2 = []
                for t in test:
                    test2.append(params[t])
                nuc_dict[s] = test2  
            self.fit_par[res] = nuc_dict

    def initialize_model_params(self):
        #build model 
        self.build_models()
        #build parameter table
        self.build_params()
        #update parameters if required
        self.update_params(self.new_params)

    def simulate(self,params,df):
        self.simulate_shift = np.zeros(len(self.selected_data.index))
        self.unpack_fit_parameters(params)

        for i, row in self.selected_data.iterrows():
            sim = self.func(row.pH,self.fit_par[row.ass][row.nucleus])            
            self.simulate_shift[i] = float(sim)

    def fit_objective(self,params,df):
        self.simulate(params,df)
        rmsd = np.sqrt(np.mean((self.simulate_shift-df.loc[:,'shift'].values)**2))
        return rmsd

    @staticmethod
    def _linear_stats(res, ftol: float = 2.220446049250313e-09) -> list:
        """Calculate standard deviation on estimated parameters using linear statistics.

        Args:
            res (scipy.optimize.OptimizeResult): fit results
            ftol (float, optional): ftol of optimization. Defaults to 2.220446049250313e-09.

        Returns:
            list: standard deviations
        """

        npar = len(res.x)
        tmp_i = np.zeros(npar)
        standard_deviations = np.array([np.inf]*npar)

        for i in range(npar):
            tmp_i[i] = 1.0
            hess_inv_i = res.hess_inv(tmp_i)[i]
            sd_i = np.sqrt(max(1.0, res.fun) * ftol * hess_inv_i)
            tmp_i[i] = 0.0

            standard_deviations[i] = sd_i

        return standard_deviations

    def fit(self):

        self.initialize_model_params()
        
        #set initial values
        x0 = self.params['ini'].values.tolist()

        #set bounds
        bounds = list(zip(self.params['lb'],self.params['ub']))

        self.fit_res = minimize(
            self.fit_objective,
            x0 = x0,
            #method="L-BFGS-B",
            args = (self.selected_data),
            bounds=bounds
        )
        standard_deviations = self._linear_stats(self.fit_res)

        self.params['opt'] = self.fit_res.x
        self.params['opt_sd'] = standard_deviations
 
    def plot(self, res, fit: bool = False):
        
        if len(self.nuclei) == 2:
            fig_full = make_subplots(rows=1, cols=2)
        else :
            fig_full = make_subplots(rows=1, cols=1)

        
        data_res = self.selected_data[self.selected_data.ass == res]

        ## ---  First nucleus 
        #Experimental data
        data_res_nucleus = data_res[data_res.nucleus==self.nuclei[0]]
        fig_1 = go.Scatter(x=data_res_nucleus.loc[:,'pH'],y=data_res_nucleus.loc[:,'shift'], name='data: '+str(self.nuclei[0]), mode='markers',marker_color="#EF553B")
        fig_full.add_trace(fig_1, row=1, col=1)

        #Fitted Curve
        if fit:
            pH_all = np.linspace(min(data_res_nucleus.loc[:,'pH'])-0.2,max(data_res_nucleus.loc[:,'pH'])+0.2,100)    
            simulated_curve = self.func(pH_all,self.fit_par[res][self.nuclei[0]])
            fig_fit = go.Scatter(x=pH_all, y=simulated_curve, mode='lines', name='best fit', marker_color="#EF553B")
            fig_full.add_trace(fig_fit, row=1, col=1)

        ## ---  Second nucleus only if it exists
        #Experimental data
        if len(self.nuclei) == 2:
            data_res_nucleus = data_res[data_res.nucleus==self.nuclei[1]]
            fig_2 = go.Scatter(x=data_res_nucleus.loc[:,'pH'],y=data_res_nucleus.loc[:,'shift'],mode='markers',marker_color="blue")
            fig_full.add_trace(fig_2, row=1, col=2)

        #Fitted Curve
            if fit:
                pH_all = np.linspace(min(data_res_nucleus.loc[:,'pH'])-0.2,max(data_res_nucleus.loc[:,'pH'])+0.2,100)    
                simulated_curve = self.func(pH_all,self.fit_par[res][self.nuclei[1]])
                fig_fit = go.Scatter(x=pH_all, y=simulated_curve, mode='lines', name='best fit', marker_color="blue")
                fig_full.add_trace(fig_fit, row=1, col=2)


        # fig.update_layout(
        #     title="",
        #     xaxis_title="pH",
        #     yaxis_title="CC",
        #     legend_title="Legend Title",
        #     )

        return fig_full

