import pandas as pd
from io import StringIO
import pynmrstar
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
from scipy.optimize import curve_fit, minimize
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os 
import re
from tqdm import tqdm


three_to_one = {
    'CYS': 'C',
    'ASP': 'D',
    'SEP': 'S',
    'SER': 'S',
    'GLN': 'Q',
    'LYS': 'K',
    'PRO': 'P',
    'THR': 'T',
    'PHE': 'F',
    'ALA': 'A',
    'HIS': 'H',
    'GLY': 'G',
    'ILE': 'I',
    'GLU': 'E',
    'LEU': 'L',
    'ARG': 'R',
    'TRP': 'W',
    'VAL': 'V',
    'ASN': 'N',
    'TYR': 'Y',
    'MET': 'M'
}

class import_bmrb(object):
    def __init__(self,bmrb_entry,nuclei,offset=False):
        self.bmrb_entry = bmrb_entry
        self.nuclei = nuclei
        self.offset = offset
        self.bmrb_df = False
        self.shifts = False
        self.shifts_to_sim = False

        self.import_bmrb()
        if nuclei in ['HN','NH']:
            self.bmrb_to_dataframe_HN()
            self.bmrb_to_simulate_HN()

    def import_bmrb(self):
    # Import a bmrb file using direcly the bmrb is entry
    # file_name =f'{bmrb_id}.bmrb.pkl'

        entry = pynmrstar.Entry.from_database(self.bmrb_entry)
        shifts = entry.get_loops_by_category("Atom_chem_shift")[0]
        shifts_csv = shifts.get_data_as_csv()
        self.bmrb_df = pd.read_csv(StringIO(shifts_csv))

        self.bmrb_df = self.bmrb_df[[
            '_Atom_chem_shift.Seq_ID',
            '_Atom_chem_shift.Atom_ID',
            '_Atom_chem_shift.Comp_ID',
            '_Atom_chem_shift.Atom_type',
            '_Atom_chem_shift.Val',
            '_Atom_chem_shift.Val_err',
            ]]
        self.bmrb_df.rename(columns={
            '_Atom_chem_shift.Seq_ID'       :'res',
            '_Atom_chem_shift.Atom_ID'      :'type',
            '_Atom_chem_shift.Comp_ID'      :'res_type',
            '_Atom_chem_shift.Atom_type'    :'atom',
            '_Atom_chem_shift.Val'          :'w',
            '_Atom_chem_shift.Val_err'      :'w_err',
        },inplace=True)

    def bmrb_to_dataframe_HN(self):
        _HN_shifts = self.bmrb_df[
        (self.bmrb_df['atom']=='H') &
        (self.bmrb_df['type']=='H')
        ]
        _N_shifts = self.bmrb_df[
        (self.bmrb_df['atom']=='N') &
        (self.bmrb_df['type']=='N')
        ]
        # for i in range(len(_PeakList)):
        #     _PeakList['res_type'].iloc[i] = three_to_one[_PeakList['res_type'].iloc[i]]
        res_list = list(set(_N_shifts['res'])&set(_HN_shifts['res']))
        All_Shifts = []
        for res in res_list:
            HShifts = _HN_shifts[_HN_shifts['res']==res]['w'].values[0]
            NShifts = _N_shifts[_N_shifts['res']==res]['w'].values[0]
            res_type = three_to_one[_N_shifts[_N_shifts['res']==res]['res_type'].values[0]]
            All_Shifts.append([res,res_type,NShifts,HShifts])
        self.shifts = pd.DataFrame(All_Shifts,columns=['Ass','ResType','15N','1H'])

    def bmrb_to_simulate_HN(self):
        if self.offset is False:
            self.offset = {'1H':0,'15N':0}
        
        tmp = []
        res_list = self.shifts.Ass
        for r in res_list:
            tmp.append([
                r,
                self.shifts[self.shifts.Ass==r].loc[:,'1H'].values[0]+self.offset['1H'],
                self.shifts[self.shifts.Ass==r].loc[:,'15N'].values[0]+self.offset['15N']
            ])

        self.shifts_to_sim = pd.DataFrame(tmp,columns=['res','1H','15N'])
    
class nmrpipe_file(object):

    def __init__(self,path,exp_num,file_name=False,time_step=False,Exp=False,info_T1rho_correction=False):
        self.path = path
        self.file_name = file_name 
        self.exp_num = exp_num
        self.time_step = time_step
        self.info_T1rho_correction = info_T1rho_correction

        self.experimental_data = False
        self.fit_params_res = False

        self.Exp = Exp

        if self.Exp is True:
            print('hello')
            self.decay_curves()

        if self.Exp == 'T1':
            self.relax_fit_parameters()
            self.relax_exp_data()
        if self.Exp == 'T1rho':
            self.relax_fit_parameters()
            # self.T1rho_correction()
        if self.Exp == 'NOE':
            self.NOE()

    def check_residue(self,res):
        # Check if the desired residue is in the residue list
        if res in self.fit_params_res.ass.tolist():
            pass
        else:
            print('##-- residue: '+str(res)+' is not in the assignment list --##')
            exit()

    def load_series_data(self):
        #reads nlin.tab file for a series of experiments
        # data path
        full_path = Path(self.path, str(self.exp_num), self.file_name)

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

    def NOE(self):
        data = self.load_series_data()

        # columns with intensities
        intensity_column_names = [s for s in data.columns.values.tolist()[1:] if "Z_" in s]
        intensity_column_names = ['HEIGHT','DHEIGHT']+intensity_column_names
        # get list of residue   
        reslist = data.loc[:,'ASS'].values.tolist()

        #consolidated data
        consolidated_data = []
        for r in reslist:
            res_data = data[data.loc[:,'ASS']==r].loc[:,intensity_column_names].iloc[0]
            consolidated_data.append([int(r),float(res_data.loc['Z_A0']),float(res_data.loc['Z_A1']),float(res_data.loc['HEIGHT']),float(res_data.loc['DHEIGHT']),float(res_data.loc['Z_A1'])])

        self.experimental_data = pd.DataFrame(consolidated_data,columns=['ass','Z_A0','Z_A1','HEIGHT','DHEIGHT','noe'])
        self.experimental_data.sort_values(by=['ass'],inplace=True)
        self.experimental_data['err_noe'] = np.sqrt(2*((self.experimental_data.DHEIGHT/self.experimental_data.HEIGHT)**2))

    def relax_fit_parameters(self):
        # list all gnu files containing fitting parameters
        all_files = os.listdir(self.path+'/'+str(self.exp_num)+'/gnu')
        consolidated_data = []
        for i in range(len(all_files)):
            with open(self.path+'/'+str(self.exp_num)+'/gnu/'+all_files[i]) as f:
                gnu_file = f.readlines()
                fit_res = [float(p) for p in gnu_file[3].split()[3::2][:-1]]
                consolidated_data.append(fit_res)
        self.fit_params_res = pd.DataFrame(consolidated_data,columns=['idx','ass','amp','err_amp','alpha','err_alpha'])
        self.fit_params_res.sort_values(by=['idx'],inplace=True)
        self.fit_params_res = self.fit_params_res.astype({"idx":"int","ass":"int"})
        #calculate rates
        self.fit_params_res['rate'] = 1000/self.fit_params_res.alpha
        self.fit_params_res['err_rate'] = self.fit_params_res.rate*(self.fit_params_res.err_alpha/self.fit_params_res.alpha)

    def relax_exp_data(self):
        # list all gnu files containing fitting parameters
        all_files = os.listdir(self.path+'/'+str(self.exp_num)+'/txt')
        tmp_res = []
        for i in range(len(all_files)):
            with open(self.path+'/'+str(self.exp_num)+'/txt/'+all_files[i]) as f:
                txt_file = f.readlines()
                idx = re.findall(r'\d+',(all_files[i]))[0]
                for k in range(len(txt_file)):
                    tmp_res.append([int(idx)]+[float(p) for p in txt_file[k].split()])
        self.experimental_data = pd.DataFrame(tmp_res,columns=['idx','delay','I','E_I'])

    def plot_relax_curve(self,res):
        self.check_residue(res)
        # def exp_function(self):
        fig, ax = plt.subplots()

        # return Amp*np.exp(-x/Alpha)

        params = self.fit_params_res[self.fit_params_res.ass==res]  
        data = self.experimental_data[self.experimental_data.idx==params.idx.values[0]]  

        time_fit = np.linspace(min(data.delay),max(data.delay),1000)
        I_fit = params.amp.values[0]*np.exp(-time_fit/params.alpha.values[0])

        ax.errorbar(data.delay, data.I, data.E_I,marker='o',ls='none',color='darkblue')
        ax.plot(time_fit,I_fit,color='darkblue')        
        ax.set_xlabel('delay (ms)')
        ax.set_ylabel(r'I/I$_0$')
        ax.text(0.05,0.1,str('alpha')+' = '+str(params.alpha.values[0])+' +/- '+str(params.err_alpha.values[0])+' $(s^{-1})$',transform=ax.transAxes)
        ax.text(0.05,0.05,str('amp')+' = '+str(params.amp.values[0])+' +/- '+str(params.err_amp.values[0]),transform=ax.transAxes)

        ax.set_ylim(0,1.1)
        plt.show()
        # print(data)

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

    def T1rho_correction(self):

        self.exp_num = self.info_T1rho_correction['t1_exp_num']
        t1_data = self.load_series_data()

        t1_data_sel = t1_data[['ASS','Y_PPM']]
        t1_data_sel = t1_data_sel.apply(pd.to_numeric)
        t1_data_sel['Offset'] = (t1_data_sel['Y_PPM'] - self.info_T1rho_correction['Carrier_Frequency'])/self.info_T1rho_correction['Observed_Frequency']
        t1_data_sel['Theta'] = np.arctan(t1_data_sel['Offset']/self.info_T1rho_correction['Offset'])

        A = 1.6*(np.sin(t1_data_sel['Theta'])**2)
        B = np.cos(t1_data_sel['Theta'])**2
        print(B)
        self.fit_params_res['rate_cor'] = (1000/self.fit_params_res['alpha']-A)/B
        self.fit_params_res['err_rate_cor'] = (1000/self.fit_params_res['alpha'])*(self.fit_params_res['err_alpha']/self.fit_params_res['alpha'])

        # print(self.fit_params_res.head())
      
class sparky_list(object):

    def __init__(self,path,list_name,peak_label,dimension):
        
        self.path = path
        self.list_name = list_name
        self.peak_label = peak_label
        self.dimension = dimension

        self.peak_list = False

        if self.dimension == '2D':
            self.read_peaklist_2D()
        if self.dimension in ['HNCA']:
            self.read_peaklist_3D()

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

        # print(len(list(self.peak_label)))
        if isinstance(self.peak_label, str) is True:
            #Obtain the residue list
            amino_acid_assignment = [int("".join(self.split(self.peak_list.Assignment.iloc[i].replace(self.peak_label,''))[1:])) for i in range(n_res) ]
            # Obtain the residue type for each residue
            amino_acid_type = ["".join(self.split(self.peak_list.Assignment.iloc[i])[0]) for i in range(n_res) ]
            exp_dimensions = self.peak_label.split('-')#self.split(self.peak_label,'-')


        if isinstance(self.peak_label, list) is True:
            amino_acid_assignment = []
            amino_acid_type = []
            peak_label = []
            for i in range(n_res):
                tmp = self.peak_list.Assignment.iloc[i]
                for pl in self.peak_label:
                    if tmp.find(pl) != -1 : 
                       amino_acid_assignment.append(int("".join(self.split(self.peak_list.Assignment.iloc[i].replace(pl,''))[1:])))
                       amino_acid_type.append("".join(self.split(self.peak_list.Assignment.iloc[i])[0]))
                       peak_label.append(pl)
                    #    self.peak_list.Assignment.iloc[i] = amino_acid_assignment
                    else:
                        pass
            self.peak_list['Peak_Label'] = peak_label
            exp_dimensions = [list(peak_label[0].split('-')[0])[0],list(peak_label[1].split('-')[1])[0]]

        self.peak_list['Assignment']  = amino_acid_assignment
        self.peak_list['Res_Type']    = amino_acid_type

        if 'Peak_Label' in self.peak_list.columns:
            self.peak_list.rename(columns={
                'Assignment':'Ass',
                'Res_Type':'res_type',
                'w1':'w_'+str(exp_dimensions[0]),
                'w2':'w_'+str(exp_dimensions[1]),
                'Peak_Label' : 'Peak_Label',
                },
                inplace=True)
        else:
            self.peak_list.rename(columns={
                'Assignment':'Ass',
                'Res_Type':'res_type',
                'w1':'w_'+str(exp_dimensions[0]),
                'w2':'w_'+str(exp_dimensions[1]),
                },
                inplace=True)

        self.peak_list = self.peak_list.reset_index(drop=True).set_index(self.peak_list['Ass'])
        if 'Data' in self.peak_list.columns:
            self.peak_list.drop(['Height'],axis=1,inplace=True)
            self.peak_list.rename(columns={'Data':'Height'},inplace=True)

    def read_peaklist_3D(self):
        full_path = Path(self.path, str(self.list_name)+'.list')

        self.peak_list = pd.read_table(
        full_path,
        sep='\s+'
        )

        # Clear values with ?
        self.peak_list = self.peak_list[~self.peak_list.Assignment.str.contains("?", regex=False)]
        self.peak_list = self.peak_list[~self.peak_list.Assignment.str.contains("X", regex=False)]
       

        if 'Data' in self.peak_list.columns:
            self.peak_list.drop([ 'Data'], axis=1,inplace=True)

        # PeakList.dropna(axis=1,inplace=True)

        
        if self.dimension == 'HNCA':
            self.peak_list = self.peak_list[~self.peak_list.Assignment.str.contains("CO", regex=False)]
            dim = 'CA'

        # Estimate the number of residues
        n_peaks = self.peak_list.shape[0]

        # Sort between i and i-1 signals
        sorted_peaks = []
        for i in range(n_peaks):

            pk_info = self.peak_list.Assignment.iloc[i].split('CA-')
            pk_C_info = re.split('(\d.*)',str(pk_info[0]))
            #Carbon ID
            pk_type_C = pk_C_info[0]
            pk_num_C = pk_C_info[1]
            if re.split('(\d.*)',str(pk_info[1])) == ['N-H']: #takes care of i peaks
                pass
            else:
                pk_HN_info = re.split('(\d.*)',str(pk_info[1]))
                pk_type_HN = pk_HN_info[0]
                pk_num_HN = pk_HN_info[1].split('N-H')[0]
            shift = self.peak_list.w1.iloc[i]
            if pk_num_C == pk_num_HN:
                pk_type ='i'
            else:
                pk_type ='i-1'

            sorted_peaks.append([pk_type_C,pk_num_C,pk_type,shift])
        
        # Format a proper output table
        df_sorted_peaks = pd.DataFrame(sorted_peaks,columns=['res_type','Ass','signal','w_C'])

        final_table = []
        res_list = df_sorted_peaks.loc[:,'Ass'].unique()
        for r in res_list:
            df_sel = df_sorted_peaks[df_sorted_peaks.loc[:,'Ass']==r]

            res_type = df_sel[df_sel.loc[:,'signal']=='i'].res_type.values

            if df_sel[df_sel.loc[:,'signal']=='i-1'].empty:
                w_Ci_1 = np.NaN
            else:
                res_type = df_sel[df_sel.loc[:,'signal']=='i-1'].res_type.values[0]
                w_Ci_1 = df_sel[df_sel.loc[:,'signal']=='i-1'].w_C.values[0]

            if df_sel[df_sel.loc[:,'signal']=='i'].empty:
                w_Ci = np.NaN
            else:
                res_type = df_sel[df_sel.loc[:,'signal']=='i'].res_type.values[0]
                w_Ci = df_sel[df_sel.loc[:,'signal']=='i'].w_C.values[0]

            # by default select the i signal otherwise use the i-1 of the following residue
            if w_Ci is np.NaN:
                w_C = w_Ci_1
            else:
                w_C = w_Ci

            final_table.append([int(r),w_C,res_type])#,w_Ci_1,w_Ci)

        self.peak_list = pd.DataFrame(final_table,columns=['Ass','w_'+str(dim),'res_type'])
        self.peak_list = self.peak_list.reset_index(drop=True).set_index(self.peak_list['Ass'])

    def split(self,word): 
        return [char for char in word] 

class csp_calculation(object):

    def __init__(self, set_data_1, set_data_2,w1,w2=False):
        self.set_data_1 =   set_data_1
        self.set_data_2 =   set_data_2
        self.w1         =   w1
        self.w2         =   w2

        self.gamma = {
            'H' :  1.0,                #'H1' 
            'C' :  0.251449530,        #'C13'
            'N' :  0.101329118,        #'N15'
            'P' :  0.404808636,        #'P31'
            'F' :  0.940746805,        #'F19'
            }

        self.dw_values = False
        self.csp_values = False

        if w2 is False:
            self.dw_calculation()
        else:
            self.csp_calculation()

    # def split(self,word): 
    #     return [char for char in word] 

    def dw_calculation(self):
        """Calculate Detla_omega two peak lists.
        
        Returns:
            dataframe: ass, dw_w1, res_type
        """
        common_reslist = list(set(self.set_data_1.ass.values.tolist()).intersection(self.set_data_2.ass.values.tolist()))

        results = []
        for i in range(len(common_reslist)):

            data_1 = self.set_data_1[self.set_data_1.ass==common_reslist[i]]
            data_2 = self.set_data_2[self.set_data_2.ass==common_reslist[i]]
    
            delta_w1 = data_1[data_1.nucleus==self.w1].iloc[0].loc['shift']-data_2[data_2.nucleus==self.w1].iloc[0].loc['shift']

            res_type = data_1[data_1.nucleus==self.w1].iloc[0].loc['res_type']
            results.append([common_reslist[i],delta_w1,res_type])
        self.dw_values = pd.DataFrame(results,columns=['ass','delta_w','res_type'])
        self.dw_values=self.dw_values.round(4)
        self.dw_values.sort_values(by=['ass'],inplace=True)

    def csp_calculation(self):
        """Calculate Detla_omegas and CSP between two peak lists.
        
        Returns:
            dataframe: ass, dw_w1, dw_W2, csp, res_type
        """
        
        alpha = self.gamma[self.w2]

        # get common peak list 
        common_reslist = list(set(self.set_data_1.ass.values.tolist()).intersection(self.set_data_2.ass.values.tolist()))
        results = []
        for i in range(len(common_reslist)):

            data_1 = self.set_data_1[self.set_data_1.ass==common_reslist[i]]
            data_2 = self.set_data_2[self.set_data_2.ass==common_reslist[i]]
    
            delta_w1 = data_1[data_1.nucleus==self.w1].iloc[0].loc['shift']-data_2[data_2.nucleus==self.w1].iloc[0].loc['shift']
            delta_w2 = data_1[data_1.nucleus==self.w2].iloc[0].loc['shift']-data_2[data_2.nucleus==self.w2].iloc[0].loc['shift']

            CSP = np.sqrt((delta_w1)**2+alpha*(delta_w2)**2)

            res_type = data_1[data_1.nucleus==self.w1].iloc[0].loc['res_type']
            results.append([common_reslist[i],delta_w1,delta_w2,CSP,res_type])
        self.csp_values = pd.DataFrame(results,columns=['ass','delta_w'+str(self.w1),'delta_w'+str(self.w2),'csp','res_type'])
        self.csp_values=self.csp_values.round(4)
        self.csp_values.sort_values(by=['ass'],inplace=True)

    def csp_plot(self, plot=False, index_residues=True):
        """Plot experimental CSP
        Args:

        Returns: 
            go.Figure: plotly figure
        """
        if plot is False:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(12, 4)
    
        if index_residues is True:
            ax.bar(self.csp_values.ass,self.csp_values.csp,color='darkblue')
            ax.set_xlim(left=min(self.csp_values.ass)-2,right=max(self.csp_values.ass)+2)
        else:
            x_ticks_position = np.arange(0,len(self.csp_values.csp),1)
            ax.bar(x_ticks_position,self.csp_values.csp,color='darkblue')
            ax.set_xticks(x_ticks_position,self.csp_values.ass)

        ax.set_xlabel("residue number")
        ax.set_ylabel('CSP (ppm)')
        return fig

    def dw_plot(self, plot=False, index_residues=True):
        """Plot experimental dw
        Args:

        Returns: 
            go.Figure: plotly figure
        """
        if plot is False:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(12, 4)
    
        if index_residues is True:
            ax.bar(self.dw_values.ass,abs(self.dw_values.delta_w),color=((self.dw_values.delta_w) > 0).map({True: 'darkblue',False: 'orange'}))
            ax.set_xlim(left=min(self.dw_values.ass)-2,right=max(self.dw_values.ass)+2)
        else:
            x_ticks_position = np.arange(0,len(self.dw_values.delta_w),1)
            ax.bar(x_ticks_position,self.dw_values.delta_w,color=((self.dw_values.delta_w) > 0).map({True: 'darkblue',False: 'orange'}))
            ax.set_xticks(x_ticks_position,self.dw_values.ass)
        
        ax.text(0.1,0.9,r'$\Delta\omega$ >0',color='darkblue',transform=ax.transAxes)
        ax.text(0.1,0.8,r'$\Delta\omega$ <0',color='orange',transform=ax.transAxes)

        ax.set_xlabel("residue number")
        ax.set_ylabel(r'$\Delta\omega$ (ppm)')

        return fig
    def csp_to_pdb(self,full_reslist,path,file_name):
        new_df = pd.DataFrame({'ass':full_reslist})
        new_df = new_df.merge(self.csp_values, on='ass', how='left')
        
        new_df[['ass','csp']].to_csv(os.path.join(path,str(file_name)+'.txt'),sep='\t',na_rep=0,index=False)

class data_consolidation(object):
    """This class 

    Args: (dic): input data as dictionnary that contains the data
    """
    def __init__(self,data,dim_data,data_type=False):
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

        if self.dim_data == '2D':
            col = 3
        else:
            col = 2

        if 'Peak_Label' not in self.data[i]['data'].columns:
            for res in self.res_list:
                for i in self.data.keys():
                    dim = list(self.data[i]['data'][self.data[i]['data'].Ass==res])[1:col]
                    for d in dim :
                        try:
                            cols = ['Ass',d,'res_type']
                            if 'Height' in pk_list.peak_list.columns:
                                cols.append('Height')
                            peak_info = list(self.data[i]['data'][self.data[i]['data'].Ass==res].loc[:,cols].values[0])

                            for k in range(len(self.data_type)):
                                peak_info.insert(1+k,self.data[i][self.data_type[k]])
                            peak_info.insert(k+2,self.split(d)[2])
                            general_table.append(peak_info)
                        except:
                            pass

        if 'Peak_Label' in self.data[i]['data'].columns:
            for i in self.data.keys():
                data = self.data[i]['data']
                for idx in range(len(data)):
                    for n in range(1,3):
                        nuc = list(data.columns)[n][2].split('-')[0]
                        "match nucleus and peak label"
                        label_list = data.iloc[idx,3].split('-')
                        label_check = [nuc in label_list[k] for k in range(2)]
                        tt = [i for i, x in enumerate(label_check) if x]
                        self.data[i][self.data_type[0]]
                        peak_info = [data.iloc[idx,0],self.data[i][self.data_type[0]],label_list[tt[0]],data.iloc[idx,n],data.iloc[idx,4]]
                        general_table.append(peak_info)

        col_names = ['ass']
        col_names.extend(list(self.data_type))
        cols = ['nucleus','shift','res_type']
        if 'Height' in pk_list.peak_list.columns:
            cols.append('Height')
        col_names.extend(cols)

        self.consolidated_data = pd.DataFrame(general_table,columns=col_names)
        self.consolidated_data.sort_values(by=['ass'],inplace=True)

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

class temperature_fitting(object):
    def __init__(self,data,res,nucleus):
        self.data = data 
        self.res = res
        self.nucleus = nucleus

        self.params = []
        self.err_params = []
        self.shift = []
        self.temperature_pts = []

        self.data_selection()

    def data_selection(self):
        selected_data = self.data[(self.data.ass==self.res)&(self.data.nucleus==self.nucleus)]
        self.temperature_pts =  selected_data.loc[:,'Temp'].tolist()
        self.temperature_pts = [i - 273 for i in self.temperature_pts] # Data in Celsius
        self.shift =  selected_data.loc[:,'shift'].tolist()

    def linear_model(sefl,a,b,T):
        return a + b*T

    def fit(self):

        popt, pcov = curve_fit(lambda T, a, b : self.linear_model(a,b,T), self.temperature_pts, self.shift)
        self.params = popt
        self.err_params = [pcov[0,0]**0.5,pcov[1,1]**0.5]

    def plot(self, fit: bool = False, plot:bool = False, color:bool = False):
        if plot is False:
            fig, ax = plt.subplots()
            plot = ax


        #Experimental data
        plot.plot(
            self.temperature_pts,
            self.shift, 
            ls='none',
            marker='o',
            color="b" if color is False else color
            )

        #Fitted Curve
        if fit:
            temperature_sim = np.linspace(min(self.temperature_pts),max(self.temperature_pts)+1,100)    
            simulated_curve = self.linear_model(self.params[0],self.params[1],temperature_sim)

            plot.plot(
                temperature_sim, 
                simulated_curve, 
                ls='-', 
                color="b" if color is False else color
                )
        plot.set_ylabel(r'I/I$_{0}$')
        plot.set_xlabel('temperature (K)')
        plot.set_title(self.res)

        if plot is False:
            return plot
        else:
            pass
    
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

    def __init__(self, data, model, fit_info=False,new_params=False,mc=False,n_mc=False,errors_mc=False):
        self.data = data     
        self.fit_res = {}
        self.model = model
        self.new_params = new_params
        self.fit_info = fit_info
        self.params = False
        self.rmsd = False
        self.params_mc = False
        self.mc=mc
        
        self.data_mc = False

        self.data_selection()

        if self.mc is True: # perform monte-carlo analysis
            # this includes error on both pH and chemical shifts
            self.errors_mc = errors_mc
            self.params_mc = pd.DataFrame()
            self.data_mc = pd.DataFrame()
            for n in tqdm(range(n_mc)):
                self.selected_data_mc = []
                
                self.prepare_monte_carlo_analysis()

                
                self.fitted_data = self.selected_data_mc
                self.fit()

                params_sel = self.params.loc[:,('par','ass','nucleus','opt')]
                params_sel.loc[:,'mc_idx'] = n
                self.params_mc = pd.concat([self.params_mc,params_sel],ignore_index=True)

                self.selected_data_mc.loc[:,'mc_idx'] = n
                self.selected_data_mc.loc[:,'pH_ini'] = self.selected_data.loc[:,'pH']
                self.selected_data_mc.loc[:,'shift_ini'] = self.selected_data.loc[:,'shift']

                self.data_mc = pd.concat([self.data_mc,self.selected_data_mc],ignore_index=True)
        else:
            self.fitted_data = self.selected_data

        self.fit()

    def data_selection(self):
        appended_data = []
        #select data to fit
        self.res_list = list(self.fit_info.keys())

        for r in self.res_list:
            nuc_list = self.fit_info[r]['nuclei']
            res_data = self.data[(self.data.ass==r)]
            appended_data.append(res_data[res_data.nucleus.isin(nuc_list)])

        self.selected_data = pd.concat(appended_data)
        self.selected_data.reset_index(inplace=True,drop=True)
        if 'Height' in self.selected_data.columns:
            self.selected_data.drop(['Height'],axis=1,inplace=True)
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
                # He1-Ce1 correlations
                'HE':{'ini':7, 'lb': 2, 'ub': 12},
                'C':{'ini':140, 'lb': 130, 'ub': 150},
                # H-N backbone
                'HN':{'ini':7, 'lb': 2, 'ub': 12},
                'N':{'ini':120, 'lb': 100, 'ub': 140},
                # H-N 2J
                'NE2_d_r':{'ini':220, 'lb': 100, 'ub': 250},
                'NE2_d_l':{'ini':220, 'lb': 100, 'ub': 250},

                'ND1_s_r':{'ini':220, 'lb': 100, 'ub': 250},
                'ND1_s_l':{'ini':220, 'lb': 100, 'ub': 250},

                'HE1_d_l':{'ini':7, 'lb': 2, 'ub': 12},
                'HE1_s_l':{'ini':7, 'lb': 2, 'ub': 12},

                'HD2_d_r':{'ini':7, 'lb': 2, 'ub': 12},
                'HD2_s_r':{'ini':7, 'lb': 2, 'ub': 12},
                
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

            for nucleus in self.fit_info[res]['nuclei']:
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
            nuc_res = self.params[self.params.ass==res]
            for s in self.fit_info[res]['nuclei']:
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

    def back_calculated_rmsd(self,df):
        rmsd = np.sqrt(np.mean((df.loc[:,'simulated']-df.loc[:,'shift'].values)**2))
        return rmsd

    def prepare_monte_carlo_analysis(self):
        for index,row in self.selected_data.iterrows():

            # Monte Carlo on pH
            pH_mc = np.random.normal(loc=row['pH'],scale=float(self.errors_mc['pH']))
            # Monte Carlo on Shift
            shift_mc = np.random.normal(loc=float(row['shift']),scale=float(self.errors_mc[str(row.nucleus)]))
            # append value
            self.selected_data_mc.append([row.ass, pH_mc, row.nucleus, shift_mc, row.res_type])

        #Create DataFrame
        self.selected_data_mc = pd.DataFrame(self.selected_data_mc,columns=self.selected_data.columns)

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
            args = (self.fitted_data),
            bounds=bounds
        )
        standard_deviations = self._linear_stats(self.fit_res)

        self.params['opt'] = self.fit_res.x
        self.params['opt_sd'] = standard_deviations
        
        self.fitted_data['simulated'] = self.simulate_shift
        # overall rmsd
        self.rmsd = self.fit_objective(self.params['opt'].values.tolist(),self.fitted_data)
        
    def simulate_for_plot(self, res, nucleus):
        data_res = self.selected_data[(self.selected_data.ass == res) & (self.selected_data.nucleus == nucleus)]
        pH_simulated = np.linspace(min(data_res.pH)-0.1,max(data_res.pH)+0.1,100)    
        if self.mc is True:
            mean_params_mc = self.params_mc.groupby(['par','ass','nucleus'], as_index=False)['opt'].mean()
            std_params_mc = self.params_mc.groupby(['par','ass','nucleus'], as_index=False)['opt'].std()

            for index, row in self.params.iterrows():
                if row.par in ['pKa','pKa_1','pKa_2']:
                    self.params.loc[index,'opt'] = mean_params_mc[(mean_params_mc.loc[:,'par']==row.par)].opt.values[0]
                    self.params.loc[index,'opt_sd'] = std_params_mc[(std_params_mc.loc[:,'par']==row.par)].opt.values[0]

                else:
                    self.params.loc[index,'opt'] = mean_params_mc[(mean_params_mc.loc[:,'ass']==row.ass) & (mean_params_mc.loc[:,'par']==row.par) & (mean_params_mc.loc[:,'nucleus']==row.nucleus)].opt.values[0]
                    self.params.loc[index,'opt_sd'] = std_params_mc[(std_params_mc.loc[:,'ass']==row.ass) & (std_params_mc.loc[:,'par']==row.par) & (std_params_mc.loc[:,'nucleus']==row.nucleus)].opt.values[0]

            self.unpack_fit_parameters(self.params['opt'].values.tolist())
        else:
            pass
        shift_simulated_curve = self.func(pH_simulated,self.fit_par[res][nucleus])
        return pH_simulated, shift_simulated_curve
    
    def data_for_plot(self,res,nucleus):
        data_res = self.selected_data[(self.selected_data.ass == res) & (self.selected_data.nucleus == nucleus)]
        pH_exp = data_res.pH
        shift_exp = data_res.loc[:,'shift']
        return pH_exp,shift_exp

    def plot(self, res, fit: bool = False):
        
        if len(self.nuclei) == 2:
            fig_full = make_subplots(rows=1, cols=2)
        else :
            fig_full = make_subplots(rows=1, cols=1)

        
        data_res = self.fitted_data[self.fitted_data.ass == res]

        ## ---  First nucleus 
        #Experimental data
        data_res_nucleus = data_res[data_res.nucleus=='HN']#self.nuclei[0]
        fig_1 = go.Scatter(x=data_res_nucleus.loc[:,'pH'],y=data_res_nucleus.loc[:,'shift'], name='data: '+str(self.nuclei[0]), mode='markers',marker_color="#EF553B")
        fig_full.add_trace(fig_1, row=1, col=1)

        #Fitted Curve
        if fit:
            pH_all = np.linspace(min(data_res_nucleus.loc[:,'pH'])-0.2,max(data_res_nucleus.loc[:,'pH'])+0.2,100)    
            simulated_curve = self.func(pH_all,self.fit_par[res]['self.nuclei[0]'])
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
    
    def plot_matplotlib(self, res, fit: bool = False, plot = False):

        if plot is False:
            fig, plot = plt.subplots()
            if len(self.nuclei) == 2:
                ax2 = plot.twinx()
        else: 
            ax2 = plot.twinx()


        if self.mc is True: # perform monte-carlo analysis
            mean_params_mc = self.params_mc.groupby(['par','ass','nucleus'], as_index=False)['opt'].mean()
            std_params_mc = self.params_mc.groupby(['par','ass','nucleus'], as_index=False)['opt'].std()

            for index, row in self.params.iterrows():
                if row.par in ['pKa','pKa_1','pKa_2']:
                    self.params.loc[index,'opt'] = mean_params_mc[(mean_params_mc.loc[:,'par']==row.par)].opt.values[0]
                    self.params.loc[index,'opt_sd'] = std_params_mc[(std_params_mc.loc[:,'par']==row.par)].opt.values[0]

                else:
                    self.params.loc[index,'opt'] = mean_params_mc[(mean_params_mc.loc[:,'ass']==row.ass) & (mean_params_mc.loc[:,'par']==row.par) & (mean_params_mc.loc[:,'nucleus']==row.nucleus)].opt.values[0]
                    self.params.loc[index,'opt_sd'] = std_params_mc[(std_params_mc.loc[:,'ass']==row.ass) & (std_params_mc.loc[:,'par']==row.par) & (std_params_mc.loc[:,'nucleus']==row.nucleus)].opt.values[0]

            self.unpack_fit_parameters(self.params['opt'].values.tolist())

        data_res = self.selected_data[self.selected_data.ass == res]

        ## ---  First nucleus 
        #Experimental data
        data_res_nucleus = data_res[data_res.nucleus==self.nuclei[0]]
        plot.plot(data_res_nucleus.loc[:,'pH'],data_res_nucleus.loc[:,'shift'], marker='o',ls='none',color="blue")
        # print(list(data_res_nucleus.loc[:,'nucleus'])[0])
        plot.set_ylabel(r'$\delta$'+str(list(data_res_nucleus.loc[:,'nucleus'])[0])+' (ppm)',color='blue')
        plot.set_xlabel('pH')
        plot.tick_params(axis ='y', labelcolor = 'blue')

        #Fitted Curve
        if fit:
            pH_all = np.linspace(min(data_res_nucleus.loc[:,'pH'])-0.2,max(data_res_nucleus.loc[:,'pH'])+0.2,100)    
            simulated_curve = self.func(pH_all,self.fit_par[res][self.nuclei[0]])
            plot.plot(pH_all, simulated_curve, color="blue")

        ## ---  Second nucleus only if it exists
        #Experimental data
        if len(self.nuclei) == 2:
            data_res_nucleus = data_res[data_res.nucleus==self.nuclei[1]]
            ax2.plot(data_res_nucleus.loc[:,'pH'],data_res_nucleus.loc[:,'shift'],marker='o',ls='none',color="red")
            ax2.set_ylabel(r'$\delta$'+str(list(data_res_nucleus.loc[:,'nucleus'])[0])+' (ppm)',color='red')
            ax2.tick_params(axis ='y', labelcolor = 'red')
        #Fitted Curve
            if fit:
                pH_all = np.linspace(min(data_res_nucleus.loc[:,'pH'])-0.2,max(data_res_nucleus.loc[:,'pH'])+0.2,100)    
                simulated_curve = self.func(pH_all,self.fit_par[res][self.nuclei[1]])
                ax2.plot(pH_all, simulated_curve, color="red")

        if plot is False:
            return fig
