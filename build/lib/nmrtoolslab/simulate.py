import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

class simulation_spectrum(object):

    def __init__(self,spec_type,field,sweep_width,carrier,peak_list,sparky_name=False,format=True):
        self.spec_type      = spec_type
        self.field          = field
        self.sweep_width    = sweep_width
        self.carrier        = carrier
        self.peak_list      = peak_list
        self.format         = format
        self.sparky_name    = sparky_name

        self.data = []
        self.udic = []
        self.spectrum_simulation()

    def spectrum_simulation(self):
        gamma = {
            'H' :  1.0,                #'H1' 
            'C' :  0.251449530,        #'C13'
            'N' :  0.101329118,        #'N15'
            'P' :  0.404808636,        #'P31'
            'F' :  0.940746805,        #'F19'
        }
        label = {
            'H' :   '1H',
            'N' :   '15N',
            'C' :   '13C',
        }


    #     format='sparky'
        lw={'H':3, 'N':3, 'C':1}
        lineshape=None

        dimensions = str.split(self.spec_type,'-')
        n_dim = len(dimensions)

        obs_frequencies = tuple(self.field*gamma[dimensions[i]] for i in range(n_dim))
        labels  = tuple(label[dimensions[i]] for i in range(n_dim))

        ##create a sparky dictionnary  
        self.udic = {
            'ndim': n_dim,
            0: {'car': obs_frequencies[0]*self.carrier[0],
                'complex': False,
                'encoding': 'states',
                'freq': True,
                'label': labels[0],
                'obs': obs_frequencies[0],
                'size': 512,
                'sw': obs_frequencies[0]*self.sweep_width[0],
                'time': False},
            1: {'car': obs_frequencies[1]*self.carrier[1],
                'complex': False,
                'encoding': 'direct',
                'freq': True,
                'label': labels[1],
                'obs': obs_frequencies[1],
                'size': 1024,
                'sw': obs_frequencies[1]*self.sweep_width[1],
                'time': False}
        }
        dic = ng.sparky.create_dic(self.udic)
        shape = tuple(self.udic[i]['size'] for i in range(n_dim))

        self.data = np.empty(shape, dtype='float32')

        npeaks = len(self.peak_list)

        # if self.format == 'sparky':
        #     # convert the peak list from PPM to points
        uc_objects = [ng.sparky.make_uc(dic, None, i) for i in range(n_dim)]

        line_width = {}
        for i in range(n_dim):
            line_width['lw_'+label[dimensions[i]]] = lw[dimensions[i]]
            # print(labels)

        # Set gaussian lineshape in both dimensions
        lineshape = tuple('g' for i in range(n_dim)) 

        params = []
        for pk in self.peak_list.iterrows():
            ResInfo = pk[1]            
            params_res = []
            for i in range(n_dim):
                params_res.append(
                    (
                    uc_objects[i].f(float(ResInfo[label[dimensions[i]]]),'ppm'),
                    line_width['lw_'+label[dimensions[i]]]
                    ))
            params.append(params_res)

        # simulate the spectrum
        amps = [1e6] * len(self.peak_list)
        self.data = ng.linesh.sim_NDregion(shape, lineshape, params, amps)

        if self.format == 'sparky':
            if self.sparky_name is False:
                file_name = 'test'
            else:
                file_name = self.sparky_name

            ng.sparky.write(str(file_name)+".ucsf", dic, self.data.astype('float32'), overwrite=True)

class exchange_simulation(object):
    def __init__(self,params):
        self.params     =   params
        self.fid_signal =   False
        self.fid_time   =   False        
        self.n_states   =   self.params['n_states']
        self.sw         =   self.params['sw']
        self.td         =   self.params['td']
        self.B0         =   self.params['B0']

        self.dw = 1/(2*self.sw*self.B0) 
        self.fid_time = np.arange(0,self.td*self.dw,self.dw)
        print(self.td*self.dw)
        if self.n_states == 2:
            self.two_site_exchange_simulation()

    def two_site_exchange_simulation(self):
        # population 
        pa = self.params['pa']
        # Exchange ratess
        kex = self.params['kex']
        # Bo field in MHz
        
        # Chemical shifts of both states A and B
        shifts_a = self.params['shifts_a']
        shifts_b = self.params['shifts_b']

        # R1 rates 
        R1_rates = self.params['R1_rates']
        if len(R1_rates) == 1:
            R1a = R1b = R1_rates[0]
        if len(R1_rates) == 2:
            R1a = R1_rates[0]
            R1b = R1_rates[1]

        # R1 rates 
        R2_rates = self.params['R2_rates']
        if len(R2_rates) == 1:
            R2a = R2b = R2_rates[0]
        if len(R2_rates) == 2:
            R2a = R2_rates[0]
            R2b = R2_rates[1]

        pb = 1 - pa
        kba = pa * kex
        kab = pb * kex

        wa = self.B0*2*np.pi*shifts_a
        wb = self.B0*2*np.pi*shifts_b

        H = np.asarray([
        [0,    0, 0, 0, 0, 0, 0], #E
        [0,    -R2a-kab,-wa,0,kba,0,0], #Max
        [0,    wa  ,-R2a-kab,0,0,kba,0], #May
        [2*R1a*pa,0,0,-R1a-kab,0,0,kba], #Maz
        [0,kab,0,0,-R2b-kba,-wb,0], #Mbx
        [0,0,kab,0,wb,-R2b-kba,0], #Mby
        [2*R1b*pb,0,0,kab,0,0,-R1b-kba] #Mbz
         ])

        M0 = np.array([[0.5, 0.0, 0.0, pa, 0.0, 0.0, pb ]]).T

        # blochmat = H*d1
        M = np.dot(expm(H),M0)

        # 90 pulse
        M[1], M[3] = M[3], -M[1]
        M[4], M[6] = M[6], -M[4]

        # aq = self.td/(2*self.sw*B0) 
        
        

        M = expm_multiply(H, M, start=0, stop=self.td*self.dw, num=self.td).transpose(0,2,1)
        detection_vector = np.array([0, 1, 1j, 0, 1, 1j, 0])
        self.signal = np.dot(M, detection_vector).reshape(-1)
        
class Simulation_1D:
    def __init__(self, args):
        self.args = args
        
    def Singlet(self):
        #Lorentzian + Gaussian   
        x = self.args[0]; x0 = self.args[1]; a = self.args[2]; h_s = self.args[3]; lw = self.args[4]
        Signal = a * h_s  / ( 1 + (( x - x0 )/lw)**2) + (1-a)*h_s*np.exp(-(x-x0)**2/(2*lw**2))    
        return Signal

    def Doublet(self):
        x = self.args[0]; x0 = self.args[1]; a = self.args[2]; h_s = self.args[3]; lw = self.args[4]; J1 = self.args[5]
        S1 = a * h_s  / ( 1 + (( x - x0 - (J1/2))/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 - (J1/2))**2/(2*lw**2))
        S2 = a * h_s  / ( 1 + (( x - x0 + (J1/2))/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 + (J1/2))**2/(2*lw**2))
        Signal = S1 + S2
        return Signal

    def DoubletOfDoublet(self):
        x = self.args[0]; x0 = self.args[1]; a = self.args[2]; h_s = self.args[3]; lw = self.args[4]; J1 = self.args[5]; J2 = self.args[6]
        S1 = a * h_s  / ( 1 + (( x - x0 - ((J1+J2)/2))/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 - ((J1+J2)/2))**2/(2*lw**2))
        S2 = a * h_s  / ( 1 + (( x - x0 - ((J1-J2)/2))/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 - ((J1-J2)/2))**2/(2*lw**2))
        S3 = a * h_s  / ( 1 + (( x - x0 + ((J1+J2)/2))/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 + ((J1+J2)/2))**2/(2*lw**2))
        S4 = a * h_s  / ( 1 + (( x - x0 + ((J1-J2)/2))/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 + ((J1-J2)/2))**2/(2*lw**2))
        Signal = S1+S2+S3+S4
        return Signal

    def Triplet(self):
        x = self.args[0]; x0 = self.args[1]; a = self.args[2]; h_s = self.args[3]; lw = self.args[4]; J1 = self.args[5]
        S1 = a * h_s  / ( 1 + (( x - x0 - J1/2)/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 - J1/2)**2/(2*lw**2))
        S2 = a * 2* h_s  / ( 1 + (( x - x0 )/lw)**2) + (1-a)*2*h_s*np.exp(-(x - x0 )**2/(2*lw**2))
        S3 = a * h_s  / ( 1 + (( x - x0 + J1/2)/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 + J1/2)**2/(2*lw**2))
        Signal = S1+S2+S3
        return Signal
    
    def Quadruplet_2J(self):
        x= self.args[0]; x0= self.args[1]; a= self.args[2]; h_s= self.args[3]; lw= self.args[4]; J1= self.args[5]; J2= self.args[6]
        #Lorentzian + Gaussian
        S1 = a * h_s  / ( 1 + (( x - x0 - J2/2)/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 - J2/2)**2/(2*lw**2))
        S2 = a * 3*h_s  / ( 1 + (( x - x0 - J1/2)/lw)**2) + (1-a)*3*h_s*np.exp(-(x - x0 - J1/2)**2/(2*lw**2))
        S3 = a * 3*h_s  / ( 1 + (( x - x0 + J1/2)/lw)**2) + (1-a)*3*h_s*np.exp(-(x - x0 + J1/2)**2/(2*lw**2))
        S4 = a * h_s  / ( 1 + (( x - x0 + J2/2)/lw)**2) + (1-a)*h_s*np.exp(-(x - x0 + J2/2)**2/(2*lw**2))
        Signal = S1+S2+S3+S4
        return Signal