import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt


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


