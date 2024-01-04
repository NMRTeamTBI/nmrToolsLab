import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import nmrtoolslab.simulate as simulate
import nmrglue as ng

params = {
    'n_states':2,
    'B0':800, 
    'pa':0.1, 
    'kex':1, 
    'shifts_a':-4,  
    'shifts_b':4, 
    'R1_rates':[1],
    'R2_rates':[0.5],
    'sw':20.7741/2,
    'td':16384,
    'zf':True,
    'td_zf':16384*4}

time = simulate.exchange_simulation(params).fid_time
fid = simulate.exchange_simulation(params).signal

if params['zf'] is True:
    fid = ng.proc_base.zf_size(fid, params['td_zf'])    # zero fill to 32768 points
    dw = 1/(2*params['sw']*params['B0']) 
    time = np.arange(0,params['td_zf']*dw,dw)

# fid = ng.process.proc_base.em(fid,lb=0.3/(params['sw']*params['B0']))

# plt.plot(time,fid)
# plt.show()
# exit()

# _t = np.linspace(0, 1, fid.shape[0])
# cos_bell = np.cos(_t * np.pi / 2)

# fid *= cos_bell
# plt.plot(time,fid)
# plt.show()
# exit()

fid_2 = ng.process.proc_base.em(fid,lb=0.3/(params['sw']*params['B0']))

spec = ng.proc_base.fft(fid)   # Fourier transform
spec = ng.proc_base.di(spec)   # discard the imaginaries
spec = ng.proc_base.rev(spec)  # reverse the data

spec_2 = ng.proc_base.fft(fid_2)   # Fourier transform
spec_2 = ng.proc_base.di(spec_2)   # discard the imaginaries
spec_2 = ng.proc_base.rev(spec_2)  # reverse the data

# data_pc_pm = ng.process.proc_autophase.autops(spec,'acme')

# exit()
# if params['zf'] is True:
#     td = max(params['td'],params['td_zf'])
#     freq = np.fft.fftshift(np.fft.fftfreq(td, params['aq']/params['td'])) / params['B0'] 
# else:
#     freq = np.fft.fftshift(np.fft.fftfreq(params['td'], params['aq']/params['td'])) / params['B0'] 

plt.plot(time,fid)
plt.plot(time,fid_2)
# plt.close()
# plt.plot(spec)
# plt.plot(spec_2)

# plt.plot(data_pc_pm)

plt.show()

exit()
B0              =   800
T1              =   [5,5]
T2              =   [30,30]
# kex             =   10000
pa              =   0.5
shiftsvect      =   [-4,4]
kex_list = [1,3,10,30,100,300,1000,3000,10000,30000,100000,300000]

sig_list = []
for kex in kex_list:    
    sig = Two_state_sim(B0,T1,T2,shiftsvect,kex,pa,0)
    sig_list.append(sig)

t = np.linspace(0, 0.1, sig_list[0].shape[0])
_t = np.linspace(0, 1, sig_list[0].shape[0])

# cos_bell = np.cos(_t * np.pi / 2)
# sig *= cos_bell
# exp_lb = np.exp(-_t * 10 / np.pi)
# sig *= exp_lb

# plt.plot(-_t,exp_lb)
# plt.show()
# exit()

spec_list = []
for i in range(len(sig_list)):
    spec = np.fft.fftshift(np.fft.fft(sig_list[i], n=1024))
    spec_list.append(spec) 
freq = np.fft.fftshift(np.fft.fftfreq(1024, 0.1/1024)) / B0 

# spec = spec[1695:8000]
# freq = freq[1695:8000]

fig,(ax1) = plt.subplots(1,1)

for i in range(len(sig_list)):
    ax1.plot(freq, i*2+(spec_list[i].real/np.max(spec_list[0].real)), lw=0.5)
ax1.invert_xaxis()
# ax1.set_xlim(11.2, 3.8)
plt.show()
