import scipy
import numpy as np
import matplotlib.pyplot as plt
import data_norm
import gwdetectors
from gwdetectors.cache.psd import PSDS
import scipy.stats
import random


# Generate an instance of noise
def noise_instance(psd, freq_arr):
    n_f = psd.draw_noise(freq_arr)
    return n_f

def signal_instance(signal, dt, freq_arr):
    freq_fft, signal_fft= data_norm.fft(signal, dt)
    interp_sig = scipy.interpolate.interp1d(freq_fft, signal_fft, kind='linear', bounds_error=False, fill_value=np.inf)
    full_signal_fft = interp_sig(freq_arr)
    return full_signal_fft

# Generate an instance of an event by superposing random noise to a predetermined signal
def data_instance(s_f, n_f):
    d_f = s_f + n_f
    return d_f

def specific_sig_f(f, A, tu, f0):
    s_f = A * tu * np.sqrt(2*np.pi) / (2*1j) * (np.e**(-2 * np.pi**2 * tu**2 * (f-f0)**2) - np.e**(-2 * np.pi**2 * tu**2 * (f+f0)**2))
    return s_f

def generate_simulated_residual(fq_arr, f_i, f_e):
    A = random.uniform(0.5, 2.5)*10e-23
    f_0 = random.uniform(95,105)
    tau = random.uniform(0.08, 0.12)
    s_f = specific_sig_f(fq_arr, A, tau, f_0)

    psd = PSDS['aligo-design']
    S_f = psd.__call__(fq_arr)

    s_s = data_norm.compute_weighted_inner(s_f, S_f, fq_arr, f_i, f_e)
    s_s_sr = np.sqrt(s_s)

    n_f = noise_instance(psd, fq_arr)
    d_f = data_instance(s_f, n_f)
    d_d = data_norm.compute_weighted_inner(d_f, S_f, fq_arr, f_i, f_e)
    rho_obs = data_norm.compute_weighted_dotproduct(d_f, s_f, S_f, fq_arr, f_i, f_e) / s_s_sr
    return d_d, s_s, rho_obs


if __name__ == '__main__':
# Define a frequency array
    df = 2.0
    fq_arr = np.arange(0, 1025, df)
    f_i, f_e = 10, 512 # Default 10, 512
    N_bin = int( (f_e-f_i)/df )
    
    
# Load the psd, noise and signal frequency modes
    psd = PSDS['aligo-design']
    S_f = psd.__call__(fq_arr)

    simu_len = 3000
    d_d = np.zeros(simu_len)
    rho_obs = np.zeros(simu_len)
    s_s = np.zeros(simu_len)

    for i in range(simu_len):
        d_d[i], s_s[i], rho_obs[i] = generate_simulated_residual(fq_arr, f_i, f_e)
    s_s_sr = np.sqrt(s_s)
    reduced_obs = (d_d - 2 * N_bin)/rho_obs**2
    reduced_opt = (d_d - 2 * N_bin)/s_s
    residual_opt = d_d - 2*N_bin - s_s
    residual_obs = d_d - 2*N_bin - rho_obs**2

    rho_obs_val = np.linspace(min(rho_obs), max(rho_obs), 1000)
    s_s_sr_val = np.linspace(min(s_s_sr), max(s_s_sr), 1000)
    

    '''
    # This part is for mock data test
    A = 10e-23
    tau = 0.1 # default 0.1
    f_0 = 100 # default 100
    d_d1 = np.zeros(simu_len*2)
    s_f1 = specific_sig_f(fq_arr, A, tau, f_0)
    s_s1 = data_norm.compute_weighted_inner(s_f1, S_f, fq_arr, f_i, f_e)
    print(s_s1)
    for i in range(simu_len*2):
        n_f = noise_instance(psd, fq_arr)
        d_f = data_instance(s_f1, n_f)
        d_d1[i] = data_norm.compute_weighted_inner(d_f, S_f, fq_arr, f_i, f_e)

    d_d1_val = np.linspace(min(d_d1), max(d_d1), 1000)
    mean1 = 2*N_bin+s_s1
    sigma1 = np.sqrt(4*N_bin+4*s_s1)
    pdf_vals = scipy.stats.norm.pdf(d_d1_val, mean1, sigma1) #df stands for degree of freedom

    plt.hist(d_d1, bins=100, density=True, alpha=0.6, label="(d,d)")
    plt.plot(d_d1_val, pdf_vals, 'r--', label="$\mu = 2N_{bin} + (s,s)$, $\sigma^2 = 4N_{bin} + 4(s,s)$")
    plt.xlabel("Values of (d,d)")
    plt.ylabel("Normalized Count")
    plt.legend()
    plt.show()
    '''



    
    event_name = np.array(["GW150914", "GW151226", "GW190412", "GW190521", "GW190814"])
    event_d_d = np.array([1179.8667375710907, 5071.358282649014, 2465.7013408534717, 90.57397852105045, 10758.944044792652])
    event_rho_obs = np.array([20.8734236116, 10.847913827419442, 9.818075659653823, 8.126939780443676, 11.608135282243486])
    event_h_h = np.array([435.69981326895686, 103.87660729448118, 115.22174813523385, 67.5365348226725, 149.8390955823494])
    event_rho_opt = np.sqrt(event_h_h)
    event_N_bin = np.array([383, 2493, 1191, 26, 4971])
    event_sigma = np.sqrt(4 * event_N_bin + 4 * event_h_h)
    zero = np.zeros(len(event_d_d))


    plt.figure(figsize=(8, 6))
    plt.scatter(rho_obs, reduced_obs, s=0.7, label='obs SNR-normalized noise-reduced power')
    #plt.plot(rho_obs_val, up_spread, linestyle='-', color = 'orange')
    #plt.plot(rho_obs_val, lw_spread, linestyle='-', color = 'orange')
    plt.xlabel(r' $\rho_{obs}$ ')
    plt.ylabel(r'[(d,d) - 2$N_{bin}$]/ $\rho_{obs}^2$ ')

    event_reduced_obs = (event_d_d-2*event_N_bin)/event_rho_obs**2
    
    plt.scatter(event_rho_obs, event_reduced_obs, color='red', s=2.0, label='events')
    plt.legend()
    plt.show()



    

# h_h residual plot
    plt.figure(figsize=(8, 6))
    sigma = np.sqrt( 4*N_bin + 4*(s_s_sr_val**2) )
    sigma2 = 2*sigma
    line0 = np.zeros(len(s_s_sr_val))

    plt.scatter(s_s_sr, residual_opt, s=0.7, alpha=0.5, label='residual of simulated events')
    plt.plot(s_s_sr_val, sigma2, linestyle='-', color = 'green')
    plt.plot(s_s_sr_val, -sigma2, linestyle='-', color = 'green')
    plt.plot(s_s_sr_val, line0, linestyle='--', color = 'grey')
    plt.xlabel(r' $\rho_{opt}$ ')
    plt.ylabel(r'(d,d) - 2$N_{bin}$ - (s,s) ')

    event_residual = event_d_d - 2*event_N_bin - event_h_h
    plt.scatter(event_rho_opt, event_residual, color='red', s=6.0, label='residual of real events')
    for i in range(len(event_rho_opt)):
        plt.text(event_rho_opt[i], event_residual[i], event_name[i], fontsize=6, ha='right', va='top')
    plt.errorbar(event_rho_opt, zero, yerr=2*event_sigma, fmt='.', color="orange", label='expected deviation')
    plt.legend()
    plt.show()


# obs SNR residual plot
    plt.figure(figsize=(8.5, 6.5))
    #plt.figure()
    sigma = np.sqrt( 4*N_bin + 4*(rho_obs_val**2) )
    sigma3 = 3*sigma
    line0 = np.zeros(len(rho_obs_val))

    plt.scatter(rho_obs, residual_obs, s=0.7, alpha=0.3, label='residual of simulated events')
    plt.plot(rho_obs_val, sigma3, linestyle='-', color = 'green', label='expected deviation of simulated events')
    plt.plot(rho_obs_val, -sigma3, linestyle='-', color = 'green')
    plt.plot(rho_obs_val, line0, linestyle='--', color = 'grey')
    plt.xlabel(r' $\rho_{obs}$ ')
    plt.ylabel(r'(d,d) - 2$N_{bin}$ - $\rho_{obs}^2$ ')

    event_residual = event_d_d - 2*event_N_bin - event_rho_obs**2
    plt.scatter(event_rho_obs, event_residual, color='red', s=6.0, label='residual of real events')
    for i in range(len(event_rho_obs)):
        plt.text(event_rho_obs[i], event_residual[i], event_name[i], fontsize=8, ha='right', va='top')
    plt.errorbar(event_rho_obs, zero, yerr=3*event_sigma, fmt='.', color="orange", label='expected deviation of real events')
    plt.legend()
    plt.show()
    

# cd C:\Users\fengz\OneDrive\Desktop\research\"2025 summer"\"model simulation"
# python data_simulation.py ..\GW150914\H1_strain_4096s_4096Hz.hdf5

# on linux
# python data_simulation.py ../GW150914/H1_strain_4096s_4096Hz.hdf5