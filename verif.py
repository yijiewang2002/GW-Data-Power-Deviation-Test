# This file verifies the 
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import data_norm
import gwdetectors
from gwdetectors.cache.psd import PSDS
from pycbc.waveform.generator import FDomainCBCGenerator
from pycbc.waveform.generator import TDomainCBCGenerator
from pycbc.detector import Detector



def extract_optimal_waveform(h5_path, posterior_path, likelihood_key, m1_key, m2_key, 
                                spin1_key, spin2_key, distance_key, model_name, s_rate, f_lb, df):
    with h5py.File(h5_path, 'r') as hf:
        # hf.visit(lambda n: print(n))
        posterior = hf[posterior_path] # Navigate to the posterior folder
        max_lh_idx = np.argmax(posterior[likelihood_key]) # Get the index of the maximum likelihood

        m1 = posterior[m1_key][max_lh_idx]
        m2 = posterior[m2_key][max_lh_idx]
        s1z = posterior[spin1_key][max_lh_idx]
        s2z = posterior[spin2_key][max_lh_idx]
        d = posterior[distance_key][max_lh_idx]
        # Generate the wavefrom (expressed in the frequency domain) from the above parameters
        Fgen = FDomainCBCGenerator(variable_args=['mass1', 'mass2', 'spin1z', 'spin2z'], delta_f=df, f_lower=f_lb, approximant=model_name, distance=d)
        hf_p, hf_c = Fgen.generate(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z)

        Tgen = TDomainCBCGenerator(variable_args=["mass1", "mass2", "spin1z", "spin2z"], delta_t=1/s_rate, f_lower=f_lb, approximant=model_name, distance=d)
        ht_p, ht_c = Tgen.generate(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z)
        return ht_p, ht_c


def extract_highestP_coef(h5_path, posterior_path, likelihood_key, ra_key, dec_key, psi_key, gps_time_key, detector_name="H1"):
    with h5py.File(h5_path, 'r') as hf:
        posterior = hf[posterior_path] # Navigate to the posterior folder
        idx = np.argmax(posterior[likelihood_key])

        ra = posterior[ra_key][idx]
        dec = posterior[dec_key][idx]
        psi = posterior[psi_key][idx]
        gps_time = posterior[gps_time_key][idx]
        #print(ra, dec, psi, gps_time)
        det = Detector(detector_name)
        fp, fc = det.antenna_pattern(ra, dec, psi, gps_time)
        return fp, fc


def extract_highestP_SNR(h5_path, posterior_path, opt_snr_key, obs_snr_key): # 
    with h5py.File(h5_path, 'r') as hf:
        posterior = hf[posterior_path] # Navigate to the posterior folder
        obs_SNR = posterior[obs_snr_key] # Extract an array of obsSNR estimation (matched filtering)
        opt_SNR = posterior[opt_snr_key]
        max_obsSNR_idx = np.argmax(obs_SNR) # Get the index of the maximum obs_SNR
        highestP_mf = obs_SNR[max_obsSNR_idx]
        highestP_h_h = opt_SNR[max_obsSNR_idx]**2
        return highestP_h_h, highestP_mf**2 # The first return represents (h,h), and the second (d,h)^2/(h,h)

def calculate_gaus_parameter(N_bin, s_s):
    return 2*N_bin+s_s, np.sqrt(4*N_bin+4*s_s)

if __name__ == '__main__':
    dura = 0.5
    s_rate = 4096
    dt = 1/s_rate
    f_lb = 20
    df = (s_rate/2)/ int( (s_rate*dura/2) )
    f_i, f_e = 10, 512 # default 10
    # f_i, f_e is the range of frequency taken into account in the inner product calculation
    # (distinct from the frequency bounds that are used in waveform generation)
    N_bin = int( (f_e-f_i)/df )
    

    # Obtain plus and cross component of waveform
    ht_p, ht_c = extract_optimal_waveform(sys.argv[1], "C01:IMRPhenomXPHM/posterior_samples", "log_likelihood", "mass_1_source", "mass_2_source", "spin_1z", "spin_2z", "luminosity_distance", "IMRPhenomXPHM", s_rate, f_lb, df)
    h_p = ht_p.numpy()
    h_c = ht_c.numpy()
    # Obtain detector coefficient on the plus and cross mode
    fp, fc = extract_highestP_coef(sys.argv[1], "C01:IMRPhenomXPHM/posterior_samples", "log_likelihood", "ra", "dec", "psi", "geocent_time", "H1")
    h = fp * h_p + fc * h_c

    '''
    # Take the slice containing the merger peak
    # h_slice = data_norm.setup_slice_data(h, s_rate, dura, 122.77) # 0.25 version
    h_slice = data_norm.setup_slice_data(h, s_rate, dura, 122.52) # 0.5 version 122.51
    # h_slice = data_norm.setup_slice_data(h, s_rate, dura, 122.01) # 1.0 version

    plt.figure(figsize=(20,15))
    x_scal = np.linspace(0, len(h_slice), len(h_slice)) / s_rate
    plt.plot(x_scal, h_slice)
    plt.show()
    
    fq_arr, h_slice_f = data_norm.fft(data_norm.multiply_window(h_slice, 'hann'), dt)

    # Get noise psd
    noise_dura = 500
    strain = data_norm.extract_strain_data(sys.argv[2], "strain/Strain")
    S_f = data_norm.get_noisePSD(strain, noise_dura, s_rate, fq_arr, dura)

    # Calculate h_h
    h_h = data_norm.compute_weighted_inner(h_slice_f, S_f, fq_arr, f_i, f_e)



    val_arr = np.linspace(0, 1600, 1600)
    pdf_vals = scipy.stats.norm.pdf(val_arr, 2*N_bin+h_h, np.sqrt(4*N_bin+4*h_h) ) #df stands for degree of freedom
    '''

    '''
    plt.figure()
    plt.plot(val_arr, pdf_vals)
    plt.show()
    print('(h,h) is', h_h)
    print('Best waveform mean is', 2*N_bin+h_h)
    print('sigma is', np.sqrt(4*N_bin+4*h_h) )
    '''


    ideal, mfSNR_2 = extract_highestP_SNR(sys.argv[1], "C01:IMRPhenomXPHM/posterior_samples", "L1_optimal_snr", "L1_matched_filter_snr")
    mean, sigma = calculate_gaus_parameter(N_bin, ideal) # Alternatively feed in mfSNR_2
    print('For this event, the obsSNR is', np.sqrt(mfSNR_2))
    print('For this event, the obsSNR squared is', mfSNR_2)
    print('The optSNR squared is', ideal)
    print('The predicted mean is', mean, ', sigma is', sigma)

    val_arr = np.linspace(0, 2.0*mean, 1000)
    pdf_vals = scipy.stats.norm.pdf(val_arr, mean, sigma) #df stands for degree of freedom
''' recover
    plt.figure()
    plt.plot(val_arr, pdf_vals)
    plt.show()
'''





# cd C:\Users\fengz\OneDrive\Desktop\research\"2025 summer"\"model simulation"
# python verif.py ..\GW150914\IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5

# on linux
# python verif.py ../GW150914/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5
# python verif.py ../GW151226/IGWN-GWTC2p1-v2-GW151226_033853_PEDataRelease_mixed_cosmo.h5
# python verif.py ../GW170817/GW170817_GWTC-1.hdf5
# python verif.py ../GW190412/IGWN-GWTC2p1-v2-GW190412_053044_PEDataRelease_mixed_cosmo.h5
# python verif.py ../GW190521/IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_cosmo.h5
# python verif.py ../GW190814/IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_cosmo.h5
