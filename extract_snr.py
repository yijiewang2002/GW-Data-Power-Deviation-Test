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
    optSNR_2, obsSNR_2 = extract_highestP_SNR(sys.argv[1], "C01:IMRPhenomXPHM/posterior_samples", "L1_optimal_snr", "L1_matched_filter_snr")

    print('For this event, the obsSNR is', np.sqrt(obsSNR_2))
    print('For this event, the obsSNR squared is', obsSNR_2)
    print('For this event, the optSNR squared is', optSNR_2)



# Example Call
#1 argument PE sample
# python extract_snr.py ../GW190814/IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_cosmo.h5
