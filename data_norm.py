# This file calculates the norm of a data weighted by a noise
import h5py
import sys
import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
import scipy
from concurrent.futures import ThreadPoolExecutor  # Or ProcessPoolExecutor


# Extract the SNR data from the h5 file, the strain path refers to the location of the strain dataset
def extract_strain_data(h5_path, strain_path): # strain_path, _key
    try:
        with h5py.File(h5_path, 'r') as hf:
            strain_data = hf[strain_path][()]
            return strain_data

    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_starting_gps_time(h5_path, strain_path):
     with h5py.File(h5_path, 'r') as hf:
        gps_sample_start = hf[strain_path].attrs['Xstart']
        return gps_sample_start

def extract_optimal_para(h5_path, posterior_path, likelihood_key, m1_key, m2_key, z_key,
                            time_key):
    with h5py.File(h5_path, 'r') as hf:
        # hf.visit(lambda n: print(n))
        posterior = hf[posterior_path] # Navigate to the posterior folder
        max_lh_idx = np.argmax(posterior[likelihood_key]) # Get the index of the maximum likelihood

        m1 = posterior[m1_key][max_lh_idx]
        m2 = posterior[m2_key][max_lh_idx]
        z = posterior[z_key][max_lh_idx]
        gps_time = posterior[time_key][max_lh_idx]
        return m1, m2, z, gps_time

# Multiply a window function to the data, window has the same size as data
def multiply_window(data, window_name):
    window = scipy.signal.get_window(window_name, len(data))
    norm_factor = np.sqrt(np.mean(window**2))
    return data * window / norm_factor

# Perform fast Fourier transform to the data
def fft(data, dt):
    # The fft for non-negative frequencies.
    data_fft = np.fft.rfft(data)
    # The frequencies range from 0 to Nyquist frequency (sample_rate / 2)
    freq_fft = np.fft.rfftfreq(len(data), dt)
    # Renormalzie the sum to approximate the integral
    data_fft *= dt
    return freq_fft, data_fft

def get_noisePSD(strain, noise_duration, sample_rate, freq_array, segment_factor): # Noise duration is in seconds, Sample rate is in number/seconds
    offsource_strain = strain[2200 * sample_rate : (2200+noise_duration) * sample_rate] # Take the first given seconds of strain data ## ADD 10 sec
    # Obtain the psd of the noises
    freq, psd = scipy.signal.welch(offsource_strain, sample_rate, nperseg=segment_factor*sample_rate, noverlap=segment_factor*sample_rate/2)
    # Interpolate the noise psd
    # The line below return a function which approximates the psd(freq) relation
    
    interp_psd = scipy.interpolate.interp1d(freq, psd, kind='linear', bounds_error=False, fill_value=np.inf)
    
    full_noise_psd = interp_psd(freq_array)
    return full_noise_psd

def compute_weighted_inner(d, S, fq_arr, freq_i, freq_e):
    # d refers to the spectral amplitude of the data
    # S refers to the psd of the noise
    # d and S shall have the same length
    df = fq_arr[1] - fq_arr[0]
    i = np.searchsorted(fq_arr, freq_i, side='right')
    e = np.searchsorted(fq_arr, freq_e, side='right')
    d2 = np.sum( np.abs(d[i:e])**2 / S[i:e] )
    d2 *= (4 * df)
    return d2

def compute_weighted_dotproduct(a, b, S, fq_arr, freq_i, freq_e):
    # d refers to the spectral amplitude of the data
    # S refers to the psd of the noise
    # d and S shall have the same length
    df = fq_arr[1] - fq_arr[0]
    i = np.searchsorted(fq_arr, freq_i, side='right')
    e = np.searchsorted(fq_arr, freq_e, side='right')
    a_b = np.sum(  (a[i:e].real * b[i:e].real + a[i:e].imag * b[i:e].imag) / S[i:e]  )
    a_b *= (4 * df)
    return a_b

def setup_slice_data(strain, sample_rate, duration, start_time):
    end_time = start_time + duration
    i_idx, e_idx = int(start_time * sample_rate), int(end_time * sample_rate)
    strain_slice = strain[i_idx:e_idx]
    return strain_slice


# Assume m1 and m2 in solar mass
def autolocate_peak(strain, sample_rate, m1, m2, z, f_low, gps_time, gps_sample_start):
    # Calculate the duration of the merger using given parameters
    m1 *= 1.988416e30 * (1+z)
    m2 *= 1.988416e30 * (1+z)
    M_c = (m1*m2)**(3/5) / (m1+m2)**(1/5)
    c = cst.c
    G = cst.G
    # Duration until merger from f_low (in seconds)
    dura_est = 5 / 256 * c**5 * (G * M_c)**(-5/3) * (np.pi * f_low)**(-8/3)

    inspiral_start = gps_time - gps_sample_start - dura_est * 0.7
    peak_slice = setup_slice_data(strain, sample_rate, dura_est, inspiral_start)
    return peak_slice, dura_est


def complete_call(strain, strain_slice, sample_rate, window_name, noise_sample_duration, psd_segment_factor, freq_i=10, freq_e=512):
    # Ensure the segment length for psd is the same as the data length fed into fft
    dt = 1/sample_rate
    freq_slice_fft, slice_fft = fft(multiply_window(strain_slice, window_name), dt)
    noise_psd = get_noisePSD(strain, noise_sample_duration, sample_rate, freq_slice_fft, psd_segment_factor)
    print(np.any(noise_psd <= 0))
    print(np.any(np.isnan(noise_psd )) )
    inner_estimate = compute_weighted_inner(slice_fft, noise_psd, freq_slice_fft, freq_i, freq_e)
    #print('The integral estimate for (d,d) is', inner_estimate)
    return inner_estimate




if __name__ == '__main__':
    strain = extract_strain_data(sys.argv[1], "strain/Strain")
    s_rate, tot = 4096, 4096
    f_low = 20
    dt = 1 / s_rate
    f_i, f_e = 10, 512 # default 10 512
    noise_dura = 500 # default 500
    
    m1, m2, z, gps_time = extract_optimal_para(sys.argv[2], "C01:IMRPhenomXPHM/posterior_samples", "log_likelihood", "mass_1_source", "mass_2_source", "redshift", "geocent_time")
    gps_sample_start = extract_starting_gps_time(sys.argv[1], "strain/Strain")
    strain_peak, dura = autolocate_peak(strain, s_rate, m1, m2, z, f_low, gps_time, gps_sample_start)
    # dura * sample_rate = number of data points in a slice = segment length in noise psd = psd segment factor * sample_rate 
    dlt_f = (s_rate/2)/(int( s_rate*dura )/2) # Nyquist frequency over fft frequency numbers
    N_bin = N_bin = int( (f_e-f_i)/dlt_f )
    print("The estimated duration is", dura)
    print("The frequency resolution is", dlt_f)
    print("The number of frequency bins from", f_i, "Hz to", f_e, "Hz is", N_bin)
    print("After", gps_time-gps_sample_start, "seconds, event started")
    
    
    inner_peak = complete_call(strain, strain_peak, s_rate, 'hann', noise_dura, dura, f_i, f_e)
    print('-------------------------------------')
    print('For this event, the integral estimate for (d,d) is', inner_peak)

# Example call
#1 argument Strain data #2 argument PE sample
# python data_norm.py ../GW190814/L-L1_GWOSC_4KHZ_R1-1249850209-4096.hdf5 ../GW190814/IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_cosmo.h5