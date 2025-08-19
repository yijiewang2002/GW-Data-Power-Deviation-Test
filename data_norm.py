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
    offsource_strain = strain[3000 * sample_rate : (3000+noise_duration) * sample_rate] # Take the first given seconds of strain data ## ADD 10 sec
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
    #1 Strain data #2 PE sample
    strain = extract_strain_data(sys.argv[1], "strain/Strain")
    s_rate, tot = 4096, 4096
    f_low = 20
    dt = 1 / s_rate
    f_i, f_e = 10, 512 # default 10 512
    
    m1, m2, z, gps_time = extract_optimal_para(sys.argv[2], "C01:IMRPhenomXPHM/posterior_samples", "log_likelihood", "mass_1_source", "mass_2_source", "redshift", "geocent_time")
    gps_sample_start = extract_starting_gps_time(sys.argv[1], "strain/Strain")
    strain_peak, dura = autolocate_peak(strain, s_rate, m1, m2, z, f_low, gps_time, gps_sample_start) # Default value is 0.5 second ####
    # dura * sample_rate = number of data points in a slice = segment length in noise psd = psd segment factor * sample_rate 
    dlt_f = (s_rate/2)/(int( s_rate*dura )/2) # Nyquist frequency over fft frequency numbers
    N_bin = N_bin = int( (f_e-f_i)/dlt_f )
    print("The estimated duration is", dura)
    print("The frequency resolution is", dlt_f)
    print("The number of frequency bins from", f_i, "Hz to", f_e, "Hz is", N_bin)
    print("After", gps_time-gps_sample_start, "sec, event started")
    
    noise_dura = 500 # default 500
    '''
    # this part is to verify the accuracy of the noise model (n,n)
    N = 300 # Number of slices
    # Use parallel programing to compute the inner prodcut
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda i: complete_call(
                strain,
                setup_slice_data(strain, s_rate, 0.5, 1 * i + 3200),
                s_rate, 'hann', noise_dura, 0.5, f_i, f_e
            ),
        range(N)
        )) ### below
        arr_test = list(executor.map(
            lambda i: fft(multiply_window(setup_slice_data(strain, s_rate, 0.5, 1 * i + 3200), 'hann'), 
                          0.5),
        range(N)
        ))

    N_bin = int( (f_e-f_i)/2.0 )
    inner_est_arr = np.array(results, dtype=object)
    est_vals = np.linspace(min(inner_est_arr), max(inner_est_arr), 1000)
    pdf_vals = scipy.stats.chi2.pdf(est_vals, df=2 * N_bin) #df stands for degree of freedom
    cdf_vals = scipy.stats.chi2.cdf(est_vals, df=2 * N_bin)

    # pdf_vals_2 = scipy.stats.gamma.pdf(est_vals, a=N_bin, loc=0, scale=2*dlt_f) # c chi, scale = 2c
    # plt.plot(est_vals, pdf_vals_2, 'g--', label="$\Gamma(N_{fq bin}, c)$")

    plt.hist(inner_est_arr, bins=15, density=True, alpha=0.6, label="(d=n,d=n)")
    plt.plot(est_vals, pdf_vals, 'r--', label="$\chi^2_{2 N_{fq bin} }$")
    plt.xlabel("Values of (d,d) for off-source slices with a 0.5 seconds duration")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    '''

    # plot psd and strain data (peak/merger part) against frequency
    # strain_peak = setup_slice_data(strain, s_rate, 0.25, 2179.88) #0.25 version
    # strain_peak = setup_slice_data(strain, s_rate, 0.5, 2179.72) #0.5 version
    # strain_peak = setup_slice_data(strain, s_rate, 1.0, 2179.5) #1.0 version
    # plot the selected slice
    '''recover when weird event need double check  
    x_scal = np.linspace(0, len(strain_peak), len(strain_peak)) / s_rate
    plt.plot(x_scal, strain_peak)
    plt.show()
    '''    

    inner_peak = complete_call(strain, strain_peak, s_rate, 'hann', noise_dura, dura, f_i, f_e)
    print('-------------------------------------')
    print('The integral estimate for (d,d) is', inner_peak)


    '''
    # This part is for ploting the psd and fft 
    freq_peak_fft, strain_peak_fft = fft(multiply_window(strain_peak, 'hann'), dt)
    strain_peak_fft_2 = np.abs(strain_peak_fft)**2
    noise_psd = get_noisePSD(strain, noise_dura, s_rate, freq_peak_fft, 1/2)
    df = freq_peak_fft[1] - freq_peak_fft[0]


    plt.figure()
    plt.loglog(freq_peak_fft, strain_peak_fft_2, linewidth=0.5, color='blue', label='strain Fourier coefficient')
    plt.loglog(freq_peak_fft, noise_psd, linewidth=0.5, color='orange', label='noise psd')
    plt.legend()
    plt.show()
    '''


    ''' 
    # This part is to check if individual frequency modes satisfy the model
    ls_arr = np.array(arr_test, dtype=object)
    lee = len(ls_arr)
    da_2_arr = np.zeros(lee)
    for i in range(lee):
        da_2_arr[i] = 4 * np.abs(ls_arr[i][1][27])**2 / noise_psd[27] * dlt_f



    plt.hist(da_2_arr, bins=20, density=True, alpha=0.6)
    freq_spec_vals = np.linspace(min(da_2_arr), max(da_2_arr), 1000)
    #pdf_da = scipy.stats.chi2.pdf(freq_spec_vals, df=1)
    plt.plot(freq_spec_vals, pdf_da)
    plt.show()
    '''


# cd C:\Users\fengz\OneDrive\Desktop\research\"2025 summer"\"model simulation"
# python data_norm.py ..\GW150914\H1_strain_32s_4096Hz.hdf5
# python data_norm.py ..\GW150914\H1_strain_4096s_4096Hz.hdf5
# python data_norm.py ..\GW150914\H1_strain_4096s_16384Hz.hdf5

# on linux      # python data_norm.py ../GW150914/H1_strain_4096s_4096Hz.hdf5 ../GW150914/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5
# python data_norm.py ../GW150914/H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5 ../GW150914/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5
# python data_norm.py ../GW151226/H-H1_GWOSC_4KHZ_R1-1135134303-4096.hdf5 ../GW151226/IGWN-GWTC2p1-v2-GW151226_033853_PEDataRelease_mixed_cosmo.h5
# python data_norm.py ../GW170817/H-H1_GWOSC_4KHZ_R1-1187006835-4096.hdf5 ../GW170817/GW170817_GWTC-1.hdf5
# python data_norm.py ../GW190412/H-H1_GWOSC_4KHZ_R1-1239080215-4096.hdf5 ../GW190412/IGWN-GWTC2p1-v2-GW190412_053044_PEDataRelease_mixed_cosmo.h5
# python data_norm.py ../GW190521/H-H1_GWOSC_4KHZ_R1-1242440920-4096.hdf5 ../GW190521/IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_cosmo.h5
# python data_norm.py ../GW190814/H-H1_GWOSC_4KHZ_R1-1249850209-4096.hdf5 ../GW190814/IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_cosmo.h5
# python data_norm.py ../GW190814/L-L1_GWOSC_4KHZ_R1-1249850209-4096.hdf5 ../GW190814/IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_cosmo.h5