import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import periodogram, welch
        
def psdnoise(psd, n_samples, delta):
    '''
    A function to generate noise with a specified power spectrum
    Based on the matlab function fftnoise by Aslak Grinsted

    Input:
    psd: the psd of a time series (must be a column vector)
  
    Output:
    noise: surrogate series with same power spectrum as the input.
    '''
    
    fft = np.array(np.sqrt(psd), dtype='complex')
    Np = (len(fft) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    fft[1:Np+1] *= phases
    fft[-1:-1-Np:-1] = np.conj(fft[1:Np+1])
    
    duration = (n_samples - 1)*delta
    noise = np.fft.ifft(fft).real * (n_samples - 1.) / np.sqrt(2.*duration)
    return noise


def noise_with_piecewise_linear_spectrum(frequencies, psddBs, n_samples=1024, delta=1., scaling='logperiod'):
    '''
    Input:
    frequencies: frequencies at which the PSD is defined
    psddBs: estimates of the PSD (in decibels) at each frequency. psddB[i] = 10.*np.log10(psd[i])
    n_samples: the number of samples in the output trace
    delta: the spacing between data points in the time domain
    scaling: 'logperiod' creates a PSD which is piecewise linear in logperiod-dB space. 
    '''
    
    fft_frequencies = np.abs(np.fft.fftfreq(n_samples, delta))
    fft_frequencies[0] = 1.e-12
    fft_logperiods = np.log10(1./fft_frequencies)
    fft_frequencies[0] = 0.
    
    log_periods = np.log10(1./frequencies)
    
    psd = np.zeros(n_samples)

    if scaling=='logperiod':
        for i in range(1,len(frequencies)):
            logp_max, logp_min = log_periods[i-1:i+1]
            psddB_max, psddB_min = psddBs[i-1:i+1]
            
            indices = np.where(np.logical_and(logp >= logp_min, logp <= logp_max))[0]
            psddB_values = [(logp[idx] - logp_min) /
                            (logp_max - logp_min) *
                            (psddB_max - psddB_min) +
                            psddB_min for idx in indices]

            psd[indices] = np.power(10., np.array(psddB_values) / 10.)
            
    else:
        raise Exception('scaling not implemented')

    times = np.linspace(0., (n_samples - 1.)*delta, n_samples)
    return times, psdnoise(psd, n_samples, delta)


if __name__ == '__main__':
    # Data from Peterson (1993)
    # acceleration low noise model
    
    fig = plt.figure()
    ax_psd = fig.add_subplot(1, 2, 1)
    ax_timeseries = fig.add_subplot(1, 2, 2)
    
    for fname in ['noise_models/Peterson_1993_NHNMacc_high_noise_model.dat',
                  'noise_models/Peterson_1993_NLNMacc_low_noise_model.dat']:
        
        periods, psddBs = np.loadtxt(fname, unpack=True)
        frequencies = 1./periods
        
        ax_psd.plot(np.log10(periods), psddBs, linewidth=3., color='yellow')
        
        for delta in [0.1, 10., 1.]: # Hz
            n_samples = 2048
            
            # Sort
            ind = np.argsort( frequencies )
            periods = periods[ind]
            frequencies = frequencies[ind]
            psddBs = psddBs[ind]
            
            times, acc_noise = noise_with_piecewise_linear_spectrum(frequencies, psddBs, n_samples=n_samples, delta=delta, scaling='logperiod')
            
            fft_frequencies, psd = periodogram(x=acc_noise, fs=1./delta)
            ax_psd.plot(np.log10(1./fft_frequencies[1:]), 10.*np.log10(psd[1:]), linestyle='--')
            
            
            
            
        ax_timeseries.plot(times[:500], acc_noise[:500])
            
    plt.show()
