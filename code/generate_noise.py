import numpy as np

def fftnoise(f):
    '''
    A function to generate noise with a specified power spectrum
    Based on the matlab function by Aslak Grinsted

    Input:
    f: the fft of a time series (must be a column vector)
  
    Output:
    noise: surrogate series with same power spectrum as f.
    '''
    
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    noise = np.fft.ifft(f).real
    return noise


def noise_with_piecewise_linear_spectrum(vertices, n_samples=1024, samplerate=1.):
    f = np.abs(np.fft.fftfreq(n_samples, 1/samplerate))
    fft = np.zeros(samples)
    for i in range(1:len(vertices)):
        f_min, v_min = vertices[i-1]
        f_max, v_max = vertices[i]
        
        indices = np.where(np.logical_and(f >= f_min, f <= f_max))[0]
        values = [np.power(10., (f[idx] - f_min)/(f_max - f_min) * (np.log10(v_max) - np.log10(v_min))) for idx in indices]
        fft[indices] = values
    return fftnoise(fft)
