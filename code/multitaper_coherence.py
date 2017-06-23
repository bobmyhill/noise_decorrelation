
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np

import scipy.stats.distributions as dist
from scipy import fftpack

from nitime import utils
import nitime.algorithms as alg



# generate random series with 1Hz sinus inside
npts = 8192
sampling_rate = 12.8
# one sine wave in one second (sampling_rate samples)
times = np.linspace(0., (npts-1)/sampling_rate, npts)

def unit_sine_wave(times, f, phase):
    return np.sin(2.*np.pi*times*f + phase)

np.random.seed(815)
xi = np.random.randn(npts) + 2.*unit_sine_wave(times, 2., 0.) + 4.*unit_sine_wave(times, 1.5, 0.)
np.random.seed(101)
xj = np.random.randn(npts) + unit_sine_wave(times, 2., 0.)
traces = np.array([xi, xj])

dt = 1./sampling_rate

# Define the number of tapers, their values and associated eigenvalues:

NW = 24
K = 2 * NW - 1

tapers, eigs = alg.dpss_windows(npts, NW, K)


"""
Multiply the data by the tapers, calculate the Fourier transform
We multiply the data by the tapers and derive the fourier transform and the
magnitude of the squared spectra (the power) for each tapered time-series:

"""

tdata = tapers[None, :, :] * traces[:, None, :]
tspectra = fftpack.fft(tdata)


'''
The coherency for real sequences is symmetric so only half
the spectrum if required
'''
L = npts // 2 + 1

if L < npts:
    freqs = np.linspace(0, 1. / (2. * dt), L)
else:
    freqs = np.linspace(0, 1. / dt, L, endpoint=False)
    
'''
Estimate adaptive weighting of the tapers, based on the data 
(see Thomsen, 2007; 10.1109/MSP.2007.4286561)
'''
w = np.empty((2, K, L))
for i in range(2):
    w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides='onesided')



'''
Calculate the multi-tapered cross spectrum 
and the PSDs for the two time-series:
'''

sxy = alg.mtm_cross_spectrum(tspectra[0], tspectra[1],
                             (w[0], w[1]), sides='onesided')
sxx = alg.mtm_cross_spectrum(tspectra[0], tspectra[0],
                             w[0], sides='onesided')
syy = alg.mtm_cross_spectrum(tspectra[1], tspectra[1],
                             w[1], sides='onesided')

'''
Coherence is : $Coh_{xy}(\lambda) = \frac{|{f_{xy}(\lambda)}|^2}{f_{xx}(\lambda) \cdot f_{yy}(\lambda)}$
'''
magnitude_squared_coherence = np.abs(sxy) ** 2 / (sxx * syy)
Z = sxy/syy # Transfer function
admittance = np.real(Z)
gain = np.absolute(Z)
phase = np.angle(Z, deg=True)


fig = plt.figure()

ax = [plt.subplot(3, 2, i) for i in range(1,7)]

for i in range(4):
    ax[i].set_xlabel('Frequency (Hz)')

ax[0].plot(freqs, magnitude_squared_coherence)
ax[1].plot(freqs, admittance)
#ax[1].plot(freqs, admittance2)
           
ax[2].plot(freqs, gain)
ax[3].plot(freqs, phase)

ax[4].plot(times, xi)
ax[4].plot(times, xj)



ax[0].set_ylabel('$\gamma^2$')
ax[1].set_ylabel('Admittance')
ax[2].set_ylabel('Gain')
ax[3].set_ylabel('Phase')
plt.show()
        
'''
The variance from the different samples is calculated using a jack-knife approach:
'''

coh_var = utils.jackknifed_coh_variance(tspectra[0], tspectra[1], eigs, adaptive=True)

'''
The coherence is normalized, based on the number of tapers:
'''

coh_norm = utils.normalize_coherence(magnitude_squared_coherence, 2 * K - 2)


'''
Calculate 95% confidence intervals based on the jack-knife variance estimate:
'''

t025_limit = coh_norm + dist.t.ppf(.025, K - 1) * np.sqrt(coh_var)
t975_limit = coh_norm + dist.t.ppf(.975, K - 1) * np.sqrt(coh_var)

plt.plot(freqs, coh_norm)
plt.plot(freqs, t025_limit)
plt.plot(freqs, t975_limit)
plt.show()

utils.normal_coherence_to_unit(t025_limit, 2 * K - 2, t025_limit)
utils.normal_coherence_to_unit(t975_limit, 2 * K - 2, t975_limit)




