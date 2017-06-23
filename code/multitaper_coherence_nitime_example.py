
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np

import scipy.stats.distributions as dist
from scipy import fftpack

from nitime import utils
import nitime.algorithms as alg


def jackknifed_variances(tx, ty, eigvals, adaptive=True, deg=True):
    """
    Returns the variance of the admittance (real-part), 
    gain (modulus) and phase of the transfer function and 
    gamma^2 (modulus-squared coherence) between x and y, 
    estimated through jack-knifing the tapered samples in {tx, ty}.

    Parameters
    ----------

    tx : ndarray, (K, L)
       The K complex spectra of tapered timeseries x
    ty : ndarray, (K, L)
       The K complex spectra of tapered timeseries y
    eigvals : ndarray (K,)
       The eigenvalues associated with the K DPSS tapers

    Returns
    -------

    jk_var : dictionary of ndarrays 
       (entries are 'admittance', 'gain', 'phase', 
       'magnitude_squared_coherence')
       The variance computed in the transformed domain
    """

    K = tx.shape[0]

    # calculate leave-one-out estimates of the admittance
    jk_admittance = []
    jk_gain = []
    jk_phase = []
    jk_magnitude_squared_coherence = []
    sides = 'onesided'
    all_orders = set(range(K))

    import nitime.algorithms as alg

    # get the leave-one-out estimates
    for i in range(K):
        items = list(all_orders.difference([i]))
        tx_i = np.take(tx, items, axis=0)
        ty_i = np.take(ty, items, axis=0)
        eigs_i = np.take(eigvals, items)
        if adaptive:
            wx, _ = utils.adaptive_weights(tx_i, eigs_i, sides=sides)
            wy, _ = utils.adaptive_weights(ty_i, eigs_i, sides=sides)
        else:
            wx = wy = eigs_i[:, None]
        # The CSD
        sxy_i = alg.mtm_cross_spectrum(tx_i, ty_i, (wx, wy), sides=sides)
        # The PSDs
        sxx_i = alg.mtm_cross_spectrum(tx_i, tx_i, wx, sides=sides)
        syy_i = alg.mtm_cross_spectrum(ty_i, ty_i, wy, sides=sides)
        
        # these are the Zr_i samples
        Z = sxy_i / syy_i
        jk_admittance.append ( np.real(Z) )
        jk_gain.append ( np.absolute(Z) )
        jk_phase.append ( np.angle(Z, deg=deg) )
        jk_magnitude_squared_coherence.append( np.abs(sxy_i) ** 2 / (sxx_i * syy_i) )


        
    # The jackknifed variance is equal to
    # (K-1)/K * sum_i ( (x_i - mean(x_i))^2 )
    jk_var = {}
    for (name, jk_variance) in [('admittance', np.array(jk_admittance)),
                                ('gain', np.array(jk_gain)),
                                ('phase', np.array(jk_phase)),
                                ('magnitude_squared_coherence',
                                 np.array(jk_magnitude_squared_coherence))]:
        jk_avg = np.mean(jk_variance, axis=0)
        jk_var[name] = (float(K - 1.) / K) * ( np.power( (jk_variance - jk_avg) , 2. ) ).sum(axis=0)

    return jk_var


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


# Estimate 95% confidence intervals
variances = jackknifed_variances(tspectra[0], tspectra[1], eigs, adaptive=True)
admittance_limits = [admittance + dist.t.ppf(.025, K - 1) * np.sqrt(variances['admittance']),
                     admittance + dist.t.ppf(.975, K - 1) * np.sqrt(variances['admittance'])]
gain_limits = [gain + dist.t.ppf(.025, K - 1) * np.sqrt(variances['gain']),
               gain + dist.t.ppf(.975, K - 1) * np.sqrt(variances['gain'])]
phase_limits = [phase + dist.t.ppf(.025, K - 1) * np.sqrt(variances['phase']),
                phase + dist.t.ppf(.975, K - 1) * np.sqrt(variances['phase'])]
magnitude_squared_coherence_limits = [magnitude_squared_coherence + dist.t.ppf(.025, K - 1) *
                                      np.sqrt(variances['magnitude_squared_coherence']),
                                      magnitude_squared_coherence + dist.t.ppf(.975, K - 1) *
                                      np.sqrt(variances['magnitude_squared_coherence'])]

fig = plt.figure()

ax = [plt.subplot(2, 2, i) for i in range(1,5)]

for i in range(4):
    ax[i].set_xlabel('Frequency (Hz)')

ax[0].fill_between(freqs, magnitude_squared_coherence_limits[0], magnitude_squared_coherence_limits[1], alpha=0.3)
ax[0].plot(freqs, magnitude_squared_coherence)

ax[1].fill_between(freqs, admittance_limits[0], admittance_limits[1], alpha=0.3)
ax[1].plot(freqs, admittance)

ax[2].fill_between(freqs, gain_limits[0], gain_limits[1], alpha=0.3)
ax[2].plot(freqs, gain)


#ax[3].fill_between(freqs, phase_limits[0], phase_limits[1], alpha=0.3)
#ax[3].plot(freqs, phase)

ax[3].fill_between(freqs,
                   np.zeros_like(freqs) - np.abs(phase_limits[0] - phase_limits[1]),
                   np.zeros_like(freqs) + np.abs(phase_limits[0] - phase_limits[1]), alpha=0.3)
ax[3].plot(freqs, np.zeros_like(freqs))

ax[3].set_ylim(-180., 180.)
#ax[4].plot(times, xi)
#ax[4].plot(times, xj)



ax[0].set_ylabel('$\gamma^2$')
ax[1].set_ylabel('Admittance')
ax[2].set_ylabel('Gain')
ax[3].set_ylabel('Phase')
plt.show()
