import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np

from multitaper_utils import multitaper_cross_spectral_estimates


# generate random series with 1Hz sinus inside
npts = 8192
delta = 0.1
# one sine wave in one second (sampling_rate samples)
times = np.linspace(0., (npts-1)*delta, npts)

def unit_sine_wave(times, f, phase):
    return np.sin(2.*np.pi*times*f + phase)

amp = 0.05
n_j = 0.05

np.random.seed(100)
xi = np.random.randn(npts) + amp*unit_sine_wave(times, 2., 0.) + 4.*unit_sine_wave(times, 1.5, 0.)
np.random.seed(105)
xj = n_j*np.random.randn(npts) + amp*unit_sine_wave(times, 2., 0.)
traces = np.array([xi, xj])

# Some runs with different variable values produced the following numbers.
# Clearly when noise in xj is low, the admittance estimates are very good,
# even when the coherence is very low. There is no systematic shift in
# variance; the random noise seed on xi governs whether the estimated variance
# is too high or too low
amp = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.2, 0.1, 0.1]
n_j = [1., 1., 1., 1., 1., 1., 0.2, 0.2, 0.1]
cohsqr_2Hz = [0.45, 0.7, 0.8, 0.85, 0.9, 0.93, 0.6, 0.25, 0.4]
adm_2Hz = [0.6, 0.75, 0.8, 0.85, 0.9, 0.93, 0.9, 0.75, 0.9]

est = multitaper_cross_spectral_estimates(traces=traces,
                                          delta=delta,
                                          NW=12,
                                          compute_confidence_intervals=True,
                                          confidence_interval=0.95)

freqs = est['frequencies']
c_bnds = est['confidence_bounds']

fig = plt.figure()

ax = [plt.subplot(2, 2, i) for i in range(1,5)]

for i in range(4):
    ax[i].set_xlabel('Frequency (Hz)')

ax[0].fill_between(freqs, c_bnds['magnitude_squared_coherence'][0],
                   c_bnds['magnitude_squared_coherence'][1], alpha=0.3)
ax[0].plot(freqs, est['magnitude_squared_coherence'])

ax[0].set_xlim(1.5, 2.5)

ax[1].fill_between(freqs, c_bnds['admittance'][0], c_bnds['admittance'][1], alpha=0.3)
ax[1].plot(freqs, est['admittance'])

ax[1].set_xlim(1.5, 2.5)
ax[1].set_ylim(-2., 2.)

ax[2].fill_between(freqs, c_bnds['gain'][0], c_bnds['gain'][1], alpha=0.3)
ax[2].plot(freqs, est['gain'])

#ax[3].fill_between(freqs, c_bnds['phase'][0], c_bnds['phase'][1], alpha=0.3)
#ax[3].plot(freqs, est['phase'])

ax[3].fill_between(freqs,
                   np.zeros_like(freqs) - np.abs(c_bnds['phase'][0] - c_bnds['phase'][1]),
                   np.zeros_like(freqs) + np.abs(c_bnds['phase'][0] - c_bnds['phase'][1]), alpha=0.3)
ax[3].plot(freqs, np.zeros_like(freqs))

ax[3].set_ylim(-180., 180.)
#ax[4].plot(times, xi)
#ax[4].plot(times, xj)



ax[0].set_ylabel('$\gamma^2$')
ax[1].set_ylabel('Admittance')
ax[2].set_ylabel('Gain')
ax[3].set_ylabel('Phase')
plt.show()
