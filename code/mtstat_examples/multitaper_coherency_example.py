import matplotlib.pyplot as plt
plt.style.use("ggplot")

from mtspec import mt_coherence
import numpy as np

# generate random series with 1Hz sinus inside
np.random.seed(815)
npts = 2048
sampling_rate = 12.8
# one sine wave in one second (sampling_rate samples)
one_hz_sin = np.sin(np.arange(0., sampling_rate) / \
                    sampling_rate * 2. * np.pi)
one_hz_sin = np.tile(one_hz_sin, int(npts // sampling_rate + 1))[:npts]
xi = np.random.randn(npts) + one_hz_sin
xj = np.random.randn(npts) + one_hz_sin
dt, tbp, kspec, nf, p = 1.0/sampling_rate, 3.5, 5, npts/2, .90

# calculate coherency

#    :param df: float; sampling rate of time series
#    :param xi: numpy.ndarray; data for first series
#    :param xj: numpy.ndarray; data for second series
#    :param tbp: float; the time-bandwidth product
#    :param kspec: integer; number of tapers to use
#    :param nf: integer; number of freq points in spectrum
#    :param p:  float; confidence for null hypothesis test, e.g. .95


#    OPTIONAL OUTPUTS, the outputs are returned as dictionary, with keys as
#    specified below and values as numpy.ndarrays. In order to activate the
#    output set the corresponding kwarg in the argument list, e.g.
#    ``mt_coherence(df, xi, xj, tbp, kspec, nf, p, freq=True, cohe=True)``

#    :param freq:     the frequency bins
#    :param cohe:     coherence of the two series (0 - 1)
#    :param phase:    the phase at each frequency
#    :param speci:    spectrum of first series
#    :param specj:    spectrum of second series
#    :param conf:     p confidence value for each freq.
#    :param cohe_ci:  95% bounds on coherence (not larger than 1)
#    :param phase_ci: 95% bounds on phase estimates

# If x is the detided signal and y is the pressure signal:
# Coherency gamma = |gamma|*exp(i*phase) = Sxy/(Sxx * Syy)^0.5
# Transfer function Z = |Z|*exp(i*phase) =  Sxy/Syy
# phaseZ = phasegamma

# Thus,
# |Z| = |gamma| * Sxx^0.5 / Syy^0.5

out = mt_coherence(dt, xi, xj, tbp, kspec, nf, p, freq=True,
                   cohe=True, iadapt=1)

# the plotting part
plt.subplot(211)
plt.plot(np.arange(npts)/sampling_rate, xi)
plt.plot(np.arange(npts)/sampling_rate, xj)
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.subplot(212)
plt.plot(out['freq'], out['cohe'])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Coherency")

plt.tight_layout()
plt.show()
