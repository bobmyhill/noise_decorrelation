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



def multitaper_cross_spectral_estimates(traces, delta, NW,
                                        compute_confidence_intervals=True,
                                        confidence_interval=0.95):

    # Define the number of tapers, their values and associated eigenvalues:
    npts = len(traces[0])
    K = 2 * NW - 1
    tapers, eigs = alg.dpss_windows(npts, NW, K)


    # Multiply the data by the tapers, calculate the Fourier transform
    # We multiply the data by the tapers and derive the fourier transform and the
    # magnitude of the squared spectra (the power) for each tapered time-series:
    tdata = tapers[None, :, :] * traces[:, None, :]
    tspectra = fftpack.fft(tdata)


    # The coherency for real sequences is symmetric so only half
    # the spectrum if required
    L = npts // 2 + 1

    if L < npts:
        freqs = np.linspace(0, 1. / (2. * delta), L)
    else:
        freqs = np.linspace(0, 1. / delta, L, endpoint=False)
    
        
    # Estimate adaptive weighting of the tapers, based on the data 
    # (see Thomsen, 2007; 10.1109/MSP.2007.4286561)
    w = np.empty((2, K, L))
    for i in range(2):
        w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides='onesided')


    # Calculate the multi-tapered cross spectrum 
    # and the PSDs for the two time-series:
    sxy = alg.mtm_cross_spectrum(tspectra[0], tspectra[1],
                                 (w[0], w[1]), sides='onesided')
    sxx = alg.mtm_cross_spectrum(tspectra[0], tspectra[0],
                                 w[0], sides='onesided')
    syy = alg.mtm_cross_spectrum(tspectra[1], tspectra[1],
                                 w[1], sides='onesided')


    Z = sxy/syy
    
    spectral_estimates = {}
    spectral_estimates['frequencies'] = freqs
    spectral_estimates['magnitude_squared_coherence'] = np.abs(sxy) ** 2 / (sxx * syy)
    spectral_estimates['transfer_function'] = Z # Transfer function
    spectral_estimates['admittance'] = np.real(Z)
    spectral_estimates['gain'] = np.absolute(Z)
    spectral_estimates['phase'] = np.angle(Z, deg=True)
    

    # Estimate confidence intervals
    if compute_confidence_intervals:
        spectral_estimates['confidence_bounds'] = {}
        c_bnds = [0.5 - confidence_interval/2., 0.5 + confidence_interval/2.]
        variances = jackknifed_variances(tspectra[0], tspectra[1], eigs, adaptive=True)
        spectral_estimates['confidence_bounds']['admittance'] = [spectral_estimates['admittance'] + dist.t.ppf(c_bnds[0], K - 1) * np.sqrt(variances['admittance']),
                                                      spectral_estimates['admittance'] + dist.t.ppf(c_bnds[1], K - 1) * np.sqrt(variances['admittance'])]
        spectral_estimates['confidence_bounds']['gain'] = [spectral_estimates['gain'] + dist.t.ppf(c_bnds[0], K - 1) * np.sqrt(variances['gain']),
                                                spectral_estimates['gain'] + dist.t.ppf(c_bnds[1], K - 1) * np.sqrt(variances['gain'])]
        spectral_estimates['confidence_bounds']['phase'] = [spectral_estimates['phase'] + dist.t.ppf(c_bnds[0], K - 1) * np.sqrt(variances['phase']),
                                                 spectral_estimates['phase'] + dist.t.ppf(c_bnds[1], K - 1) * np.sqrt(variances['phase'])]
        spectral_estimates['confidence_bounds']['magnitude_squared_coherence'] = [spectral_estimates['magnitude_squared_coherence'] + dist.t.ppf(c_bnds[0], K - 1) *
                                                                       np.sqrt(variances['magnitude_squared_coherence']),
                                                                       spectral_estimates['magnitude_squared_coherence'] + dist.t.ppf(c_bnds[1], K - 1) *
                                                                       np.sqrt(variances['magnitude_squared_coherence'])]

    
    return spectral_estimates

def multitaper_cross_spectral_estimates_figure(fig, est, frequency_bounds=[-1., 1.e20], log_frequency=True, n_octave_y_scaling=None):
    c_bnds = est['confidence_bounds']
    
    ax = [fig.add_subplot(2, 2, i) for i in range(1,5)]


    freqs = est['frequencies']
    
    mask = [idx for (idx, f) in enumerate(freqs) if frequency_bounds[0] < f and f < frequency_bounds[1]]

    if n_octave_y_scaling == None:
        scale_mask = mask
    else:
        scale_mask = [idx for (idx, f) in enumerate(freqs) if frequency_bounds[0] < f and f < frequency_bounds[0]*np.power(2., n_octave_y_scaling)]
    
    for i in range(4):
        ax[i].set_xlabel('Frequency (Hz)')
        
        if log_frequency:
            ax[i].set_xscale("log", nonposx='clip')

    # Plot magnitude-squared coherence
    ax[0].fill_between(freqs[mask], c_bnds['magnitude_squared_coherence'][0][mask],
                       c_bnds['magnitude_squared_coherence'][1][mask], alpha=0.3)
    ax[0].plot(freqs[mask], est['magnitude_squared_coherence'][mask])

    ax[0].set_ylim(0., 1.)
            
    # Plot admittance and gain
    for (i, name) in [(1, 'admittance'), (2, 'gain')]:
        ax[i].fill_between(freqs[mask], c_bnds[name][0][mask], c_bnds[name][1][mask], alpha=0.3)
        ax[i].plot(freqs[mask], est[name][mask])
        
        lim = np.max(np.abs(np.array([c_bnds[name][0][scale_mask],
                                      c_bnds[name][1][scale_mask]])))
        lim *= np.mean(est[name][scale_mask])/np.abs(np.mean(est[name][scale_mask]))
        if lim < 0.:
            ax[i].set_ylim(1.1*lim, -0.2*lim)
        else:
            ax[i].set_ylim(-0.2*lim, 1.1*lim)

    # Plot phase (potentially quite noisy
    ax[3].fill_between(freqs[mask],
                       np.zeros_like(freqs[mask]) -
                       np.abs(c_bnds['phase'][0][mask] -
                              c_bnds['phase'][1][mask]),
                       np.zeros_like(freqs[mask]) +
                       np.abs(c_bnds['phase'][0][mask] -
                              c_bnds['phase'][1][mask]), alpha=0.3)
    ax[3].plot(freqs[mask], np.zeros_like(freqs[mask]))
    
    ax[3].set_ylim(-180., 180.)


    ax[0].set_ylabel('$\gamma^2$')
    ax[1].set_ylabel('Admittance')
    ax[2].set_ylabel('Gain')
    ax[3].set_ylabel('Phase')

    return fig

