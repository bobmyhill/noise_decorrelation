
import numpy as np
from scipy.optimize import curve_fit
from constants import constants, earth_tides
def make_synthetic_tides(tidal_periods):
    def synthetic_tides(t, *parameters):
        n_periods = len(tidal_periods)
        As = parameters[2:n_periods+2]
        phases = parameters[n_periods+2:]

        y = parameters[0] + parameters[1]*t
        for (period, A, phi) in zip(*[tidal_periods, As, phases]):
            y += A*np.sin(2.*np.pi*t/period + phi)
            
        return y
    return synthetic_tides

def detrend_detide(trace, period_cutoff):
    times = np.linspace(0., (trace.stats['npts']-1.)/trace.stats['sampling_rate'], trace.stats['npts'])
    amplitudes = trace.data
    max_amplitude = np.max(amplitudes)

    amplitudes = amplitudes/max_amplitude
    
    tidal_periods = np.array([v for v in list(earth_tides.values()) if v < period_cutoff])

    # First a first guess at the detrending parameters
    guesses = [0., 0.]
    guesses.extend([0.5 for p in tidal_periods]) # amplitudes first
    guesses.extend([np.pi for p in tidal_periods]) # then phases
    
    max_bounds = [max_amplitude, np.inf] # detrend
    max_bounds.extend([np.inf for p in tidal_periods]) # amplitudes first
    max_bounds.extend([2.*np.pi for p in tidal_periods]) # then phases
    bounds = (0,max_bounds)

    fn = make_synthetic_tides(tidal_periods)
    popt, pcov = curve_fit(fn, times, amplitudes, p0=guesses, bounds=bounds)
    trace.data = (amplitudes - fn(times, *popt))*max_amplitude
    
    #plt.plot(times, amplitudes)
    #plt.plot(times, fn(times, *popt))
    #plt.show()
    
    return 0.
