import os
from obspy import core, read, read_inventory, signal
from obspy.signal import filter
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.xseed import Parser
from obspy.signal import PPSD
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
from scipy.signal import csd, coherence
from scipy.optimize import curve_fit
from sets import Set
from obspy.clients.fdsn import Client

from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_sac_taper
    
# Earth tides with amplitudes (vertical or horizontal) > 10 mm

h2s=60.*60.
d2s=23.9345*h2s
y2s=365.24*d2s

'''
earth_tides = {'K2': 11.96723606*h2s, # lunisolar semidiurnal
               'M2': 12.4206012*h2s, # principal lunar semidiurnal
               'N2': 12.65834751*h2s, # larger lunar elliptic semidiurnal
               'S2': 12.*h2s, # principal solar semidiurnal
               'K1': 23.93447213*h2s, # lunar diurnal
               'O1': 25.81933871*h2s, # lunar diurnal
               'P1': 24.06588766*h2s, # solar diurnal
               'phi1': 23.804*h2s, # solar diurnal
               'psi1': 23.869*h2s, # solar diurnal
               'S1': 24.*h2s, # solar diurnal
               'Mf': 13.661*d2s,
               'Lunar month': 27.555*d2s,
               'Solar semi-annual': 0.5*y2s,
               'Lunar node': 18.613*y2s,
               'Solar annual': y2s}
'''

# Constants
G = 6.67408e-11 # m^3/kg/s^2
c = 330. # m/s
g_0 = 9.81 # m/s


# Only major tides (displacements > 10mm)
earth_tides = {'K2': 11.96723606*h2s, # lunisolar semidiurnal
               'M2': 12.4206012*h2s, # principal lunar semidiurnal
               'N2': 12.65834751*h2s, # larger lunar elliptic semidiurnal
               'S2': 12.*h2s, # principal solar semidiurnal
               'K1': 23.93447213*h2s, # lunar diurnal
               'O1': 25.81933871*h2s, # lunar diurnal
               'P1': 24.06588766*h2s, # solar diurnal
               'Mf': 13.661*d2s,
               'Lunar month': 27.555*d2s,
               'Solar semi-annual': 0.5*y2s,
               'Lunar node': 18.613*y2s}

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

def vertical_admittance_twm(c_h, mu, lmda, dgdz=3.e-6):
    # vertical admittance for the travelling wave model (Zurn et al., 2007a)
    # https://doi.org/10.1111/j.1365-246X.2006.03189.x
    # Input parameters: speed of the wave, the Lame parameters mu and lambda
    # and the vertical gradient of gravitational acceleration.
    # The default value for dgdz is that used by Zurn et al.
    # Returns: a function which takes a list of frequencies as input
    def func(frequencies):
        omega = 2.*np.pi*frequencies
        domega = 1e-11
        k_h = omega+domega/c_h
        g_BPM =  2.*np.pi*G / g_0 # eq. 1
        delta_gN = g_BPM / (1. + omega*c*c/(c_h*g_0)) # eq. 14
        delta_gF = -0.5*dgdz/mu*((lmda + 2.*mu)/(lmda + mu))/k_h # eq. 15
        delta_gl = -0.5/mu*((lmda + 2.*mu)/(lmda + mu))*omega*omega/k_h # eq. 16

        delta_g = delta_gN + delta_gF + delta_gl
        return delta_g

    return func

def horizontal_admittance_twm(c_h, mu, lmda):
    # horizontal admittance for the travelling wave model (Zurn et al., 2007b)
    # https://doi.org/10.1111/j.1365-246X.2007.03553.x
    # Input parameters: speed of the wave and the Lame parameters mu and lambda
    # The default value for dgdz is that used by Zurn et al.
    # Returns: a function which takes a list of frequencies as input
    def func(frequencies):
        omega = 2.*np.pi*frequencies
        return ( 2.*np.pi*G/g_0 / (1. + omega*c*c/(c_h*g_0)) +
                 0.5*g_0/mu*((lmda + 2.*mu)/(lmda + mu)) +
                 0.5/mu*(mu/(lmda + mu))*omega*c_h ) 

    return func


def convolve(trace, transfer_function, pre_filt=False):
    # write data to a numpy array
    data = trace.data
    npts = len(data)
    nfft = _npts2nfft(npts)
    
    # Transform data to frequency domain
    data = np.fft.rfft(data, n=nfft)

    # find the frequencies corresponding to each element of the fft
    delta = trace.stats.delta
    fy =  1. / (delta * 2.0)
    # start at zero to get zero for offset/ DC of fft
    frequencies = np.linspace(0, fy, nfft // 2 + 1)

    # pre-filter the data
    if pre_filt:
        freq_domain_taper = cosine_sac_taper(frequencies, flimit=pre_filt)
        data *= freq_domain_taper

    # perform the convolution
    data *= transfer_function(frequencies)
    data[-1] = abs(data[-1]) + 0.0j

    # transform data back into the time domain
    data = np.fft.irfft(data)[0:npts]

    # write data back into trace
    trace.data = data
    
    
def load_data(loc, channels, starttime, endtime, data_directory):

    mseed_file = '{0:s}/data/{1:s}_{2:s}_{3:s}_{4:s}_{5:s}.mseed'.format(data_directory,
                                                                         loc['network'],
                                                                         loc['station'],
                                                                         loc['location'],
                                                                         starttime,
                                                                         endtime)
    xml_file = '{0:s}/inventories/{1:s}_{2:s}_{3:s}_{4:s}_{5:s}.xml'.format(data_directory,
                                                                            loc['network'],
                                                                            loc['station'],
                                                                            loc['location'],
                                                                            starttime,
                                                                            endtime)

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        os.makedirs(data_directory+'/data')
        os.makedirs(data_directory+'/inventories')
        
    if os.path.isfile(mseed_file) and os.path.isfile(xml_file):
        print('Reading data from local files')
        st = read(mseed_file)
        inv = read_inventory(xml_file, format='stationxml')
    else:
        print('Collecting data from IRIS')
        # Fetch waveform from IRIS FDSN web service into a ObsPy stream object
        # and automatically attach correct response
        t1 = UTCDateTime(starttime)
        t2 = UTCDateTime(endtime)

        fdsn_client = Client('IRIS')
        st = fdsn_client.get_waveforms(network=loc['network'], station=loc['station'],
                                       location=loc['location'], channel=channels,
                                       starttime=t1, endtime=t2)
        inv = fdsn_client.get_stations(network=loc['network'], station=loc['station'],
                                       location=loc['location'], channel=channels,
                                       starttime=t1, endtime=t2,
                                       level='response')

        # write mseed and inventory files
        st.write(mseed_file, format='MSEED')
        inv.write(xml_file, format='stationxml')

    st.attach_response(inv)
    return (st, inv)
        
t_bounds = ['2000-07-02T00:00:00', '2000-07-10T00:00:00']
#t_bounds = ['2002-10-13T00:00:00', '2002-10-15T00:00:00']
#t_bounds = ['2005-02-10T00:00:00', '2005-02-13T00:00:00']
st, inv = load_data(loc = {'network': 'II', 'station': 'BFO', 'location':'00'},
                    channels = 'VH?,WDI',
                    starttime = t_bounds[0],
                    endtime = t_bounds[1],
                    data_directory = 'waveform_data')

# In Zuern et al. (2007), the traces are decimated to 20s
# Note that in obspy, the low-pass prefiltering to the decimation is done automatically
st.decimate(1)


# In Zuern et al. (2007), the seismograms are detided
for ch in ['VHZ', 'VHE', 'VHN']:
    tr = st.select(channel=ch)
    detrend_detide(trace=tr[0], period_cutoff=d2s*2.)

# Now let's deal with the pressure records
# In Zuern et al. (2007), the microbarometric record is demeaned
st.select(channel='WDI').detrend(type='linear')

# Remove instrument response using the information from the given RESP file
# seedresp requires the RESP filename and units for the response output ('DIS', 'VEL' or 'ACC')
# define a filter band to prevent amplifying noise during the deconvolution
pre_filt = (0.0001, 0.00012, 0.01, 0.012)
st.remove_response(output='VEL', pre_filt=pre_filt, taper=True, taper_fraction=0.1, water_level=None)


# Zuern et al. (2007) now run an optional low pass filter on all the traces


# Finally, let's create the horizontal and vertical "pressure seismograms"
local_deformation_model = st.select(channel='WDI')[0] * 3
local_deformation_model[0].stats.channel = 'VHZ'
local_deformation_model[1].stats.channel = 'VHE'
local_deformation_model[2].stats.channel = 'VHN'


travelling_wave_model = st.select(channel='WDI')[0] * 3
travelling_wave_model[0].stats.channel = 'VHZ'
travelling_wave_model[1].stats.channel = 'VHE'
travelling_wave_model[2].stats.channel = 'VHN'


c_h = 50. # m/s
mu = 50.e9
lmdaovermu = 1.2 # reasonable approximation for many minerals and rocks (Ji et al., 2010)

lmda = lmdaovermu*mu

convolve(travelling_wave_model.select(channel='VHZ')[0],
         vertical_admittance_twm(c_h, mu, lmda), pre_filt=pre_filt)
convolve(travelling_wave_model.select(channel='VHE')[0],
         horizontal_admittance_twm(c_h, mu, lmda), pre_filt=pre_filt)
convolve(travelling_wave_model.select(channel='VHN')[0],
         horizontal_admittance_twm(c_h, mu, lmda), pre_filt=pre_filt)


# Cut the beginning and end of the time series to avoid edge effects
cuttime=3600.*6. # 6 hours
new_starttime = st[0].stats['starttime'] + cuttime
new_endtime = st[0].stats['endtime'] - cuttime

st = st.slice(starttime=new_starttime, endtime=new_endtime)
st_noise_removed = st.copy()
local_deformation_model = local_deformation_model.slice(starttime=new_starttime, endtime=new_endtime)
travelling_wave_model = travelling_wave_model.slice(starttime=new_starttime, endtime=new_endtime)

print(travelling_wave_model.select(channel='VHE')[0].stats)
print(st.select(channel='VHE')[0].stats)

# Check that sampling rate is the same for the barometer and the seismometers
assert(np.all(np.array([trace.stats['sampling_rate'] == st[0].stats['sampling_rate'] for trace in st])))



def remove_noise(data_trace, model_trace):

    def lsq(x0):
        f = x0[0]
        return np.sum((data_trace.data - f*model_trace.data)*(data_trace.data - f*model_trace.data))

    return lsq

from scipy.optimize import minimize

for ch in ['VHN', 'VHZ', 'VHE']:
    print('Removing noise from channel {0}'.format(ch))
    data_trace = st.select(channel=ch)[0]
    #model_trace = travelling_wave_model.select(channel=ch)[0]
    model_trace = local_deformation_model.select(channel=ch)[0]
    #guess = ( np.sum( np.abs( data_trace.data ) ) /
    #          np.sum( np.abs( model_trace.data ) ) )
    # Here we want an estimate of the admittance (the real part of the transfer function)
    # between the data and model
    guess = 0.
    res = minimize(remove_noise(data_trace, model_trace), [guess], tol=guess*1.e-6)
    f = res.x[0]
    print(guess, f)
    st_noise_removed.select(channel=ch)[0].data = ( data_trace.data -
                                                    f*model_trace.data )


# Plots the traces
st.plot(equal_scale=False)
st_noise_removed.plot(equal_scale=False)
travelling_wave_model.plot(equal_scale=False)
# Plots the probabilistic power spectral density for all records
'''
for ch in ['VHN', 'VHZ', 'VHE', 'WDI']:
    ppsd = PPSD(stats=traces.select(channel=ch)[0].stats,
                metadata=dname+'/RESP.II.BFO.00.'+ch,
                ppsd_length=3600.*24.,
                overlap=0.5)
    ppsd.add(traces.select(channel=ch)[0])
    ppsd.plot(period_lim=(1.e-4, 1.e-1), xaxis_frequency=True)
'''

# Plots the coherence
for ch in ['VHN', 'VHZ', 'VHE']:
    #f, Cxy = coherence(x=st.select(channel=ch)[0].data,
    #                   y=st.select(channel='WDI')[0].data,
    #                   fs=st.select(channel='WDI')[0].stats['sampling_rate'],
    #                   nperseg=8192)
    
    f, Cxy = coherence(x=st.select(channel=ch)[0].data,
                       y=travelling_wave_model.select(channel=ch)[0].data,
                       fs=st.select(channel=ch)[0].stats['sampling_rate'],
                       nperseg=8192)
    
    plt.semilogx(f, Cxy, label=ch)

plt.legend(loc='upper left')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.xlim(1.e-4, 1.e-1)
plt.ylim(0., 1.)
plt.show()

for ch in ['VHN', 'VHZ', 'VHE']:
    #f, Cxy = coherence(x=st.select(channel=ch)[0].data,
    #                   y=st.select(channel='WDI')[0].data,
    #                   fs=st.select(channel='WDI')[0].stats['sampling_rate'],
    #                   nperseg=8192)
    
    f, Cxy = coherence(x=st_noise_removed.select(channel=ch)[0].data,
                       y=travelling_wave_model.select(channel=ch)[0].data,
                       fs=st_noise_removed.select(channel=ch)[0].stats['sampling_rate'],
                       nperseg=8192)
    
    plt.semilogx(f, Cxy, label=ch)

plt.legend(loc='upper left')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.xlim(1.e-4, 1.e-1)
plt.ylim(0., 1.)
plt.show()




    
