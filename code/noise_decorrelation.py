import os
import numpy as np
#from sets import Set

#from scipy.signal import csd, coherence

#from obspy import core, read, read_inventory, signal
#from obspy.core.utcdatetime import UTCDateTime
#from obspy.io.xseed import Parser
#from obspy.clients.fdsn import Client
#from obspy.signal import filter, PPSD
#from obspy.signal.util import _npts2nfft
#from obspy.signal.invsim import cosine_sac_taper

import matplotlib.pyplot as plt
from matplotlib import mlab
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = 10, 8 # inches

from read import load_data
from constants import constants
from tides import detrend_detide
from multitaper_utils import multitaper_cross_spectral_estimates, multitaper_cross_spectral_estimates_figure


# Read in data

BFO = {'network': 'II', 'station': 'BFO', 'location':'00'}
H17A = {'network': 'TA', 'station': 'H17A', 'location':'*'} # from 2011-09-22 (for LDV)
KMSC = {'network': 'TA', 'station': 'KMSC', 'location':'*'} # from 2013-03-06 (for LDV)
Y22D = {'network': 'TA', 'station': 'Y22D', 'location':'*'} # from 2011-03-18 (for LDV)

s_ch = ['VHZ', 'VHN', 'VHE']
b_ch = 'WDI'
loc_times = [(BFO, s_ch, b_ch, ['2000-07-02T00:00:00', '2000-07-10T00:00:00']), # Zurn
             (BFO, s_ch, b_ch, ['2002-10-13T00:00:00', '2002-10-15T00:00:00']), # Zurn
             (BFO, s_ch, b_ch, ['2005-02-10T00:00:00', '2005-02-13T00:00:00']), # Zurn
             (BFO, s_ch, b_ch, ['2004-12-16T00:00:00', '2004-12-26T00:00:00']), # pre-Sumatra-Andaman quake
             (BFO, s_ch, b_ch, ['2004-12-26T00:00:00', '2004-12-28T00:00:00'])] # Sumatra-Andaman quake:

s_ch = ['VHZ', 'VHN', 'VHE']
b_ch = 'LDO'
loc_times = [(H17A, s_ch, b_ch, ['2014-03-01T00:00:00', '2014-03-09T00:00:00']),
             (KMSC, s_ch, b_ch, ['2014-03-01T00:00:00', '2014-03-09T00:00:00']),
             (Y22D, s_ch, b_ch, ['2014-03-01T00:00:00', '2014-03-09T00:00:00'])]

loc_times = [(H17A, s_ch, b_ch, ['2015-12-20T00:00:00', '2015-12-30T00:00:00'])]
#(H17A, s_ch, b_ch, ['2014-11-08T00:00:00', '2014-11-18T00:00:00']),

loc_times = []
for name in ['WHTX', '214A', 'TIGA', 'TUL3', 'SFIN', 'P52A', 'KSCO', 'K22A', 'SPMN', 'W18A', 'O53A', 'M65A', 'O48B', 'R11B', 'MDND', 'MSTX', 'Y22D', 'M53A', 'TPFO', 'ABTX', 'H17A', 'BGNE', '121A', 'L44A', 'KMSC', 'SUSD', 'L61B', 'O20A', 'P49A']:
    sta = {'network': 'TA', 'station': name, 'location':'*'}
    loc_times.append((sta, s_ch, b_ch, ['2015-12-20T00:00:00', '2015-12-30T00:00:00']))


for (loc, seis_channels, barometer_channel, t_bounds) in loc_times:
    try:
        print ('Downloading traces for {0}'.format(loc['station']))
        channels = s_ch[0]+','+s_ch[1]+','+s_ch[2]+',LD?,WD?'
        st, inv = load_data(loc = loc,
                            channels = channels,
                            starttime = t_bounds[0],
                            endtime = t_bounds[1],
                            data_directory = 'waveform_data')
        
        print('Number of traces: {0} (should be >=4)'.format(len(st)))
        min_rate = np.min(np.array([tr.stats.sampling_rate for tr in st]))
        # Resample to the highest sampling rate
        # (with automatic low-pass prefiltering)
        st.resample(sampling_rate=min_rate)
        print('Resampled to {0} Hz'.format(min_rate))
        
        # Detide the seismograms and demean the microbarometric record
        for ch in seis_channels:
            tr = st.select(channel=ch)
            detrend_detide(trace=tr[0], period_cutoff=constants['d2s']*2.)
        st.select(channel=barometer_channel).detrend(type='linear')

        # Remove instrument response using the information from the given RESP file
        # seedresp requires the RESP filename and units for the response output ('DIS', 'VEL' or 'ACC')
        # define a filter band to prevent amplifying noise during the deconvolution
        pre_filt = (0.0001, 0.00012, 0.01, 0.012)
        st.remove_response(output='VEL', pre_filt=pre_filt, taper=True, taper_fraction=0.1, water_level=None)

        
        # Cut the beginning and end of the time series to avoid edge effects
        cuttime=3600.*6. # 6 hours
        new_starttime = st[0].stats['starttime'] + cuttime
        new_endtime = st[0].stats['endtime'] - cuttime
    
        st = st.slice(starttime=new_starttime, endtime=new_endtime)
        
        # Calculate cross-spectral estimates of the signals using multitapers
        for ch in seis_channels:
            print('Processing channel {0:s}'.format(ch))
            traces = np.array([st.select(channel=ch)[0].data, st.select(channel=barometer_channel)[0].data])
            delta = 1./st[0].stats['sampling_rate']
            
            est = multitaper_cross_spectral_estimates(traces=traces,
                                                      delta=delta,
                                                      NW=24,
                                                      compute_confidence_intervals=True,
                                                      confidence_interval=0.95)
            
            fig = plt.figure()
            multitaper_cross_spectral_estimates_figure(fig, est,
                                                       frequency_bounds=[1.e-4, 1.e-2],
                                                       log_frequency=True,
                                                       n_octave_y_scaling=2.)
            
            startstamp = '{0:04d}-{1:02d}-{2:02d}T{3:02d}{4:02d}'.format(new_starttime.year,
                                                                         new_starttime.month,
                                                                         new_starttime.day,
                                                                         new_starttime.hour,
                                                                         new_starttime.minute)
            endstamp = '{0:04d}-{1:02d}-{2:02d}T{3:02d}{4:02d}'.format(new_endtime.year,
                                                                       new_endtime.month,
                                                                       new_endtime.day,
                                                                       new_endtime.hour,
                                                                       new_endtime.minute)               
            
            outfile = 'output_figures/{0:s}_{1:s}_{2:s}_{3:s}_{4:s}_{5:s}.pdf'.format(loc['network'],
                                                                                      loc['station'],
                                                                                      loc['location'],
                                                                                      ch,
                                                                                      startstamp,
                                                                                      endstamp)
        
            plt.savefig(outfile)
            print('Figure saved to {0:s}'.format(outfile))   
            #plt.show()
    except:
        pass
