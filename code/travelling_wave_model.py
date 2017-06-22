import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

G = 6.67408e-11 # m^3/kg/s^2
c = 330. # m/s
g_0 = 9.81 # m/s

def vertical_admittance_twm(c_h, mu, lmda, dgdz=3.e-6):
    # vertical admittance for the travelling wave model (Zurn et al., 2007a)
    # https://doi.org/10.1111/j.1365-246X.2006.03189.x
    # Input parameters: speed of the wave, the Lame parameters mu and lambda
    # and the vertical gradient of gravitational acceleration.
    # The default value for dgdz is that used by Zurn et al.
    # Returns: a function which takes a list of frequencies as input
    def func(frequencies):
        omega = 2.*np.pi*frequencies
        k_h = omega/c_h
        g_BPM =  2.*np.pi*G / g_0 # eq. 1
        delta_gN = g_BPM / (1. + omega*c*c/(c_h*g_0)) # eq. 14
        delta_gF = -0.5*dgdz/mu*((lmda + 2.*mu)/(lmda + mu))/k_h # eq. 15
        delta_gl = -0.5/mu*((lmda + 2.*mu)/(lmda + mu))*omega*omega/k_h # eq. 16
        return delta_gN + delta_gF + delta_gl

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


# Plot vertical and horizontal da/dp
fig = plt.figure()

# Plot figures from paper
ax_vim = fig.add_subplot(1, 2, 1)
fig6 = mpimg.imread('figures/Zurn_2007_vertical_tfn_fig6.png')
ax_vim.imshow(fig6, extent=[0, 1, 0, 1], aspect='auto')
ax_vim.set_xlim(0,1)
ax_vim.set_ylim(0,1)
ax_vim.set_xticks([])
ax_vim.set_yticks([])

ax_him = fig.add_subplot(1, 2, 2)
fig7 = mpimg.imread('figures/Zurn_2007_horizontal_tfn_fig7.png')
ax_him.imshow(fig7, extent=[0, 1, 0, 1], aspect='auto')
ax_him.set_xlim(0,1)
ax_him.set_ylim(0,1)
ax_him.set_xticks([])
ax_him.set_yticks([])


# Plot calculated values for the magnitude of the vertical transfer function
ax_v = fig.add_subplot(1, 2, 1, frame_on=False)
frequencies = np.logspace(-4, -1, 101)
lmdaovermu = 1.2 # reasonable approximation for many minerals and rocks (Ji et al., 2010)
lmdaovermu = 1.0 # Assumption in Zurn et al. (2007 a, b)

for c_h in [10., 330.]:
    for mu in [20.e9, 50.e9, 100.e9]:
        lmda = lmdaovermu*mu
        ax_v.loglog(frequencies, 100.*np.abs(vertical_admittance_twm(c_h, mu, lmda)(frequencies)))
        #ax_v.loglog(frequencies, 100.*vertical_admittance_twm(c_h, mu, lmda)(frequencies))

ax_v.set_xlim(1.5e-4, 4.5e-2)
ax_v.set_ylim(7.e-11, 2e-7)


# Plot calculated values for the magnitude of the horizontal transfer function
ax_h = fig.add_subplot(1, 2, 2, frame_on=False)

frequencies = np.logspace(-6, -1, 101)
for c_h in [10., 80., 330.]:
    for mu in [20.e9, 100.e9]:
        lmda = lmdaovermu*mu
        ax_h.loglog(frequencies, 1.e9*100.*horizontal_admittance_twm(c_h, mu, lmda)(frequencies))

ax_h.set_xlim(1.e-6, 1.e-1)
ax_h.set_ylim(7., 300.)

plt.show()


def convolve(trace, func):
    
    from obspy.signal.util import _npts2nfft
    from obspy.signal.invsim import cosine_sac_taper
    
    data = trace.data
    delta = trace.stats.delta
    
    npts = len(data)
    nfft = _npts2nfft(npts)
    
    # Transform data to Frequency domain
    data = np.fft.rfft(data, n=nfft)

    fy =  1. / (delta * 2.0)
    # start at zero to get zero for offset/ DC of fft
    freqs = np.linspace(0, fy, nfft // 2 + 1)
    freq_response = func(freqs)
    
    if pre_filt:
        freq_domain_taper = cosine_sac_taper(freqs, flimit=pre_filt)
        data *= freq_domain_taper
        
    '''
    if water_level is None:
        # No water level used, so just directly invert the response.
        # First entry is at zero frequency and value is zero, too.
        # Just do not invert the first value (and set to 0 to make sure).
        freq_response[0] = 0.0
        freq_response[1:] = 1.0 / freq_response[1:]
    else:
        # Invert spectrum with specified water level.
        invert_spectrum(freq_response, water_level)
    '''

    
    data *= freq_response
    data[-1] = abs(data[-1]) + 0.0j

    # transform data back into the time domain
    data = np.fft.irfft(data)[0:npts]
    
