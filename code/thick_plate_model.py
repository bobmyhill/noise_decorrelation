import numpy as np
import matplotlib.pyplot as plt

def displacement(omega, z, c, lmda, mu, rho, g, H):
    k = omega/c

    xi = rho*g*(lmda + 2.*mu)/(4.*mu*(lmda + mu)*k)
    
    delta = ( (0.5 - xi) * np.exp(-2.*k*H) +
              (0.5 + xi) * np.exp(2.*k*H) -
              1. + 4.*xi*k*H - 2.*k*k*H*H )
    D = 1./delta * (0.5 - xi + k*H + (-0.5 + xi)*np.exp(-2.*k*H))
    C = 1./delta * (-0.5 - xi + k*H + (0.5 + xi)*np.exp(2.*k*H))
    B = 0.5 * ( 1 - C - D)
    A = 1. - B
    
    u_x = 1./(4.*mu*(lmda + mu)*k)*( (lmda + 2.*mu)*( (A - 2.*C) * np.exp(-k*z) +
                                                      (B + 2.*D) * np.exp(k*z) + 
                                                      C * k * z * np.exp(-k*z) + 
                                                      D * k * z * np.exp(k*z) ) +
                                     lmda * ( A * np.exp(-k*z) +
                                              B * np.exp(k*z) +
                                              C * k * z * np.exp(-k*z) +
                                              D * k * z * np.exp(k*z) ) )

    u_z = 1./(4.*mu*(lmda + mu)*k)*( (lmda + 2.*mu)*( (A + C) * np.exp(-k*z) +
                                                      (-B + D) * np.exp(k*z) + 
                                                      C * k * z * np.exp(-k*z) - 
                                                      D * k * z * np.exp(k*z) ) +
                                     lmda * ( (A - C) * np.exp(-k*z) -
                                              (B + D) * np.exp(k*z) +
                                              C * k * z * np.exp(-k*z) -
                                              D * k * z * np.exp(k*z) ) )


    return (u_x, u_z)

omega = np.logspace(-4, -2., 101)
c=40.
for mu in np.linspace(10.e9, 100.e9, 10):
    lmda = mu
    u_x, u_z = displacement(omega, 0., c, lmda, mu, 2600., 10., 20.e3)
    plt.semilogx(omega, u_z, label='x')
    #plt.semilogx(omega, 10*np.log10(u_z), label='z c: '+str(int(c)))

plt.ylabel('Amplitude (dB)')
plt.legend(loc='upper right')
plt.show()
exit()
